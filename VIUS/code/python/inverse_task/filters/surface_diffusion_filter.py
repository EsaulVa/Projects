#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_diffusion_filter.py
===========================
Пространственная фильтрация геометрии оправки (поверхности вращения)
на основе уравнения диффузии (теплопроводности) в метрике поверхности.

Поддерживает независимые граничные условия Дирихле, Неймана и Робина
на левом и правом концах меридиана.

Опционально: сохранение параметризации по осевой координате z
(для совместимости с CAD и построения графиков r(z)).
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.linalg import solve_banded
from dataclasses import dataclass
from typing import Optional
import csv
from pathlib import Path


@dataclass
class BoundaryCondition:
    """Граничное условие для конца меридиана."""
    kind: str
    value: float = 0.0
    coeff_a: float = 1.0
    coeff_b: float = 0.0

    @classmethod
    def dirichlet(cls, value: float):
        return cls('dirichlet', value=value)

    @classmethod
    def neumann(cls, flux: float = 0.0):
        return cls('neumann', value=flux)

    @classmethod
    def robin(cls, coeff_a: float, coeff_b: float, value: float):
        return cls('robin', value=value, coeff_a=coeff_a, coeff_b=coeff_b)

    def with_value(self, new_value: float):
        return BoundaryCondition(self.kind, new_value, self.coeff_a, self.coeff_b)


class DiffusedRevolutionSurface:
    """
    Сглаженная поверхность вращения, построенная диффузией профиля меридиана.

    По умолчанию параметр u — длина дуги меридиана s (от 0 до s_max).
    При preserve_z_parameter=True дополнительно доступна параметризация по z.
    """

    def __init__(self, base_surface, N=800, tau=10.0, n_steps=20,
                 bc_left: Optional[BoundaryCondition] = None,
                 bc_right: Optional[BoundaryCondition] = None,
                 save_meridian_path: Optional[str] = None,
                 preserve_z_parameter: bool = False):
        """
        preserve_z_parameter : если True, строит дополнительные интерполяторы
                               r(z) и s(z) для доступа по осевой координате.
        """
        self.base = base_surface
        self.N = N
        self.tau = tau
        self.n_steps = n_steps
        self.bc_left = bc_left if bc_left is not None else BoundaryCondition.neumann(0.0)
        self.bc_right = bc_right if bc_right is not None else BoundaryCondition.neumann(0.0)
        self.preserve_z = preserve_z_parameter

        # --- 1. Извлечь меридиан в исходном параметре u (u = z для PiecewisePolynomialRevolution) ---
        u_raw = np.linspace(base_surface.u_min, base_surface.u_max, N)
        pts = np.array([base_surface.position(u, 0.0) for u in u_raw])
        self._r_raw = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        self._z_raw = pts[:, 2]
        self._u_grid_z = u_raw  # исходная сетка по z

        # --- 2. Перепараметризация по длине дуги s ---
        ds = np.sqrt(np.diff(self._r_raw)**2 + np.diff(self._z_raw)**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        self.u_min = 0.0
        self.u_max = float(s[-1])
        self._s_grid = s

        # --- 3. Диффузия r(s) и z(s) ---
        bc_l_r = (self.bc_left.with_value(self._r_raw[0])
                  if self.bc_left.kind == 'dirichlet' else self.bc_left)
        bc_r_r = (self.bc_right.with_value(self._r_raw[-1])
                  if self.bc_right.kind == 'dirichlet' else self.bc_right)
        bc_l_z = (self.bc_left.with_value(self._z_raw[0])
                  if self.bc_left.kind == 'dirichlet' else self.bc_left)
        bc_r_z = (self.bc_right.with_value(self._z_raw[-1])
                  if self.bc_right.kind == 'dirichlet' else self.bc_right)

        r_smooth = self._diffuse(s, self._r_raw, tau, n_steps, bc_l_r, bc_r_r)
        z_smooth = self._diffuse(s, self._z_raw, tau, n_steps, bc_l_z, bc_r_z)
        self._r_smooth = r_smooth
        self._z_smooth = z_smooth

        # --- 4. Интерполяторы по длине дуги s (основные) ---
        self._cs_r = CubicSpline(s, r_smooth)
        self._cs_z = CubicSpline(s, z_smooth)
        self._cs_r_deriv = self._cs_r.derivative(1)
        self._cs_z_deriv = self._cs_z.derivative(1)
        self._cs_r_deriv2 = self._cs_r.derivative(2)
        self._cs_z_deriv2 = self._cs_z.derivative(2)

        # --- 5. Опционально: интерполяторы по осевой координате z ---
        if self.preserve_z:
            # Проверяем монотонность z(s) — необходима для обратной интерполяции
            dz_ds = np.gradient(z_smooth, s)
            if np.any(dz_ds <= 0):
                # Если z(s) не монотонна, используем только возрастающий участок
                # или выдаём предупреждение
                print("[DiffusedRevolutionSurface] ВНИМАНИЕ: z(s) не монотонна. "
                      "Интерполяция s(z) может быть неоднозначной.")
            # Строим интерполяторы s(z) и r(z) на сглаженном профиле
            # Используем interp1d с kind='cubic' (CubicSpline требует монотонности)
            z_min, z_max = z_smooth.min(), z_smooth.max()
            self._z_min = z_min
            self._z_max = z_max
            # Сортируем по z для корректной интерполяции
            order = np.argsort(z_smooth)
            z_sorted = z_smooth[order]
            s_sorted = s[order]
            r_sorted = r_smooth[order]
            # Убираем дубли z (если есть горизонтальный участок)
            z_unique, idx = np.unique(z_sorted, return_index=True)
            self._cs_s_by_z = interp1d(z_unique, s_sorted[idx], kind='cubic',
                                       bounds_error=False, fill_value=(s[0], s[-1]))
            self._cs_r_by_z = interp1d(z_unique, r_sorted[idx], kind='cubic',
                                        bounds_error=False, fill_value=(r_smooth[0], r_smooth[-1]))
            # Также сохраняем сплайн z(s) для производных
            self._cs_z_inv = self._cs_s_by_z  # alias

        # --- 6. Сохранение меридиана в CSV (опционально) ---
        if save_meridian_path is not None:
            self._save_meridian_to_csv(save_meridian_path, s)

    # ------------------------------------------------------------------
    # Доступ по осевой координате z (только при preserve_z=True)
    # ------------------------------------------------------------------
    def s_from_z(self, z: float) -> float:
        """Возвращает длину дуги s для заданной осевой координаты z."""
        if not self.preserve_z:
            raise RuntimeError("preserve_z_parameter=False. Используйте position(s,v).")
        return float(self._cs_s_by_z(np.clip(z, self._z_min, self._z_max)))

    def radius(self, z: float) -> float:
        """Возвращает радиус r для заданной осевой координаты z."""
        if not self.preserve_z:
            raise RuntimeError("preserve_z_parameter=False. Используйте position(s,v).")
        return float(self._cs_r_by_z(np.clip(z, self._z_min, self._z_max)))

    def position_by_z(self, z: float, v: float) -> np.ndarray:
        """Точка на сглаженной поверхности по осевой координате z и углу v."""
        r = self.radius(z)
        return np.array([r * np.cos(v), r * np.sin(v), z])

    def derivatives_by_z(self, z: float, v: float) -> dict:
        """Производные по параметрам (z, v) — для совместимости с CAD."""
        s = self.s_from_z(z)
        r = float(self._cs_r(s))
        dr_ds = float(self._cs_r_deriv(s))
        dz_ds = float(self._cs_z_deriv(s))
        # dr/dz = (dr/ds) / (dz/ds)
        if abs(dz_ds) < 1e-14:
            dr_dz = 0.0
        else:
            dr_dz = dr_ds / dz_ds
        # Базисы по (z, v)
        ru = np.array([dr_dz * np.cos(v), dr_dz * np.sin(v), 1.0])
        rv = np.array([-r * np.sin(v), r * np.cos(v), 0.0])
        n = np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / norm
        return {"r": np.array([r * np.cos(v), r * np.sin(v), z]),
                "ru": ru, "rv": rv, "normal": n}

    # ------------------------------------------------------------------
    # Сохранение меридиана
    # ------------------------------------------------------------------
    def _save_meridian_to_csv(self, path: str, s: np.ndarray):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                's', 'z_orig', 'r_orig', 'z_smooth', 'r_smooth',
                'dr_orig', 'dz_orig', 'dr_smooth', 'dz_smooth',
                'curvature_orig', 'curvature_smooth'
            ])
            dr_orig = np.gradient(self._r_raw, s)
            dz_orig = np.gradient(self._z_raw, s)
            d2r_orig = np.gradient(dr_orig, s)
            d2z_orig = np.gradient(dz_orig, s)
            k_orig = np.abs(d2r_orig * dz_orig - dr_orig * d2z_orig) / \
                     (dr_orig**2 + dz_orig**2 + 1e-12)**1.5

            dr_smooth = self._cs_r_deriv(s)
            dz_smooth = self._cs_z_deriv(s)
            d2r_smooth = self._cs_r_deriv2(s)
            d2z_smooth = self._cs_z_deriv2(s)
            k_smooth = np.abs(d2r_smooth * dz_smooth - dr_smooth * d2z_smooth) / \
                       (dr_smooth**2 + dz_smooth**2 + 1e-12)**1.5

            for i in range(len(s)):
                writer.writerow([
                    f'{s[i]:.6f}',
                    f'{self._z_raw[i]:.6f}', f'{self._r_raw[i]:.6f}',
                    f'{self._z_smooth[i]:.6f}', f'{self._r_smooth[i]:.6f}',
                    f'{dr_orig[i]:.6f}', f'{dz_orig[i]:.6f}',
                    f'{dr_smooth[i]:.6f}', f'{dz_smooth[i]:.6f}',
                    f'{k_orig[i]:.8f}', f'{k_smooth[i]:.8f}'
                ])
        print(f"[DiffusedRevolutionSurface] Меридиан сохранён: {path}")

    # ------------------------------------------------------------------
    # Ядро: неявная прогонка (без изменений)
    # ------------------------------------------------------------------
    def _diffuse(self, s, h, tau, n_steps, bc_left, bc_right):
        N = len(s)
        s_uni = np.linspace(s[0], s[-1], N)
        h_uni = np.interp(s_uni, s, h)

        u_uni = np.linspace(self.base.u_min, self.base.u_max, N)
        pts_uni = np.array([self.base.position(u, 0.0) for u in u_uni])
        r_uni = np.sqrt(pts_uni[:, 0]**2 + pts_uni[:, 1]**2)

        ds = (s_uni[-1] - s_uni[0]) / (N - 1)

        r_prime = np.zeros(N)
        r_prime[1:-1] = (r_uni[2:] - r_uni[:-2]) / (2 * ds)
        r_prime[0] = (r_uni[1] - r_uni[0]) / ds
        r_prime[-1] = (r_uni[-1] - r_uni[-2]) / ds
        a = r_prime / np.maximum(r_uni, 1e-12)

        alpha = tau / ds**2 - tau * a / (2 * ds)
        beta = np.full(N, 1.0 + 2.0 * tau / ds**2)
        gamma = tau / ds**2 + tau * a / (2 * ds)

        ab = np.zeros((3, N))
        ab[0, 1:] = -gamma[:-1]
        ab[1, :] = beta
        ab[2, :-1] = -alpha[1:]

        ab = self._build_bc_matrix(ab, bc_left, bc_right, ds)

        h = h_uni.copy()
        for _ in range(n_steps):
            rhs = h.copy()
            rhs = self._build_bc_rhs(rhs, bc_left, bc_right, ds)
            h = solve_banded((1, 1), ab, rhs)

        return np.interp(s, s_uni, h)

    @staticmethod
    def _build_bc_matrix(ab, bc_left, bc_right, ds):
        if bc_left.kind == 'dirichlet':
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0
        elif bc_left.kind == 'neumann':
            ab[1, 0] = -1.0
            ab[0, 1] = 1.0
        elif bc_left.kind == 'robin':
            a, b = bc_left.coeff_a, bc_left.coeff_b
            ab[1, 0] = a + b / ds
            ab[0, 1] = -b / ds

        if bc_right.kind == 'dirichlet':
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0
        elif bc_right.kind == 'neumann':
            ab[2, -2] = -1.0
            ab[1, -1] = 1.0
        elif bc_right.kind == 'robin':
            a, b = bc_right.coeff_a, bc_right.coeff_b
            ab[2, -2] = -b / ds
            ab[1, -1] = a + b / ds

        return ab

    @staticmethod
    def _build_bc_rhs(rhs, bc_left, bc_right, ds):
        if bc_left.kind == 'dirichlet':
            rhs[0] = bc_left.value
        elif bc_left.kind == 'neumann':
            rhs[0] = bc_left.value * ds
        elif bc_left.kind == 'robin':
            rhs[0] = bc_left.value

        if bc_right.kind == 'dirichlet':
            rhs[-1] = bc_right.value
        elif bc_right.kind == 'neumann':
            rhs[-1] = bc_right.value * ds
        elif bc_right.kind == 'robin':
            rhs[-1] = bc_right.value

        return rhs

    # ------------------------------------------------------------------
    # Стандартный интерфейс по длине дуги s (без изменений)
    # ------------------------------------------------------------------
    def position(self, s, v):
        r = float(self._cs_r(s))
        z = float(self._cs_z(s))
        return np.array([r * np.cos(v), r * np.sin(v), z])

    def derivatives(self, s, v):
        r = float(self._cs_r(s))
        z = float(self._cs_z(s))
        dr = float(self._cs_r_deriv(s))
        dz = float(self._cs_z_deriv(s))

        ru = np.array([dr * np.cos(v), dr * np.sin(v), dz])
        rv = np.array([-r * np.sin(v), r * np.cos(v), 0.0])

        n = np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / norm
        return {"r": np.array([r * np.cos(v), r * np.sin(v), z]),
                "ru": ru, "rv": rv, "normal": n}

    def normal(self, s, v):
        return self.derivatives(s, v)["normal"]

    def first_fundamental_form(self, s, v):
        d = self.derivatives(s, v)
        E = float(np.dot(d["ru"], d["ru"]))
        F = float(np.dot(d["ru"], d["rv"]))
        G = float(np.dot(d["rv"], d["rv"]))
        return E, F, G

    def second_fundamental_form(self, s, v):
        d = self.derivatives(s, v)
        du = 1e-6
        r_uu = (self.derivatives(s + du, v)["ru"] - self.derivatives(s - du, v)["ru"]) / (2 * du)
        r_vv = (self.derivatives(s, v + 1e-6)["rv"] - self.derivatives(s, v - 1e-6)["rv"]) / (2 * 1e-6)
        r_uv = (self.derivatives(s + du, v + 1e-6)["ru"] - self.derivatives(s + du, v - 1e-6)["ru"]
                - self.derivatives(s - du, v + 1e-6)["ru"] + self.derivatives(s - du, v - 1e-6)["ru"]) / (4 * du * 1e-6)
        n = d["normal"]
        L = float(np.dot(r_uu, n))
        M = float(np.dot(r_uv, n))
        N = float(np.dot(r_vv, n))
        return L, M, N