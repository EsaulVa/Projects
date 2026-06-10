#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_diffusion_filter.py
===========================
Пространственная фильтрация геометрии оправки (поверхности вращения)
на основе уравнения диффузии (теплопроводности) в метрике поверхности.

Поддерживает независимые граничные условия Дирихле, Неймана и Робина
на левом и правом концах меридиана.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundaryCondition:
    """
    Граничное условие для конца меридиана.

    Дирихле  : h = value
    Неймана  : dh/ds = value  (производная по параметру длины дуги s)
    Робин    : coeff_a * h + coeff_b * dh/dn = value
               где dh/dn — производная по внешней нормали.
               Для левого конца  dh/dn = -dh/ds.
               Для правого конца dh/dn = +dh/ds.
    """
    kind: str               # 'dirichlet', 'neumann', 'robin'
    value: float = 0.0      # фиксированное значение (Dirichlet) или правая часть
    coeff_a: float = 1.0    # для Robin: коэффициент при h
    coeff_b: float = 0.0    # для Robin: коэффициент при dh/dn

    @classmethod
    def dirichlet(cls, value: float):
        """h = value"""
        return cls('dirichlet', value=value)

    @classmethod
    def neumann(cls, flux: float = 0.0):
        """dh/ds = flux"""
        return cls('neumann', value=flux)

    @classmethod
    def robin(cls, coeff_a: float, coeff_b: float, value: float):
        """coeff_a * h + coeff_b * dh/dn = value"""
        return cls('robin', value=value, coeff_a=coeff_a, coeff_b=coeff_b)

    def with_value(self, new_value: float):
        """Возвращает копию с другим числовым значением (удобно для Дирихле
        при разных фиксациях r и z на одном и том же конце)."""
        return BoundaryCondition(self.kind, new_value, self.coeff_a, self.coeff_b)


class DiffusedRevolutionSurface:
    """
    Сглаженная поверхность вращения, построенная диффузией профиля меридиана.

    Параметр u — длина дуги меридиана s (от 0 до s_max).
    Параметр v — угол поворота 0..2π.
    """

    def __init__(self, base_surface, N=800, tau=10.0, n_steps=20,
                 bc_left: Optional[BoundaryCondition] = None,
                 bc_right: Optional[BoundaryCondition] = None):
        """
        base_surface : объект с методами position(u,v), normal(u,v), u_min, u_max
        N            : число точек дискретизации меридиана
        tau          : шаг по времени диффузии (мм²)
        n_steps      : число шагов (полное время t = tau * n_steps)
        bc_left      : ГУ на левом конце (s = s_min). По умолчанию Неймана (0).
        bc_right     : ГУ на правом конце (s = s_max). По умолчанию Неймана (0).
        """
        self.base = base_surface
        self.N = N
        self.tau = tau
        self.n_steps = n_steps
        self.bc_left = bc_left if bc_left is not None else BoundaryCondition.neumann(0.0)
        self.bc_right = bc_right if bc_right is not None else BoundaryCondition.neumann(0.0)

        # --- 1. Извлечь меридиан в исходном параметре u ---
        u_raw = np.linspace(base_surface.u_min, base_surface.u_max, N)
        pts = np.array([base_surface.position(u, 0.0) for u in u_raw])
        r_raw = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        z_raw = pts[:, 2]

        # --- 2. Перепараметризация по длине дуги s ---
        ds = np.sqrt(np.diff(r_raw)**2 + np.diff(z_raw)**2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        self.u_min = 0.0
        self.u_max = float(s[-1])

        # --- 3. Диффузия r(s) и z(s) с автовычислением Дирихле ---
        # Для Дирихле подставляем реальные концевые значения профиля,
        # чтобы концы меридиана не уезжали.
        bc_l_r = (self.bc_left.with_value(r_raw[0])
                  if self.bc_left.kind == 'dirichlet' else self.bc_left)
        bc_r_r = (self.bc_right.with_value(r_raw[-1])
                  if self.bc_right.kind == 'dirichlet' else self.bc_right)
        bc_l_z = (self.bc_left.with_value(z_raw[0])
                  if self.bc_left.kind == 'dirichlet' else self.bc_left)
        bc_r_z = (self.bc_right.with_value(z_raw[-1])
                  if self.bc_right.kind == 'dirichlet' else self.bc_right)

        r_smooth = self._diffuse(s, r_raw, tau, n_steps, bc_l_r, bc_r_r)
        z_smooth = self._diffuse(s, z_raw, tau, n_steps, bc_l_z, bc_r_z)

        # --- 4. Интерполяторы сглаженного профиля ---
        self._cs_r = CubicSpline(s, r_smooth)
        self._cs_z = CubicSpline(s, z_smooth)
        self._cs_r_deriv = self._cs_r.derivative(1)
        self._cs_z_deriv = self._cs_z.derivative(1)
        self._cs_r_deriv2 = self._cs_r.derivative(2)
        self._cs_z_deriv2 = self._cs_z.derivative(2)

    # ------------------------------------------------------------------
    # Ядро: неявная прогонка с произвольными ГУ
    # ------------------------------------------------------------------
    def _diffuse(self, s, h, tau, n_steps, bc_left, bc_right):
        """Неявная прогонка уравнения диффузии на меридиане."""
        N = len(s)
        # Равномерная сетка по s для численной схемы
        s_uni = np.linspace(s[0], s[-1], N)
        h_uni = np.interp(s_uni, s, h)

        # Радиус параллели на равномерной сетке (из исходной поверхности)
        u_uni = np.linspace(self.base.u_min, self.base.u_max, N)
        pts_uni = np.array([self.base.position(u, 0.0) for u in u_uni])
        r_uni = np.sqrt(pts_uni[:, 0]**2 + pts_uni[:, 1]**2)

        ds = (s_uni[-1] - s_uni[0]) / (N - 1)

        # Коэффициент a(s) = r'(s) / r(s)  (центральные разности)
        r_prime = np.zeros(N)
        r_prime[1:-1] = (r_uni[2:] - r_uni[:-2]) / (2 * ds)
        r_prime[0] = (r_uni[1] - r_uni[0]) / ds
        r_prime[-1] = (r_uni[-1] - r_uni[-2]) / ds
        a = r_prime / np.maximum(r_uni, 1e-12)

        # Коэффициенты неявной схемы для внутренних точек
        alpha = tau / ds**2 - tau * a / (2 * ds)
        beta = np.full(N, 1.0 + 2.0 * tau / ds**2)
        gamma = tau / ds**2 + tau * a / (2 * ds)

        # Формат solve_banded (l=1, u=1):
        # ab[0, 1:] = верхняя диагональ  M[i, i+1] = -gamma_i
        # ab[1, :]  = главная диагональ   M[i, i]   = beta_i
        # ab[2, :-1] = нижняя диагональ  M[i+1, i] = -alpha_{i+1}
        ab = np.zeros((3, N))
        ab[0, 1:] = -gamma[:-1]
        ab[1, :] = beta
        ab[2, :-1] = -alpha[1:]

        # Перестраиваем граничные строки матрицы под конкретные ГУ
        ab = self._build_bc_matrix(ab, bc_left, bc_right, ds)

        # Шаги по времени
        h = h_uni.copy()
        for _ in range(n_steps):
            rhs = h.copy()          # правая часть = предыдущее решение
            rhs = self._build_bc_rhs(rhs, bc_left, bc_right, ds)
            h = solve_banded((1, 1), ab, rhs)

        # Вернуть на исходную сетку s (для сплайна)
        return np.interp(s, s_uni, h)

    # ------------------------------------------------------------------
    # Сборка матрицы с ГУ
    # ------------------------------------------------------------------
    @staticmethod
    def _build_bc_matrix(ab, bc_left, bc_right, ds):
        """Модифицирует трехдиагональную матрицу согласно ГУ."""
        # --- Левая граница (i = 0) ---
        if bc_left.kind == 'dirichlet':
            # h_0 = value  =>  M[0,0] = 1, M[0,1] = 0
            ab[1, 0] = 1.0
            ab[0, 1] = 0.0

        elif bc_left.kind == 'neumann':
            # dh/ds = flux. Односторонняя вперед: (h_1 - h_0)/ds = flux
            # => -h_0 + h_1 = flux * ds
            ab[1, 0] = -1.0
            ab[0, 1] = 1.0

        elif bc_left.kind == 'robin':
            # a*h + b*dh/dn = value. Для левого конца dh/dn = -dh/ds.
            # Односторонняя: a*h_0 - b*(h_1 - h_0)/ds = value
            # => h_0*(a + b/ds) - h_1*(b/ds) = value
            a, b = bc_left.coeff_a, bc_left.coeff_b
            ab[1, 0] = a + b / ds
            ab[0, 1] = -b / ds

        # --- Правая граница (i = N-1) ---
        if bc_right.kind == 'dirichlet':
            # h_{N-1} = value  =>  M[N-1,N-1] = 1, M[N-1,N-2] = 0
            ab[1, -1] = 1.0
            ab[2, -2] = 0.0

        elif bc_right.kind == 'neumann':
            # dh/ds = flux. Односторонняя назад: (h_{N-1} - h_{N-2})/ds = flux
            # => -h_{N-2} + h_{N-1} = flux * ds
            ab[2, -2] = -1.0
            ab[1, -1] = 1.0

        elif bc_right.kind == 'robin':
            # a*h + b*dh/dn = value. Для правого конца dh/dn = +dh/ds.
            # Односторонняя: a*h_{N-1} + b*(h_{N-1} - h_{N-2})/ds = value
            # => -h_{N-2}*(b/ds) + h_{N-1}*(a + b/ds) = value
            a, b = bc_right.coeff_a, bc_right.coeff_b
            ab[2, -2] = -b / ds
            ab[1, -1] = a + b / ds

        return ab

    # ------------------------------------------------------------------
    # Сборка правой части с ГУ
    # ------------------------------------------------------------------
    @staticmethod
    def _build_bc_rhs(rhs, bc_left, bc_right, ds):
        """Модифицирует правую часть согласно ГУ."""
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
    # Стандартный интерфейс поверхности
    # ------------------------------------------------------------------
    def position(self, s, v):
        """Точка на сглаженной поверхности. s — длина дуги меридиана, v — угол."""
        r = float(self._cs_r(s))
        z = float(self._cs_z(s))
        return np.array([r * np.cos(v), r * np.sin(v), z])

    def derivatives(self, s, v):
        """Аналитические производные через сплайны профиля."""
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
        # Численное второе дифференцирование
        r_uu = (self.derivatives(s + du, v)["ru"] - self.derivatives(s - du, v)["ru"]) / (2 * du)
        r_vv = (self.derivatives(s, v + 1e-6)["rv"] - self.derivatives(s, v - 1e-6)["rv"]) / (2 * 1e-6)
        r_uv = (self.derivatives(s + du, v + 1e-6)["ru"] - self.derivatives(s + du, v - 1e-6)["ru"]
                - self.derivatives(s - du, v + 1e-6)["ru"] + self.derivatives(s - du, v - 1e-6)["ru"]) / (4 * du * 1e-6)
        n = d["normal"]
        L = float(np.dot(r_uu, n))
        M = float(np.dot(r_uv, n))
        N = float(np.dot(r_vv, n))
        return L, M, N