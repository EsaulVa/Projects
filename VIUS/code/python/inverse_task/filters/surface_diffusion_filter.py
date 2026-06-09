#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_diffusion_filter.py
===========================
Пространственная фильтрация геометрии оправки (поверхности вращения)
на основе уравнения диффузии (теплопроводности) в метрике поверхности.

Для поверхности вращения с меридианом (r(s), z(s)), параметризованным
длиной дуги s, оператор Лапласа–Бельтрами на осесимметричной функции:

    Δ_G h = h''(s) + (r'(s)/r(s)) * h'(s)

Решается неявной схемой (прогонка) для r(s) и z(s) отдельно.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded


class DiffusedRevolutionSurface:
    """
    Сглаженная поверхность вращения, построенная диффузией профиля меридиана.

    Параметр u — длина дуги меридиана s (от 0 до s_max).
    Параметр v — угол поворота 0..2π.
    """

    def __init__(self, base_surface, N=800, tau=10.0, n_steps=20):
        """
        base_surface : объект с методами position(u,v), normal(u,v), u_min, u_max
        N            : число точек дискретизации меридиана
        tau          : шаг по времени диффузии (мм²)
        n_steps      : число шагов (полное время t = tau * n_steps)
        """
        self.base = base_surface
        self.N = N
        self.tau = tau
        self.n_steps = n_steps

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

        # --- 3. Диффузия r(s) и z(s) ---
        r_smooth = self._diffuse(s, r_raw, tau, n_steps)
        z_smooth = self._diffuse(s, z_raw, tau, n_steps)

        # --- 4. Интерполяторы сглаженного профиля ---
        self._cs_r = CubicSpline(s, r_smooth)
        self._cs_z = CubicSpline(s, z_smooth)
        self._cs_r_deriv = self._cs_r.derivative(1)
        self._cs_z_deriv = self._cs_z.derivative(1)
        self._cs_r_deriv2 = self._cs_r.derivative(2)
        self._cs_z_deriv2 = self._cs_z.derivative(2)

    def _diffuse(self, s, h, tau, n_steps):
        """Неявная прогонка уравнения диффузии на меридиане."""
        N = len(s)
        # Равномерная сетка по s для численной схемы
        s_uni = np.linspace(s[0], s[-1], N)
        h_uni = np.interp(s_uni, s, h)
        r_uni = np.interp(s_uni, s, np.sqrt(
            np.array([self.base.position(u, 0.0) for u in
                      np.linspace(self.base.u_min, self.base.u_max, N)])[:, 0]**2 +
            np.array([self.base.position(u, 0.0) for u in
                      np.linspace(self.base.u_min, self.base.u_max, N)])[:, 1]**2
        ))
        ds = (s_uni[-1] - s_uni[0]) / (N - 1)

        # Коэффициент a(s) = r'(s) / r(s)  (центральные разности)
        r_prime = np.zeros(N)
        r_prime[1:-1] = (r_uni[2:] - r_uni[:-2]) / (2 * ds)
        r_prime[0] = (r_uni[1] - r_uni[0]) / ds
        r_prime[-1] = (r_uni[-1] - r_uni[-2]) / ds
        a = r_prime / np.maximum(r_uni, 1e-12)

        # Матрица неявной схемы:  -α h_{i-1} + β h_i - γ h_{i+1} = h_i^n
        alpha = tau / ds**2 - tau * a / (2 * ds)
        beta = np.full(N, 1.0 + 2.0 * tau / ds**2)
        gamma = tau / ds**2 + tau * a / (2 * ds)

        # Формат solve_banded (l=1, u=1): ab[0,1:] = super, ab[1,:] = diag, ab[2,:-1] = sub
        ab = np.zeros((3, N))
        ab[0, 1:] = -gamma[:-1]   # M[i, i+1] = -gamma_i
        ab[1, :] = beta            # M[i, i]   = beta_i
        ab[2, :-1] = -alpha[1:]   # M[i+1, i] = -alpha_{i+1}

        # Граничные условия Неймана (нулевая производная на концах)
        # i=0:  (beta_0 - alpha_0) h_0 - gamma_0 h_1 = h_0^n
        ab[1, 0] = beta[0] - alpha[0]
        # i=N-1: -alpha_{N-1} h_{N-2} + (beta_{N-1} - gamma_{N-1}) h_{N-1} = h_{N-1}^n
        ab[1, -1] = beta[-1] - gamma[-1]

        # Шаги по времени
        h = h_uni.copy()
        for _ in range(n_steps):
            h = solve_banded((1, 1), ab, h)

        # Вернуть на исходную сетку s (для сплайна)
        return np.interp(s, s_uni, h)

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