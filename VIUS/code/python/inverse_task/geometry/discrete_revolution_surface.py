#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discrete_revolution_surface.py
==============================
Поверхность вращения, построенная из дискретных точек меридиана (z, r).

Реализует полный интерфейс, необходимый для DiffusedRevolutionSurface:
  position, derivatives, normal, first_fundamental_form, second_fundamental_form
"""

import numpy as np
from scipy.interpolate import CubicSpline


class DiscreteRevolutionSurface:
    """
    Поверхность вращения, заданная таблицей (z_i, r_i).
    Параметр u = z (осевая координата), v = угол поворота.
    """

    def __init__(self, z_array, r_array, bc_type="natural"):
        """
        z_array : 1D-array, осевая координата (монотонно возрастающая)
        r_array : 1D-array, радиус параллели
        bc_type : тип граничных условий сплайна ('natural', 'not-a-knot', 'clamped')
        """
        z = np.asarray(z_array, dtype=float)
        r = np.asarray(r_array, dtype=float)
        if len(z) != len(r):
            raise ValueError("z_array и r_array должны иметь одинаковую длину")
        if len(z) < 2:
            raise ValueError("Минимум 2 точки")
        # Убедимся в монотонности z (строго возрастающей)
        if np.any(np.diff(z) <= 0):
            # Удалим дубли
            z_unique, idx = np.unique(z, return_index=True)
            idx = np.sort(idx)
            z = z[idx]
            r = r[idx]
        self._z = z
        self._r = r
        self.u_min = float(z[0])
        self.u_max = float(z[-1])
        # Сплайн r(z) с аналитическими производными
        self._cs_r = CubicSpline(z, r, bc_type=bc_type)
        self._cs_dr = self._cs_r.derivative(1)
        self._cs_d2r = self._cs_r.derivative(2)

    def _eval(self, u):
        """Интерполяция r(u) и её производных в точке u=z."""
        u = float(np.clip(u, self.u_min, self.u_max))
        r = self._cs_r(u)
        dr = self._cs_dr(u)
        d2r = self._cs_d2r(u)
        return r, dr, d2r

    def position(self, u, v):
        r, _, _ = self._eval(u)
        return np.array([r * np.cos(v), r * np.sin(v), u])

    def derivatives(self, u, v):
        r, dr, _ = self._eval(u)
        ru = np.array([dr * np.cos(v), dr * np.sin(v), 1.0])
        rv = np.array([-r * np.sin(v), r * np.cos(v), 0.0])
        n = np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / norm
        return {"r": np.array([r * np.cos(v), r * np.sin(v), u]),
                "ru": ru, "rv": rv, "normal": n}

    def normal(self, u, v):
        return self.derivatives(u, v)["normal"]

    def first_fundamental_form(self, u, v):
        _, dr, _ = self._eval(u)
        E = 1.0 + dr**2
        F = 0.0
        G = self._cs_r(u)**2
        return float(E), float(F), float(G)

    def second_fundamental_form(self, u, v):
        r, dr, d2r = self._eval(u)
        # Аналитические формулы для поверхности вращения (u=z, v=angle)
        denom = np.sqrt(1.0 + dr**2)
        if denom < 1e-14:
            return 0.0, 0.0, 0.0
        L = -d2r / denom
        M = 0.0
        N = r / denom
        return float(L), float(M), float(N)

    def radius(self, u):
        """Возвращает радиус параллели для заданной осевой координаты u=z."""
        return float(self._cs_r(np.clip(u, self.u_min, self.u_max)))
