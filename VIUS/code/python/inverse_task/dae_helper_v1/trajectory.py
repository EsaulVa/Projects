#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trajectory.py
=============
Траектория раскладчика, параметризованная по длине дуги.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz


class TrajectoryByArcLength:
    """
    Траектория R(s), где s — длина дуги.
    Строится из дискретных точек через хордовую параметризацию
    и интегрирование длины дуги.
    """

    def __init__(self, points):
        points = np.asarray(points)
        self._build(points)

    def _build(self, points):
        n = len(points)
        chord_lengths = np.zeros(n)
        for i in range(1, n):
            chord_lengths[i] = chord_lengths[i-1] + np.linalg.norm(points[i] - points[i-1])
        total_chord = chord_lengths[-1]
        if total_chord < 1e-12:
            raise ValueError("Слишком короткая траектория")
        u_param = chord_lengths / total_chord
        self._sx = CubicSpline(u_param, points[:, 0], bc_type="natural")
        self._sy = CubicSpline(u_param, points[:, 1], bc_type="natural")
        self._sz = CubicSpline(u_param, points[:, 2], bc_type="natural")
        self._u_param = u_param
        u_fine = np.linspace(0, 1, 5000)
        dx = self._sx(u_fine, 1)
        dy = self._sy(u_fine, 1)
        dz = self._sz(u_fine, 1)
        speed = np.sqrt(dx**2 + dy**2 + dz**2)
        s_fine = cumtrapz(speed, u_fine, initial=0)
        self.total_length = float(s_fine[-1])
        self._u_of_s = interp1d(s_fine, u_fine, kind="cubic",
                                 bounds_error=False, fill_value=(0.0, 1.0))

    def R(self, s):
        s = np.clip(float(s), 0.0, self.total_length)
        u = float(self._u_of_s(s))
        return np.array([self._sx(u), self._sy(u), self._sz(u)])

    def R_deriv(self, s):
        """Единичный вектор касательной dr/ds."""
        s = np.clip(float(s), 0.0, self.total_length)
        u = float(self._u_of_s(s))
        dx = self._sx(u, 1)
        dy = self._sy(u, 1)
        dz = self._sz(u, 1)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return np.array([dx, dy, dz]) / norm

    def R_array(self, s_array):
        """Векторизованная версия R(s)."""
        return np.array([self.R(s) for s in s_array])
