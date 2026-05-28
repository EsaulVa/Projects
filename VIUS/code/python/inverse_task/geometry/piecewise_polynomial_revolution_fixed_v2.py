import numpy as np
from numpy.polynomial import polynomial as P
from typing import Dict, Tuple
from geometry.tsurfaces import AnalyticalSurface
from core.exceptions import *

class PiecewisePolynomialRevolution(AnalyticalSurface):
    def __init__(self, phi_coeffs, R_coeffs, segment_bounds, cylinder_radius):
        self.phi_coeffs = phi_coeffs
        self.R_coeffs = R_coeffs
        self.a, self.b, self.c, self.d = segment_bounds
        self.cylinder_radius = cylinder_radius

        self.phi_prime_coeffs = np.polyder(phi_coeffs)
        self.phi_double_prime_coeffs = np.polyder(self.phi_prime_coeffs)
        self.R_prime_coeffs = np.polyder(R_coeffs)
        self.R_double_prime_coeffs = np.polyder(self.R_prime_coeffs)

        self.v_min = 0
        self.v_max = np.pi * 2
        self.u_min = self.a
        self.u_max = self.d

    def radius(self, u):
        r, _, _ = self._compute_r_and_derivs(u)
        return r

    def _get_segment(self, u):
        eps = 1e-2
        if u < self.a - eps or u > self.d + eps:
            # ИСПРАВЛЕНИЕ: нормальное сообщение вместо Ellipsis
            raise GeometryOutOfBoundsError('u', u, (self.a, self.d))
        if u < self.a:
            u = self.a
        if u > self.d:
            u = self.d
        if self.a <= u <= self.b:
            return 1
        elif self.b < u < self.c:
            return 2
        elif self.c <= u <= self.d:
            return 3
        else:
            raise GeometryOutOfBoundsError('u', u, (self.a, self.d))

    def _compute_r_and_derivs(self, u):
        seg = self._get_segment(u)
        if seg == 1:
            phi = np.polyval(self.phi_coeffs, u)
            phi_p = np.polyval(self.phi_prime_coeffs, u)
            phi_pp = np.polyval(self.phi_double_prime_coeffs, u)
            R = np.polyval(self.R_coeffs, phi)
            Rp = np.polyval(self.R_prime_coeffs, phi)
            Rpp = np.polyval(self.R_double_prime_coeffs, phi)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            r = R * sin_phi
            temp = Rp * sin_phi + R * cos_phi
            r_prime = phi_p * temp
            temp2 = Rpp * sin_phi + 2 * Rp * cos_phi - R * sin_phi
            r_double = phi_pp * temp + phi_p**2 * temp2
            return r, r_prime, r_double
        elif seg == 2:
            r = self.cylinder_radius
            return r, 0.0, 0.0
        else:
            u_sym = self.a + self.d - u
            r_sym, r_prime_sym, r_double_sym = self._compute_r_and_derivs(u_sym)
            r = r_sym
            r_prime = -r_prime_sym
            r_double = r_double_sym
            return r, r_prime, r_double

    def position(self, u, v):
        r, _, _ = self._compute_r_and_derivs(u)
        return np.array([r * np.cos(v), r * np.sin(v), u])

    def derivatives(self, u, v):
        r, rp, _ = self._compute_r_and_derivs(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        point = np.array([r * cos_v, r * sin_v, u])
        ru = np.array([rp * cos_v, rp * sin_v, 1.0])
        rv = np.array([-r * sin_v, r * cos_v, 0.0])
        # ВНЕШНЯЯ нормаль
        cross = np.array([r * cos_v, r * sin_v, -r * rp])
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-12:
            normal = cross / norm_cross
        else:
            normal = np.array([cos_v, sin_v, 0.0])
        return {'r': point, 'ru': ru, 'rv': rv, 'normal': normal}

    def partial_u(self, u, v):
        return self.derivatives(u, v)['ru']

    def partial_v(self, u, v):
        return self.derivatives(u, v)['rv']

    def normal(self, u, v):
        return self.derivatives(u, v)['normal']

    def first_fundamental_form(self, u, v):
        r, rp, _ = self._compute_r_and_derivs(u)
        E = 1.0 + rp * rp
        F = 0.0
        G = r * r
        return E, F, G

    def second_fundamental_form(self, u, v):
        r, rp, rpp = self._compute_r_and_derivs(u)
        denom = np.sqrt(1.0 + rp * rp)
        L = rpp / denom
        M = 0.0
        N = -r / denom
        return L, M, N

    def metric_derivatives(self, u, v):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def uv_from_point(self, point):
        x, y, z = point
        v = np.arctan2(y, x)
        u = z
        return u, v
