import numpy as np
from numpy.polynomial import polynomial as P
from typing import Dict, Tuple
from geometry.tsurfaces import AnalyticalSurface
from core.exceptions import *

class PiecewisePolynomialRevolution(AnalyticalSurface):
    def __init__(self, phi_coeffs, R_coeffs, segment_bounds, cylinder_radius):
        """
        phi_coeffs: коэффициенты полинома phi(u) от старшего к младшему (степень 4)
        R_coeffs: коэффициенты полинома R(phi) от старшего к младшему (степень 5)
        segment_bounds: [a, b, c, d] длины в мм
        cylinder_radius: радиус цилиндрического сегмента (между b и c)
        """
        self.phi_coeffs = phi_coeffs
        self.R_coeffs = R_coeffs
        self.a, self.b, self.c, self.d = segment_bounds
        self.cylinder_radius = cylinder_radius

        # Производные полиномов
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
            # нижнее днище
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
            # цилиндр
            r = self.cylinder_radius
            return r, 0.0, 0.0
        else:  # seg == 3
            # верхнее днище: зеркальное отражение нижнего
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
        # ru = dr/du
        ru = np.array([rp * cos_v, rp * sin_v, 1.0])
        # rv = dr/dv
        rv = np.array([-r * sin_v, r * cos_v, 0.0])
        # ВНЕШНЯЯ нормаль: ru x rv
        # ru = (rp cos v, rp sin v, 1)
        # rv = (-r sin v, r cos v, 0)
        # cross = ( -r cos v, -r sin v, r*rp )  -- это ВНУТРЕННЯЯ
        # Для ВНЕШНЕЙ берём обратное направление: rv x ru = (r cos v, r sin v, -r*rp)
        cross = np.array([r * cos_v, r * sin_v, -r * rp])
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-12:
            normal = cross / norm_cross
        else:
            # На цилиндре rp=0, cross=(r cos v, r sin v, 0), нормаль = (cos v, sin v, 0)
            normal = np.array([cos_v, sin_v, 0.0])
        return {'r': point, 'ru': ru, 'rv': rv, 'normal': normal}

    def partial_u(self, u, v):
        return self.derivatives(u, v)['ru']

    def partial_v(self, u, v):
        return self.derivatives(u, v)['rv']

    def normal(self, u, v):
        return self.derivatives(u, v)['normal']

    # ===================================================================
    # ИСПРАВЛЕНИЕ 1: first_fundamental_form
    # -------------------------------------------------------------------
    # Стандарт дифференциальной геометрии:
    #   E = |ru|^2,  F = ru·rv,  G = |rv|^2
    # Для поверхности вращения r(u,v) = (r(u)cos v, r(u)sin v, u):
    #   ru = (r' cos v, r' sin v, 1)  => |ru|^2 = r'^2 + 1
    #   rv = (-r sin v, r cos v, 0)   => |rv|^2 = r^2
    # БЫЛО: E = r^2, G = 1+r'^2  (перепутаны местами)
    # ===================================================================
    def first_fundamental_form(self, u, v):
        r, rp, _ = self._compute_r_and_derivs(u)
        E = 1.0 + rp * rp   # |ru|^2
        F = 0.0             # ru · rv = 0 для поверхности вращения
        G = r * r           # |rv|^2
        return E, F, G

    # ===================================================================
    # ИСПРАВЛЕНИЕ 2: second_fundamental_form
    # -------------------------------------------------------------------
    # Вторая фундаментальная форма зависит от выбора нормали.
    # Формула:  L = r_uu · n,  M = r_uv · n,  N = r_vv · n
    # Для поверхности вращения с ВНЕШНЕЙ нормалью n = (cos v, sin v, -r'/√(1+r'^2)) / 
    #   ... подождём, давайте выведем аккуратно.
    #
    # Параметризация: r(u,v) = (r(u)cos v, r(u)sin v, u)
    # ru = (r' cos v, r' sin v, 1)
    # rv = (-r sin v, r cos v, 0)
    # ruu = (r'' cos v, r'' sin v, 0)
    # ruv = (-r' sin v, r' cos v, 0)
    # rvv = (-r cos v, -r sin v, 0)
    #
    # Внешняя нормаль (единичная): n = (cos v, sin v, -r'/√(1+r'^2)) * r/|cross|
    # Но |cross| = r√(1+r'^2), поэтому:
    #   n = (cos v/√(1+r'^2), sin v/√(1+r'^2), -r'/√(1+r'^2))
    #
    # Тогда:
    #   L = ruu · n = r'' cos v * cos v/√(...) + r'' sin v * sin v/√(...) + 0
    #     = r'' / √(1+r'^2)
    #   M = ruv · n = (-r' sin v)*cos v/√(...) + (r' cos v)*sin v/√(...) = 0
    #   N = rvv · n = (-r cos v)*cos v/√(...) + (-r sin v)*sin v/√(...) 
    #     = -r / √(1+r'^2)
    #
    # БЫЛО (для внутренней нормали): L = -r''/√(...), N = r/√(...)
    # СТАЛО (для внешней нормали):  L =  r''/√(...), N = -r/√(...)
    #
    # ВАЖНО: compute_grad_Phi в inverse_method.py использует II форму для
    # вычисления градиента связи. Если знак II формы не совпадает со знаком
    # нормали, корректор Ньютона движет точку В ПРОТИВОПОЛОЖНУЮ сторону,
    # увеличивая зазор вместо его уменьшения.
    # ===================================================================
    def second_fundamental_form(self, u, v):
        r, rp, rpp = self._compute_r_and_derivs(u)
        denom = np.sqrt(1.0 + rp * rp)
        L = rpp / denom   # согласовано с ВНЕШНЕЙ нормалью
        M = 0.0
        N = -r / denom    # согласовано с ВНЕШНЕЙ нормалью
        return L, M, N

    def metric_derivatives(self, u, v):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def uv_from_point(self, point):
        x, y, z = point
        v = np.arctan2(y, x)
        u = z
        return u, v
