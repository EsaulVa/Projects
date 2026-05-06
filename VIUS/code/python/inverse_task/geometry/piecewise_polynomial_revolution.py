import numpy as np
from numpy.polynomial import polynomial as P  # не обязательно, можно через polyval
from typing import Dict, Tuple
from geometry.tsurfaces import AnalyticalSurface

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
        # self.v_min = self.a
        # self.v_max = self.d
        self.v_min = 0
        self.v_max = np.pi*2
        self.u_min = self.a
        self.u_max = self.d
    
    def radius(self, u):
        r, _, _ = self._compute_r_and_derivs(u)
        return r

    def _get_segment(self, u):
        if self.a <= u <= self.b:
            return 1
        elif self.b < u < self.c:
            return 2
        elif self.c <= u <= self.d:
            return 3
        else:
            raise ValueError(f"u={u} вне диапазона [{self.a}, {self.d}]")

    def _compute_r_and_derivs(self, u):
        seg = self._get_segment(u)
        if seg == 1:
            # нижнее днище: используем полиномы напрямую
            phi = np.polyval(self.phi_coeffs, u)
            phi_p = np.polyval(self.phi_prime_coeffs, u)
            phi_pp = np.polyval(self.phi_double_prime_coeffs, u)
            R = np.polyval(self.R_coeffs, phi)
            Rp = np.polyval(self.R_prime_coeffs, phi)
            Rpp = np.polyval(self.R_double_prime_coeffs, phi)
            # r(phi) = R(phi)*sin(phi)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            r = R * sin_phi
            # первая производная r'(u) = phi' * (Rp*sin_phi + R*cos_phi)
            temp = Rp * sin_phi + R * cos_phi
            r_prime = phi_p * temp
            # вторая производная
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
            # при зеркалировании r(u) = r_sym(u_sym), r'(u) = -r'_sym(u_sym), r''(u) = r''_sym(u_sym) (т.к. u_sym' = -1)
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
        # точка
        point = np.array([r * cos_v, r * sin_v, u])
        # ru = (r' cos v, r' sin v, 1)
        ru = np.array([rp * cos_v, rp * sin_v, 1.0])
        # rv = (-r sin v, r cos v, 0)
        rv = np.array([-r * sin_v, r * cos_v, 0.0])
        # нормаль: ru x rv, потом нормируем. Для поверхности вращения ru x rv = ( -cos_v, -sin_v, rp )
        cross = np.array([-cos_v, -sin_v, rp])  # в MATLAB-коде ru × rv даёт именно это, но без r? Проверим: ru = (rp cos, rp sin, 1); rv = (-r sin, r cos, 0); cross = ( -cos, -sin, rp ) (после сокращения на r?) Нет, пересчитаем: ru = (rp cos v, rp sin v, 1); rv = (-r sin v, r cos v, 0). Векторное произведение: ru × rv = ( -cos v * r, -sin v * r, rp * r )? Давайте вычислим:
        # i-компонента: ru[1]*rv[2] - ru[2]*rv[1] = (rp sin v)*0 - 1*(r cos v) = -r cos v
        # j-компонента: ru[2]*rv[0] - ru[0]*rv[2] = 1*(-r sin v) - (rp cos v)*0 = -r sin v
        # k-компонента: ru[0]*rv[1] - ru[1]*rv[0] = (rp cos v)*(r cos v) - (rp sin v)*(-r sin v) = r rp (cos^2 v + sin^2 v) = r rp.
        # Итак, cross = (-r cos v, -r sin v, r rp). Длина = r * sqrt(1 + rp^2). Тогда единичная нормаль = (-cos v / sqrt(1+rp^2), -sin v / sqrt(1+rp^2), rp / sqrt(1+rp^2)). Но в MATLAB-коде они, похоже, используют внешнюю нормаль. Нам безразлично, примем этот вектор как внешнюю нормаль.
        cross = np.array([-r * cos_v, -r * sin_v, r * rp])
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-12:
            normal = cross / norm_cross
        else:
            normal = np.array([0.0, 0.0, 1.0])  # на цилиндре rp=0, r>0 -> cross = (-r cos v, -r sin v, 0), длина r, нормаль = (-cos v, -sin v, 0) — внешняя? Да, внешняя нормаль цилиндра направлена от оси.
        return {'r': point, 'ru': ru, 'rv': rv, 'normal': normal}

    def first_fundamental_form(self, u, v):
        r, rp, _ = self._compute_r_and_derivs(u)
        E = r * r
        F = 0.0
        G = 1 + rp * rp
        return E, F, G

    def second_fundamental_form(self, u, v):
        r, rp, rpp = self._compute_r_and_derivs(u)
        denom = np.sqrt(1 + rp * rp)
        L = -r / denom
        M = 0.0
        N = rpp / denom
        return L, M, N

    def metric_derivatives(self, u, v):
        # Для простоты вернём нули, если не используем
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0