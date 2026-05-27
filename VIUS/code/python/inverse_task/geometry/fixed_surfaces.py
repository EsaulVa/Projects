import numpy as np
from scipy.optimize import brentq
# geometry/fixed_surfaces.py
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
def get_surface_height_bounds(surface):
    """
    Универсально извлекает аксиальные (высотные) границы из ЛЮБОЙ поверхности.
    Порядок: u_min/u_max → axial_min/axial_max → z_min/z_max → v_min/v_max.
    """
    z_min = getattr(surface, 'u_min',
            getattr(surface, 'axial_min',
            getattr(surface, 'z_min',
            getattr(surface, 'v_min', -np.inf))))
    z_max = getattr(surface, 'u_max',
            getattr(surface, 'axial_max',
            getattr(surface, 'z_max',
            getattr(surface, 'v_max', np.inf))))
    return float(z_min), float(z_max)


def safe_initial_point(surface, point_3d, default_azimuth=None):
    """
    Безопасное определение начальных (u,v) для любой поверхности вращения.
    Исправляет Патч C: использует аксиальные границы вместо угловых.
    """
    x, y, z = np.asarray(point_3d, dtype=float)
    z_min, z_max = get_surface_height_bounds(surface)
    u0 = np.clip(z, z_min, z_max)
    v0 = default_azimuth if default_azimuth is not None else np.arctan2(y, x)
    return u0, v0

class FixedPiecewisePolynomialRevolution(PiecewisePolynomialRevolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axial_min = getattr(self, 'u_min', getattr(self, 'v_min', -np.inf))
        self.axial_max = getattr(self, 'u_max', getattr(self, 'v_max', np.inf))

    def normal(self, u, v):
        """Внешняя нормаль (согласованная с second_fundamental_form)."""
        n = super().normal(u, v)
        return -n

    def second_fundamental_form(self, u, v):
        """Для внешней нормали: L = r''/√(1+r'²), N = −r/√(1+r'²)."""
        r, rp, rpp = self._compute_r_and_derivs(u)
        denom = np.sqrt(1.0 + rp * rp)
        L = rpp / denom
        M = 0.0
        N = -r / denom
        return L, M, N
# class FixedPiecewisePolynomialRevolution(PiecewisePolynomialRevolution):
#     """
#     Наследник PiecewisePolynomialRevolution.
#     Исправления:
#       – убрана рекурсия в second_fundamental_form;
#       – L и N переставлены в правильные места;
#       – добавлены явные алиасы axial_min / axial_max для универсальности.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Публикуем границы под универсальными именами
#         self.axial_min = getattr(self, 'u_min', getattr(self, 'v_min', -np.inf))
#         self.axial_max = getattr(self, 'u_max', getattr(self, 'v_max', np.inf))

#     def second_fundamental_form(self, u, v):
#         """
#         Исправленная вторая фундаментальная форма.
#         Для r(u,v) = (r(u)cos v, r(u)sin v, u):
#             L = -r''(u) / sqrt(1+r'²)   (меридиан)
#             M = 0
#             N =  r(u)  / sqrt(1+r'²)   (параллель)
#         """
#         r, rp, rpp = self._compute_r_and_derivs(u)
#         denom = np.sqrt(1.0 + rp * rp)
#         L = -rpp / denom   # кривизна образующей (u-направление)
#         M = 0.0
#         N = r / denom      # кривизна параллели  (v-направление)
#         return L, M, N