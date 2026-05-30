import numpy as np
from scipy.optimize import brentq
from geometry.piecewise_polynomial_revolution_fixed_v2 import PiecewisePolynomialRevolution

def get_surface_height_bounds(surface):
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

    # ===================================================================
    # ИСПРАВЛЕНИЕ: second_fundamental_form
    # -------------------------------------------------------------------
    # Родитель (PiecewisePolynomialRevolution) теперь возвращает II форму,
    # согласованную с ВНЕШНЕЙ нормалью:
    #   L =  r'' / √(1+r'^2)   (меридиан)
    #   N = -r   / √(1+r'^2)   (параллель)
    #
    # Ранее здесь было переопределение для ВНУТРЕННЕЙ нормали, что
    # противоречило normal() родителя и ломало корректор Ньютона.
    #
    # Теперь переопределение НЕ ТРЕБУЕТСЯ — родитель уже правильный.
    # Но если нужно явно зафиксировать поведение, оставляем вызов super().
    # ===================================================================
    def second_fundamental_form(self, u, v):
        # Явно используем родительскую реализацию, согласованную с внешней нормалью
        return super().second_fundamental_form(u, v)
