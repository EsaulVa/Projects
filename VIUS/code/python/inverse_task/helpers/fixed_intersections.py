
# helpers/fixed_intersections.py
import numpy as np
from scipy.optimize import brentq
from helpers.intersection import (
    PiecewisePolynomialIntersection,
    RobustRevolutionIntersection
)
from geometry.fixed_surfaces import get_surface_height_bounds
class FixedPiecewisePolynomialIntersection(PiecewisePolynomialIntersection):
    """
    Переопределён только метод проверки границ — теперь универсальный.
    Вся остальная логика (Brent + сканирование) наследуется без изменений.
    """

    def intersect(self, surface, origin, direction, t_min, t_max, sweep_steps=200):
        # Универсальные высотные границы вместо жёсткого surface.u_min / surface.v_min
        z_min, z_max = get_surface_height_bounds(surface)

        ro = np.asarray(origin, dtype=float)
        rd = np.asarray(direction, dtype=float)

        def get_R_surf(z):
            if z < z_min or z > z_max:
                return None
            pt = surface.position(z, 0.0)
            return np.hypot(pt[0], pt[1])

        def objective(t):
            pt = ro + t * rd
            R_surf = get_R_surf(pt[2])
            if R_surf is None:
                return 1e9
            R_ray = np.hypot(pt[0], pt[1])
            return R_ray - R_surf

        # --- дальше точная копия логики родителя ---
        dt = (t_max - t_min) / sweep_steps
        t_prev, f_prev = t_min, objective(t_min)

        for i in range(1, sweep_steps + 1):
            t_curr = t_min + i * dt
            f_curr = objective(t_curr)

            if abs(f_curr) < 1e-8:
                return t_curr, ro + t_curr * rd

            if f_prev * f_curr < 0:
                try:
                    t_hit = brentq(objective, t_prev, t_curr, xtol=1e-6)
                    return t_hit, ro + t_hit * rd
                except ValueError:
                    pass

            t_prev, f_prev = t_curr, f_curr

        return None, None


class FixedRobustRevolutionIntersection(RobustRevolutionIntersection):
    """
    Та же замена: проверка границ через get_surface_height_bounds.
    """

    def intersect(self, surface, origin, direction, t_min=1e-6, t_max=1e6, n_steps=1000):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        z_min, z_max = get_surface_height_bounds(surface)

        def signed_distance(t):
            pt = ro + t * rd
            if pt[2] < z_min or pt[2] > z_max:
                return -1e9
            r_ray = np.hypot(pt[0], pt[1])
            r_surf = surface.radius(pt[2])
            return r_ray - r_surf

        # --- дальше точная копия логики родителя ---
        dt = (t_max - t_min) / n_steps
        t_prev = t_min
        f_prev = signed_distance(t_prev)

        for i in range(1, n_steps + 1):
            t_curr = t_min + i * dt
            f_curr = signed_distance(t_curr)

            if abs(f_curr) < 1e-12:
                pt = ro + t_curr * rd
                if z_min <= pt[2] <= z_max:
                    return t_curr, pt

            if f_prev * f_curr < 0:
                lo, hi = t_prev, t_curr
                for _ in range(50):
                    mid = (lo + hi) / 2
                    f_mid = signed_distance(mid)
                    if abs(f_mid) < 1e-12 or hi - lo < 1e-12:
                        break
                    if f_prev * f_mid < 0:
                        hi = mid
                    else:
                        lo = mid
                        f_prev = f_mid
                t = (lo + hi) / 2
                pt = ro + t * rd
                return t, pt

            t_prev, f_prev = t_curr, f_curr

        return None, None