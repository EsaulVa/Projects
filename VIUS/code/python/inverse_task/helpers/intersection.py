# ======================================================================
# 1. ИНТЕРФЕЙС АЛГОРИТМА ПЕРЕСЕЧЕНИЯ ЛУЧА
# ======================================================================
import numpy as np
from abc import ABC, abstractmethod
from geometry.tsurfaces import *

class IntersectionAlgorithm(ABC):
    """Стратегия поиска пересечения луча с конкретным типом поверхности."""
    
    @abstractmethod
    def intersect(self, surface, origin, direction, t_min, t_max):
        """
        Параметры
        ----------
        surface : объект поверхности (CylinderAnalytical, SphereSegment, Ellipsoid и т.д.)
        origin : np.ndarray (3,)
        direction : np.ndarray (3,) – единичный вектор луча
        t_min, t_max : float – допустимый диапазон параметра t
        
        Возвращает
        ----------
        (t, point) или (None, None), если пересечение не найдено
        """
        pass


# ======================================================================
# 2. АНАЛИТИЧЕСКИЕ РЕАЛИЗАЦИИ ДЛЯ КОНКРЕТНЫХ ПОВЕРХНОСТЕЙ
# ======================================================================

class CylinderIntersection(IntersectionAlgorithm):
    def intersect(self, surface, origin, direction, t_min, t_max):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        R = surface.R  # у CylinderSegment радиус хранится в R
        a = rd[0]**2 + rd[1]**2
        b = 2 * (ro[0]*rd[0] + ro[1]*rd[1])
        c = ro[0]**2 + ro[1]**2 - R**2
        if abs(a) < 1e-12: return None, None
        D = b*b - 4*a*c
        if D < 0: return None, None
        sqrtD = np.sqrt(D)
        t1 = (-b - sqrtD) / (2*a)
        t2 = (-b + sqrtD) / (2*a)
        for t in sorted([t1, t2]):
            if t < t_min or t > t_max: continue
            pt = ro + t * rd
            # проверяем, попадает ли точка в диапазон высот сегмента
            if surface.z_min <= pt[2] <= surface.z_max:
                return t, pt
        return None, None

class SphereIntersection(IntersectionAlgorithm):
    def intersect(self, surface, origin, direction, t_min, t_max):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        R = surface.R
        z0 = surface.z0
        z_min, z_max = surface.z_min, surface.z_max
        ro_shifted = ro - np.array([0, 0, z0])
        a = rd[0]**2 + rd[1]**2 + rd[2]**2
        b = 2 * np.dot(ro_shifted, rd)
        c = np.dot(ro_shifted, ro_shifted) - R**2
        if abs(a) < 1e-12: return None, None
        D = b*b - 4*a*c
        if D < 0: return None, None
        sqrtD = np.sqrt(D)
        t1 = (-b - sqrtD) / (2*a)
        t2 = (-b + sqrtD) / (2*a)
        for t in sorted([t1, t2]):
            if t < t_min or t > t_max: continue
            pt = ro + t * rd
            if z_min <= pt[2] <= z_max:
                return t, pt
        return None, None

class EllipsoidIntersection(IntersectionAlgorithm):
    def intersect(self, surface, origin, direction, t_min, t_max):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        a, b, c = surface.a, surface.b, surface.c
        def f(t):
            pt = ro + t * rd
            return (pt[0]/a)**2 + (pt[1]/b)**2 + (pt[2]/c)**2 - 1.0
        try:
            from scipy.optimize import root_scalar
            sol = root_scalar(f, bracket=[t_min, t_max], method='brentq', xtol=1e-8)
            if sol.converged:
                t = sol.root
                if t_min <= t <= t_max:
                    return t, ro + t * rd
        except (ValueError, RuntimeError):
            pass
        return None, None

class RevolutionIntersection(IntersectionAlgorithm):
    def intersect(self, surface, origin, direction, t_min, t_max):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        def f(t):
            pt = ro + t * rd
            r_ray = np.hypot(pt[0], pt[1])
            r_surf = surface.radius(pt[2])   # образующая на высоте z
            return r_ray - r_surf
        try:
            from scipy.optimize import root_scalar
            sol = root_scalar(f, bracket=[t_min, t_max], method='brentq', xtol=1e-8)
            if sol.converged:
                t = sol.root
                if t_min <= t <= t_max:
                    return t, ro + t * rd
        except (ValueError, RuntimeError):
            pass
        return None, None

class RobustRevolutionIntersection(IntersectionAlgorithm):
    def intersect(self, surface, origin, direction, t_min=1e-6, t_max=1e6, n_steps=1000):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        
        def signed_distance(t):
            pt = ro + t * rd
            r_ray = np.hypot(pt[0], pt[1])
            # Проверяем, что высота pt[2] находится в пределах поверхности
            if hasattr(surface, 'v_min') and hasattr(surface, 'v_max'):
                if pt[2] < surface.v_min or pt[2] > surface.v_max:
                    return -1e9  # далеко от поверхности, отрицательное, чтобы не было ложного пересечения
            r_surf = surface.radius(pt[2])
            return r_ray - r_surf
        
        # Шаг 1: ищем интервал, где функция меняет знак
        dt = (t_max - t_min) / n_steps
        t_prev = t_min
        f_prev = signed_distance(t_prev)
        
        for i in range(1, n_steps + 1):
            t_curr = t_min + i * dt
            f_curr = signed_distance(t_curr)
            if abs(f_curr) < 1e-12:  # прямое попадание
                pt = ro + t_curr * rd
                if hasattr(surface, 'v_min') and (pt[2] < surface.v_min or pt[2] > surface.v_max):
                    continue
                return t_curr, pt
            if f_prev * f_curr < 0:  # смена знака
                # Уточняем корень бисекцией
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

# ======================================================================
# 3. РЕЕСТР АЛГОРИТМОВ И ТРАССИРОВЩИК
# ======================================================================
class RayTracer:
    def __init__(self):
        self.algorithms = {}

    def register(self, surface_class, algorithm):
        self.algorithms[surface_class] = algorithm

    def trace(self, surface, origin, direction, t_min=1e-6, t_max=1e6):
        # Составная поверхность – перебираем сегменты
        if hasattr(surface, 'segments'):
            best_t, best_point = None, None
            for seg in surface.segments:
                algo = self.algorithms.get(type(seg))
                if algo is None: continue
                t, pt = algo.intersect(seg, origin, direction, t_min, t_max)
                if t is not None and (best_t is None or t < best_t):
                    best_t, best_point = t, pt
            return best_t, best_point
        # Цельная поверхность
        algo = self.algorithms.get(type(surface))
        if algo:
            return algo.intersect(surface, origin, direction, t_min, t_max)
        return None, None


# ======================================================================
# 4. МЕТОД uv_from_point ДЛЯ ПОВЕРХНОСТЕЙ (примеры)
# ======================================================================
# Эти методы уже должны быть в классах поверхностей.
# Для полноты приведём их здесь.

# class CylinderAnalytical:
#     ...
#     def uv_from_point(self, point):
#         return np.arctan2(point[1], point[0]), point[2]

# class SphereSegment:
#     ...
#     def uv_from_point(self, point):
#         return np.arctan2(point[1], point[0]), point[2]

# class EllipsoidWithDerivatives:
#     ...
#     def uv_from_point(self, point):
#         x, y, z = point
#         theta = np.arccos(z / self.c)
#         phi = np.arctan2(y, x)
#         return theta, phi