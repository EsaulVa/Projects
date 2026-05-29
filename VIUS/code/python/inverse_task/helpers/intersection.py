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
# 2. ЧИСТАЯ АЛЬТЕРНАТИВА: Численный алгоритм-стратегия
# ======================================================================
from scipy.optimize import brentq
class PiecewisePolynomialIntersection(IntersectionAlgorithm):
    """
    Чистая стратегия пересечения для поверхностей вида PiecewisePolynomialRevolution.
    Не вторгается в класс поверхности. Использует численный поиск корня уравнения
    R_ray(z) - R_surf(z) = 0 со сканированием интервала для robustness.
    """
    def intersect(self, surface, origin, direction, t_min, t_max, sweep_steps=200):
        ro = np.asarray(origin, dtype=float)
        rd = np.asarray(direction, dtype=float)
        
        def get_R_surf(z):
            # Уважаем границы поверхности
            if hasattr(surface, 'u_min') and (z < surface.u_min or z > surface.u_max):
                return None
            # Берем радиус на высоте z при угле v=0
            pt = surface.position(z, 0.0)
            return np.hypot(pt[0], pt[1])
                
        def objective(t):
            pt = ro + t * rd
            R_surf = get_R_surf(pt[2])
            if R_surf is None: 
                return 1e9 # Штраф за выход за пределы высоты
            R_ray = np.hypot(pt[0], pt[1])
            return R_ray - R_surf

        # Сканирование: ищем интервал, где функция меняет знак
        dt = (t_max - t_min) / sweep_steps
        t_prev, f_prev = t_min, objective(t_min)
        
        for i in range(1, sweep_steps + 1):
            t_curr = t_min + i * dt
            f_curr = objective(t_curr)
            
            if abs(f_curr) < 1e-8: # Прямое попадание
                return t_curr, ro + t_curr * rd
                
            if f_prev * f_curr < 0: # Нашли смену знака - там есть корень!
                try:
                    # Используем брентq только на найденном узком отрезке
                    t_hit = brentq(objective, t_prev, t_curr, xtol=1e-6)
                    return t_hit, ro + t_hit * rd
                except ValueError:
                    pass # Если брентq не сошелся (редко), идем дальше
                    
            t_prev, f_prev = t_curr, f_curr
            
        return None, None # Пересечение не найдено


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
        
        z_min = getattr(surface, 'v_min', -np.inf)
        z_max = getattr(surface, 'v_max', np.inf)
        
        # Расширяем зону поиска на 10% или минимум на 50мм, 
        # чтобы ловить пересечения на границах сегментов
        margin = max((z_max - z_min) * 0.1, 50.0)
        z_search_min = z_min - margin
        z_search_max = z_max + margin

        def signed_distance(t):
            pt = ro + t * rd
            z_pt = pt[2]
            
            # Мягкий штраф вместо жесткого обрыва
            # Если точка далеко за пределами — возвращаем большое положительное число
            # (считаем, что мы "снаружи" бесконечного цилиндра)
            if z_pt < z_search_min or z_pt > z_search_max:
                return 1e9 
            
            r_ray = np.hypot(pt[0], pt[1])
            
            # Безопасный вызов radius: если z выходит за пределы определения полинома,
            # экстраполируем или используем ближайшее значение
            try:
                r_surf = surface.radius(z_pt)
            except (ValueError, IndexError):
                # Fallback: если radius не определен, считаем расстояние до оси огромным
                return 1e9
                
            return r_ray - r_surf
        
        # Сканирование интервала
        dt = (t_max - t_min) / n_steps
        t_prev = t_min
        f_prev = signed_distance(t_prev)
        
        for i in range(1, n_steps + 1):
            t_curr = t_min + i * dt
            f_curr = signed_distance(t_curr)
            
            # Прямое попадание
            if abs(f_curr) < 1e-12:
                pt = ro + t_curr * rd
                # Финальная проверка: точка должна быть СТРОГО в пределах поверхности
                if z_min <= pt[2] <= z_max:
                    return t_curr, pt
                    
            # Смена знака -> уточнение корня
            if f_prev * f_curr < 0:
                lo, hi = t_prev, t_curr
                # Бисекция/Брент для уточнения
                try:
                    from scipy.optimize import brentq
                    t_root = brentq(signed_distance, lo, hi, xtol=1e-9)
                except ValueError:
                    # Ручная бисекция как fallback
                    for _ in range(50):
                        mid = (lo + hi) / 2.0
                        f_mid = signed_distance(mid)
                        if abs(f_mid) < 1e-12 or hi - lo < 1e-12:
                            break
                        if f_prev * f_mid < 0:
                            hi = mid
                        else:
                            lo = mid
                            f_prev = f_mid
                    t_root = (lo + hi) / 2.0
                    
                pt = ro + t_root * rd
                
                # ВАЖНО: Возвращаем только если точка попала в реальные границы
                if z_min <= pt[2] <= z_max:
                    return t_root, pt
            
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

    # def trace(self, surface, origin, direction, t_min=1e-6, t_max=1e6):
    def trace(self, surface, origin, direction, t_min=1e-6, t_max=1e6, current_uv=None):
        # Составная поверхность – перебираем сегменты
        # Составная поверхность
        if hasattr(surface, 'segments'):
            best_t, best_point, best_seg_idx = None, None, None
            
            for idx, seg in enumerate(surface.segments):
                algo = self.algorithms.get(type(seg))
                if algo is None: continue
                
                t, pt = algo.intersect(seg, origin, direction, t_min, t_max)
                
                if t is not None:
                    # Если передана текущая точка, проверяем "разумность" пересечения
                    if current_uv is not None:
                        try:
                            # Пытаемся получить UV найденной точки на этом сегменте
                            # (Предполагаем, что у сегмента есть uv_from_point или мы знаем его геометрию)
                            # Для поверхностей вращения u обычно равно Z точки.
                            u_candidate = pt[2] 
                            u_cur = current_uv[0]
                            
                            # Если кандидат слишком далеко по высоте от текущего u — штраф
                            if abs(u_candidate - u_cur) > 50.0: # Эвристический порог
                                continue # Игнорируем далекие сегменты
                        except:
                            pass

                    # Стандартный выбор ближайшего по t, если фильтры прошли
                    if best_t is None or t < best_t:
                        best_t, best_point, best_seg_idx = t, pt, idx
                        
            return best_t, best_point
        # ...
        # if hasattr(surface, 'segments'):
        #     best_t, best_point = None, None
        #     for seg in surface.segments:
        #         algo = self.algorithms.get(type(seg))
        #         if algo is None: continue
        #         t, pt = algo.intersect(seg, origin, direction, t_min, t_max)
        #         if t is not None and (best_t is None or t < best_t):
        #             best_t, best_point = t, pt
        #     return best_t, best_point
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