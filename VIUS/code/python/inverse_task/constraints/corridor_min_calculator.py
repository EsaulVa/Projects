import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.trajectory import Trajectory
from geometry.tsurfaces import AnalyticalSurface
# Возможно, понадобится импорт вашей функции кривизны, если она вынесена
# from helpers.inverse_method import normal_curvature 

@dataclass
class CorridorMinResult:
    """Контейнер для результатов расчета нижней границы коридора."""
    s_array: np.ndarray               # Параметры длины дуги
    lu_points: np.ndarray             # 3D точки на линии укладки
    lambda_min: np.ndarray            # Минимально допустимые расстояния
    valid_mask: np.ndarray            # Булева маска (на случай ошибок геометрии)
    curvature_array: np.ndarray       # (Опционально) Вычисленная нормальная кривизна


class CorridorMinCalculator:
    """
    Вычисляет нижнюю границу коридора (lambda_min) на основе нормальной 
    кривизны поверхности, чтобы исключить отрыв нити на вогнутостях.
    """
    def __init__(self, 
                 lu_trajectory: Trajectory, 
                 mandrel_surface: AnalyticalSurface,
                 safe_margin: float = 5.0):
        """
        Параметры
        ----------
        lu_trajectory : Trajectory
            Объект линии укладки.
        mandrel_surface : AnalyticalSurface
            Поверхность ОПРАВКИ (E2), на которую укладывается нить (не стена безопасности!).
        safe_margin : float
            Минимальное расстояние от поверхности, даже если кривизна равна нулю 
            (выпуклый участок), чтобы нить не терлась об оправку.
        """
        self.traj = lu_trajectory
        self.surface = mandrel_surface
        self.safe_margin = safe_margin

    def calculate(self, num_points: int = 200) -> CorridorMinResult:
        s_array = np.linspace(0, self.traj.total_length, num_points)
        lu_points = np.zeros((num_points, 3))
        lambda_min = np.zeros(num_points)
        valid_mask = np.ones(num_points, dtype=bool)
        curvature_array = np.zeros(num_points)
        
        for i, s in enumerate(s_array):
            r = self.traj.R(s)
            tau = self.traj.R_deriv(s)
            lu_points[i] = r
            
            try:
                # 1. Восстанавливаем координаты на оправке
                u, v = self.surface.uv_from_point(r)
                
                # 2. Получаем нормаль к поверхности в этой точке
                normal = self.surface.normal(u, v)
                
                # 3. Считаем нормальную кривизну поверхности вдоль направления нити
                # ВАЖНО: Здесь нужно вызвать ваш алгоритм вычисления кривизны.
                # Если у вас есть функция normal_curvature(surface, u, v, up, vp), 
                # ее нужно адаптировать под наши данные.
                kappa_n = self._estimate_normal_curvature(u, v, tau, normal)
                curvature_array[i] = kappa_n
                
                # 4. Расчет lambda_min по физическому критерию
                if kappa_n > 1e-5:
                    # На вогнутости: нить стремится оторваться. 
                    # Приближенная формула: lambda_min ~ cos(beta) / kappa_n
                    # где cos(beta) - проекция tau на нормаль
                    cos_beta = abs(np.dot(tau, normal))
                    if cos_beta > 1.0: cos_beta = 1.0
                    
                    # Коэффициент запаса (зависит от натяжения, для прототипа берем 1.0)
                    lam_calc = cos_beta / kappa_n
                    
                    lambda_min[i] = max(lam_calc, self.safe_margin)
                else:
                    # На выпуклости или цилиндре: нить сама прижимается к поверхности
                    lambda_min[i] = self.safe_margin
                    
            except Exception as e:
                # Если uv_from_point упало (например, точка на стыке сегментов)
                valid_mask[i] = False
                lambda_min[i] = self.safe_margin
                curvature_array[i] = 0.0

        return CorridorMinResult(
            s_array=s_array,
            lu_points=lu_points,
            lambda_min=lambda_min,
            valid_mask=valid_mask,
            curvature_array=curvature_array
        )

    def _estimate_normal_curvature(self, u, v, tau, normal):
        """
        Вспомогательный метод для оценки нормальной кривизны.
        Использует классическую формулу дифференциальной геометрии: 
        k_n = II / I, где II и I - квадратичные формы.
        """
        try:
            # Получаем первую квадратичную форму
            E, F, G = self.surface.first_fundamental_form(u, v)
            
            # Получаем вторую квадратичную форму
            L, M, N_coeff = self.surface.second_fundamental_form(u, v)
            
            # Разложим касательный вектор tau на базисные векторы поверхности
            # tau = up * r_u + vp * r_v
            derivs = self.surface.derivatives(u, v)
            ru = derivs['ru']
            rv = derivs['rv']
            
            # Решаем систему: [ru, rv]^T * [up, vp]^T = tau
            # Матрица метрики: [[E, F], [F, G]]
            det = E * G - F * F
            if abs(det) < 1e-12:
                return 0.0
                
            tau_u = np.dot(tau, ru)
            tau_v = np.dot(tau, rv)
            
            up = ( G * tau_u - F * tau_v) / det
            vp = (-F * tau_u + E * tau_v) / det
            
            # Считаем I (первая квадратичная форма для вектора tau)
            I_form = E * up**2 + 2 * F * up * vp + G * vp**2
            
            # Считаем II (вторая квадратичная форма)
            II_form = L * up**2 + 2 * M * up * vp + N_coeff * vp**2
            
            if abs(I_form) < 1e-12:
                return 0.0
                
            # Нормальная кривизна
            kappa_n = II_form / I_form
            
            # На вогнутостях нормальная кривизна обычно положительна (по отношению к внешней нормали)
            # Если нормаль направлена наружу, а нить огибает впадину, кривизна > 0
            # ВАЖНО: Знак зависит от направления normal в вашем классе поверхности!
            # return kappa_n
        # ИСПРАВЛЕНИЕ ЗДЕСЬ: Возвращаем по модулю, так как знак зависит от стороны нормали
            return abs(kappa_n)  
            
        except Exception:
            return 0.0