import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.trajectory import Trajectory
from helpers.intersection import RayTracer

@dataclass
class CorridorMaxResult:
    """Контейнер для результатов расчета верхней границы коридора."""
    s_array: np.ndarray               # Параметры длины дуги (N точек)
    lu_points: np.ndarray             # 3D точки на линии укладки (N x 3)
    safety_points: np.ndarray         # 3D точки на поверхности E2 (N x 3)
    lambda_max: np.ndarray            # Расстояния (lambda_max) (N,)
    
    safety_trajectory: Optional[Trajectory] # Сглаженная траектория по точкам E2 (если удалось построить)
    valid_mask: np.ndarray            # Булева маска, где True - луч попал в E2


class CorridorMaxCalculator:
    """
    Вычисляет верхнюю границу коридора (lambda_max) путем 
    трассировки касательных лучей от ЛУ к поверхности безопасности E2.
    """
    def __init__(self, 
                 lu_trajectory: Trajectory, 
                 safety_surface, 
                 ray_tracer: RayTracer, 
                 safe_distance: float = 10.0):
        """
        Параметры
        ----------
        lu_trajectory : Trajectory
            Объект линии укладки (уже содержит точки на оправке E1 и касательные).
        safety_surface : AnalyticalSurface или CompositeSurface
            Поверхность безопасности E2.
        ray_tracer : RayTracer
            Настроенный трассировщик лучей с зарегистрированными алгоритмами.
        safe_distance : float
            Минимальное расстояние от оправки, с которого начинается поиск пересечения,
            чтобы избежать самопересечения луча с оправкой в точке старта.
        """
        self.traj = lu_trajectory
        self.surface = safety_surface
        self.tracer = ray_tracer
        self.t_min = safe_distance

    def calculate(self, num_points: int = 500, t_max: float = 3000.0) -> CorridorMaxResult:
        """
        Выполняет расчет коридора.
        
        Parameters
        ----------
        num_points : int
            Количество точек сэмплирования по длине ЛУ (включая начало и конец).
        t_max : float
            Максимально допустимая длина луча (габарит станка).
            
        Returns
        -------
        CorridorMaxResult
        """
        # 1. Генерируем равномерную сетку по длине дуги [0, total_length]
        s_array = np.linspace(0, self.traj.total_length, num_points)
        
        # 2. Инициализируем массивы под результаты
        lu_points = np.zeros((num_points, 3))
        safety_points = np.zeros((num_points, 3))
        lambda_max = np.zeros(num_points)
        valid_mask = np.zeros(num_points, dtype=bool)
        
        # 3. Основной цикл простреливания
        for i, s in enumerate(s_array):
            # Получаем данные из линии укладки
            r = self.traj.R(s)
            tau = self.traj.R_deriv(s) # Единичный вектор!
            
            lu_points[i] = r
            
            # Пускаем луч
            t, pt = self._find_intersection(r, tau, t_max)
            
            if t is not None:
                safety_points[i] = pt
                lambda_max[i] = t
                valid_mask[i] = True
            else:
                # Луч не попал в E2 (улетел в бесконечность)
                safety_points[i] = r + t_max * tau # Уводим точку далеко для графика
                lambda_max[i] = np.inf

        # 4. Пытаемся собрать сглаженную траекторию на E2
        safety_trajectory = self._build_safety_trajectory(safety_points, valid_mask)

        return CorridorMaxResult(
            s_array=s_array,
            lu_points=lu_points,
            safety_points=safety_points,
            lambda_max=lambda_max,
            safety_trajectory=safety_trajectory,
            valid_mask=valid_mask
        )

    def _find_intersection(self, origin: np.ndarray, direction: np.ndarray, t_max: float):
        """Вспомогательный метод для трассировки одного луча."""
        try:
            t, pt = self.tracer.trace(
                surface=self.surface, 
                origin=origin, 
                direction=direction, 
                t_min=self.t_min, 
                t_max=t_max
            )
            return t, pt
        except Exception:
            # Защита от любых математических сбоев внутри трассировщика
            return None, None

    def _build_safety_trajectory(self, points: np.ndarray, mask: np.ndarray) -> Optional[Trajectory]:
        """
        Создает объект Trajectory из валидных точек пересечения.
        Если точек слишком мало или они идут не подряд, сплайн построить не удастся.
        """
        valid_points = points[mask]
        
        if len(valid_points) < 4:
            return None
            
        try:
            # from_points сам вызовет хордовую параметризацию и фабрику сплайнов
            return Trajectory.from_points(valid_points, method='cubic')
        except Exception:
            # Если точки выстроились в безобразную линию (из-за сложной формы E2),
            # сплайн может выдать ошибку. Просто возвращаем None.
            return None