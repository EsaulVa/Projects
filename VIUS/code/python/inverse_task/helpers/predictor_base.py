# predictors/predictor_base.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from geometry.tsurfaces import AnalyticalSurface
from core.trajectory import Trajectory

class Predictor(ABC):
    """
    Абстрактный базовый класс для стратегии предсказания следующей точки
    на поверхности оправки в обратной задаче намотки.
    """

    @abstractmethod
    def predict(
        self,
        z_k: float,
        z_next: float,
        u_cur: float,
        v_cur: float,
        surface: 'AnalyticalSurface',
        traj: 'Trajectory'
    ) -> Optional[Tuple[float, float]]:
        """
        Предсказать криволинейные координаты (u_pred, v_pred) на поверхности
        для параметра траектории z_next.

        Параметры
        ----------
        z_k : float
            Текущее значение натурального параметра траектории.
        z_next : float
            Следующее значение натурального параметра, для которого
            требуется предсказание.
        u_cur, v_cur : float
            Криволинейные координаты на поверхности в точке z_k.
        surface : AnalyticalSurface
            Поверхность оправки.
        traj : Trajectory
            Траектория точки схода нити.

        Возвращает
        ----------
        (u_pred, v_pred) : tuple of float
            Предсказанные координаты на поверхности.
        None : если предсказание невозможно (например, луч не пересекает
               поверхность или численный метод не сошёлся).
        """
        pass