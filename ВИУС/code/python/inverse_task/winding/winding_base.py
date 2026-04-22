from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, List
import numpy as np
class WindingLineBuilderBase(ABC):
    @abstractmethod
    def build(self, initial_point, end_param=None, eval_points=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

class WindingResultProvider(ABC):
    @abstractmethod
    def get_uv_states(self) -> Optional[np.ndarray]: pass
    @abstractmethod
    def get_tangents(self) -> Optional[np.ndarray]: pass
    @abstractmethod
    def get_3d_points(self) -> Optional[np.ndarray]: pass
    @property
    @abstractmethod
    def last_run_successful(self) -> bool: pass

# class WindingLineBuilderBase(ABC):
#     """
#     Базовый абстрактный класс для построителей линии укладки на поверхности оправки.
#     Определяет общий интерфейс для решения прямой (система 2.10) и обратной (система 3.41) задач.
#     """

#     @abstractmethod
#     def build(
#         self,
#         initial_point: Tuple[float, float],
#         initial_tangent: Optional[Tuple[float, float]] = None,
#         end_param: Optional[float] = None,
#         eval_points: Optional[np.ndarray] = None,
#         **kwargs
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Запускает построение линии укладки.

#         Параметры
#         ---------
#         initial_point : (u0, v0)
#             Криволинейные координаты начальной точки на поверхности.
#         initial_tangent : (u_prime0, v_prime0) или (alpha0,), optional
#             Начальное направление:
#             - Для прямой задачи: производные du/ds, dv/ds или угол намотки α.
#             - Для обратной задачи: обычно не требуется (определяется траекторией).
#             Если None, то для обратной задачи направление определяется автоматически.
#         end_param : float, optional
#             Конечное значение параметра интегрирования:
#             - Для прямой задачи: длина s линии укладки.
#             - Для обратной задачи: длина z траектории точки схода.
#             Если None, используется полная длина (траектории для обратной задачи,
#             либо должно быть задано eval_points).
#         eval_points : np.ndarray, optional
#             Явный массив значений параметра, в которых требуется получить решение.
#             Если задан, игнорирует end_param и возвращает решение только в этих точках.
#         **kwargs
#             Дополнительные параметры, специфичные для конкретной реализации:
#             - Для прямой задачи: объект DeviationLaw, флаг нормировки.
#             - Для обратной задачи: может отсутствовать (уже заданы в конструкторе).

#         Возвращает
#         ----------
#         param_values : np.ndarray
#             Значения параметра интегрирования (s или z) в точках вывода.
#         points_3d : np.ndarray
#             Соответствующие 3D-координаты линии укладки на поверхности.
#         """
#         pass

#     @abstractmethod
#     def get_uv_states(self) -> Optional[np.ndarray]:
#         """
#         Возвращает массив криволинейных координат (u, v) после последнего расчёта.

#         Returns
#         -------
#         np.ndarray or None
#             Массив формы (N, 2) с колонками [u, v], если расчёт был выполнен,
#             иначе None.
#         """
#         pass

#     @abstractmethod
#     def get_tangents(self) -> Optional[np.ndarray]:
#         """
#         Возвращает массив производных (du/ds, dv/ds) или (du/dz, dv/dz)
#         после последнего расчёта.

#         Returns
#         -------
#         np.ndarray or None
#             Массив формы (N, 2), если расчёт был выполнен, иначе None.
#         """
#         pass

#     @abstractmethod
#     def get_3d_points(self) -> Optional[np.ndarray]:
#         """
#         Возвращает 3D-точки линии укладки после последнего расчёта.

#         Returns
#         -------
#         np.ndarray or None
#             Массив формы (N, 3), если расчёт был выполнен, иначе None.
#         """
#         pass

#     @property
#     @abstractmethod
#     def last_run_successful(self) -> bool:
#         """True, если последний вызов build завершился успешно."""
#         pass