# winding/winding_builder.py
import numpy as np
from typing import Optional, Tuple
from geometry.tsurfaces import AnalyticalSurface
from core.trajectory import Trajectory
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.base_solver import ODESolver
from winding.winding_base import WindingLineBuilderBase


class InvWindingLineBuilder:
    """
    Построитель линии укладки на поверхности оправки.

    Инкапсулирует процесс численного интегрирования системы (3.41)
    для восстановления линии укладки по известной траектории точки схода.
    Не занимается визуализацией или сохранением результатов.
    """

    def __init__(
        self,
        surface: AnalyticalSurface,
        trajectory: Trajectory,
        rhs_calculator: RightHandSideCalculator,
        solver: ODESolver
    ):
        """
        Параметры
        ---------
        surface : AnalyticalSurface
            Поверхность оправки (предоставляет геометрию).
        trajectory : Trajectory
            Траектория точки схода нити с натуральной параметризацией.
        rhs_calculator : RightHandSideCalculator
            Вычислитель правых частей системы (3.41).
        solver : ODESolver
            Решатель ОДУ (любая реализация абстрактного ODESolver).
        """
        self._surface = surface
        self._trajectory = trajectory
        self._rhs_calc = rhs_calculator
        self._solver = solver

        # Результаты последнего расчёта
        self._z_values: Optional[np.ndarray] = None
        self._uv_states: Optional[np.ndarray] = None   # (N, 2)
        self._points_3d: Optional[np.ndarray] = None   # (N, 3)

    def compute(
        self,
        u0: float,
        v0: float,
        z_end: Optional[float] = None,
        z_eval: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запускает интегрирование системы (3.41) от z=0 до z_end.

        Параметры
        ---------
        u0, v0 : float
            Начальные криволинейные координаты на поверхности,
            соответствующие началу траектории (z = 0).
        z_end : float, optional
            Конечное значение натурального параметра траектории.
            Если None, используется полная длина траектории.
        z_eval : np.ndarray, optional
            Явный массив значений z, в которых требуется получить решение.
            Если None, решатель сам определяет точки вывода (адаптивно).
        **solver_kwargs
            Дополнительные аргументы, передаваемые в solver.solve().
            Например, rtol, atol, max_step.

        Возвращает
        ----------
        z_values : np.ndarray
            Значения z, в которых вычислено решение.
        points_3d : np.ndarray
            Соответствующие 3D-точки линии укладки на поверхности (N, 3).
        """
        # Определяем конечную точку интегрирования
        if z_end is None:
            z_end = self._trajectory.total_length

        # Оборачиваем правые части в формат, ожидаемый решателем:
        # fun(z, state) -> производные
        def rhs_wrapper(z: float, state: np.ndarray) -> np.ndarray:
            du, dv = self._rhs_calc(z, state)
            return np.array([du, dv])

        # Начальный вектор состояния
        y0 = np.array([u0, v0], dtype=float)

        # Вызов решателя ОДУ
        z_vals, uv = self._solver.solve(
            fun=rhs_wrapper,
            t_span=(0.0, z_end),
            y0=y0,
            t_eval=z_eval,
            **solver_kwargs
        )

        # Сохраняем сырые результаты
        self._z_values = z_vals
        self._uv_states = uv

        # Преобразуем (u, v) в 3D-точки на поверхности
        self._points_3d = np.array([
            self._surface.position(u, v) for u, v in uv
        ])

        return self._z_values, self._points_3d

    def get_uv_states(self) -> Optional[np.ndarray]:
        """
        Возвращает массив криволинейных координат (u, v) после последнего расчёта.

        Returns
        -------
        np.ndarray or None
            Массив формы (N, 2) с колонками [u, v], если расчёт был выполнен,
            иначе None.
        """
        return self._uv_states

    # Дополнительные геттеры (опционально, для удобства клиента)
    def get_z_values(self) -> Optional[np.ndarray]:
        """Возвращает массив z, на которых было получено решение."""
        return self._z_values

    def get_3d_points(self) -> Optional[np.ndarray]:
        """Возвращает 3D-точки линии укладки."""
        return self._points_3d

 #Целевой класс как адаптер между классом InvWindingLineBuilder и интерфейсом WindingLineBuilderBase
class InverseWindingLineBuilder(WindingLineBuilderBase):
    def __init__(self, inverse_builder: InvWindingLineBuilder):
        self._builder = inverse_builder
        self._last_success = False

    def build(self, initial_point, initial_tangent=None, end_param=None,
              eval_points=None, **kwargs):
        u0, v0 = initial_point
        if initial_tangent is not None:
            # В обратной задаче начальное направление определяется траекторией,
            # можно либо игнорировать, либо выдавать предупреждение
            pass
        z_vals, points = self._builder.compute(
            u0, v0, z_end=end_param, z_eval=eval_points, **kwargs
        )
        self._last_success = True
        return z_vals, points

    def get_uv_states(self):
        return self._builder.get_uv_states()

    def get_tangents(self):
        # Для обратной задачи можно вычислить производные по s из внутренних данных,
        # либо вернуть None, если не требуется
        return None

    def get_3d_points(self):
        return self._builder.get_3d_points()

    @property
    def last_run_successful(self):
        return self._last_success