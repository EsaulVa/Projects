from winding.winding_base import WindingLineBuilderBase
from geometry.tsurfaces import AnalyticalSurface
from solvers.base_solver import ODESolver
from core.base_deviation_law import DeviationLaw
import numpy as np
from typing import Callable, Optional, Tuple
from solvers.linear_solver import *
from .forward_rhs_calculator import ForwardRHS,QRegularizedForwardRHS,DelegatingForwardRHS

# forward_winding/forward_winding_builder.py
# import numpy as np
# from typing import Optional, Tuple, Callable
# from geometry.tsurfaces import AnalyticalSurface
# from winding.deviation_law import DeviationLaw
# from solvers.base_solver import ODESolver
# from solvers.local_linear_solver import LocalLinearSolver, QRegularizedSolver
# from forward_winding.delegating_forward_rhs import DelegatingForwardRHS

class ForwardWindingBuilder:
    """Построитель линии укладки (прямая задача). Поддерживает внешний RHS и диагностику."""

    def __init__(
        self,
        surface: AnalyticalSurface,
        deviation_law: DeviationLaw,
        solver: ODESolver,
        rhs: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        linear_solver: Optional[LocalLinearSolver] = None,
        normalize_tangent: bool = True,
        eps: float = 1e-12
    ):
        self._surface = surface
        self._law = deviation_law
        self._solver = solver
        self._normalize = normalize_tangent
        self.eps = eps

        # RHS
        if rhs is not None:
            self._rhs = rhs
        else:
            if linear_solver is None:
                linear_solver = QRegularizedSolver()
            self._rhs = DelegatingForwardRHS(surface, deviation_law, linear_solver, eps=eps)

        # Результаты
        self._s_values: Optional[np.ndarray] = None
        self._uv_states: Optional[np.ndarray] = None      # (N,2)
        self._tangents: Optional[np.ndarray] = None       # (N,2) du/ds, dv/ds
        self._points_3d: Optional[np.ndarray] = None      # (N,3)
        self._success: bool = False
        self._diagnostics: dict = {}

    # ---------- основной метод ----------
    def build(
        self,
        initial_point: Tuple[float, float],
        initial_tangent: Optional[Tuple[float, float]] = None,
        end_param: Optional[float] = None,
        eval_points: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        u0, v0 = initial_point

        # начальное направление
        if initial_tangent is None:
            raise ValueError("Для прямой задачи необходимо задать initial_tangent")
        if len(initial_tangent) == 1:
            alpha = initial_tangent[0]
            u_prime0, v_prime0 = self._angle_to_derivatives(u0, v0, alpha)
        else:
            u_prime0, v_prime0 = initial_tangent

        # нормировка
        if self._normalize:
            E, F, G = self._surface.first_fundamental_form(u0, v0)
            norm2 = E * u_prime0**2 + 2*F * u_prime0*v_prime0 + G * v_prime0**2
            if norm2 < self.eps:
                raise ValueError(f"Вырожденная метрика в начальной точке (u={u0}, v={v0})")
            scale = 1.0 / np.sqrt(norm2)
            u_prime0 *= scale
            v_prime0 *= scale

        y0 = np.array([u0, u_prime0, v0, v_prime0])

        if eval_points is not None:
            s_span = (eval_points[0], eval_points[-1])
            s_eval = eval_points
        elif end_param is not None:
            s_span = (0.0, end_param)
            s_eval = None
        else:
            raise ValueError("Необходимо задать end_param или eval_points")

        def rhs_wrapper(s, y):
            return self._rhs(s, y)

        try:
            s_vals, y_vals = self._solver.solve(
                rhs_wrapper, s_span, y0, t_eval=s_eval, **kwargs
            )
        except Exception as e:
            self._success = False
            self._diagnostics = {'success': False, 'message': f'Исключение: {str(e)}', 'exception': str(e)}
            return np.array([]), np.array([])

        self._s_values = s_vals
        self._uv_states = y_vals[:, [0, 2]]
        self._tangents = y_vals[:, [1, 3]]
        self._points_3d = np.array([self._surface.position(u, v) for u, v in self._uv_states])
        self._success = True
        self._diagnostics = {'success': True, 'message': 'Успешно завершено', 'num_points': len(s_vals)}
        return self._s_values, self._points_3d

    # ---------- геттеры ----------
    def get_uv_states(self) -> Optional[np.ndarray]:
        return self._uv_states

    def get_tangents(self) -> Optional[np.ndarray]:
        return self._tangents

    def get_3d_points(self) -> Optional[np.ndarray]:
        return self._points_3d

    @property
    def last_run_successful(self) -> bool:
        return self._success

    def get_diagnostics(self) -> dict:
        return self._diagnostics

    # ---------- вспомогательные методы ----------
    def set_rhs(self, rhs: Callable):
        self._rhs = rhs

    def _angle_to_derivatives(self, u, v, alpha):
        E, F, G = self._surface.first_fundamental_form(u, v)
        det = E*G - F*F
        sqrt_det = np.sqrt(max(det, self.eps))
        inv_sqrt_E = 1.0/np.sqrt(E) if E > self.eps else 0.0
        F_over_sqrt_det = F / sqrt_det
        u_prime = inv_sqrt_E * (np.cos(alpha) - F_over_sqrt_det * np.sin(alpha))
        v_prime = (np.sqrt(E) / sqrt_det) * np.sin(alpha)
        return u_prime, v_prime
# class ForwardWindingBuilder(WindingLineBuilderBase):
#     def __init__(
#         self,
#         surface: AnalyticalSurface,
#         deviation_law: DeviationLaw,
#         solver: ODESolver,
#         rhs: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
#         linear_solver: Optional[LocalLinearSolver] = None,
#         normalize_tangent: bool = True,
#         eps: float = 1e-12
#     ):
#         self._surface = surface
#         self._law = deviation_law
#         self._solver = solver
#         self._normalize = normalize_tangent
#         self.eps = eps

#         # Если RHS не передан, создаём по умолчанию
#         if rhs is not None:
#             self._rhs = rhs
#         else:
#             if linear_solver is None:
#                 linear_solver = GMRESSolver()
#             self._rhs = DelegatingForwardRHS(surface, deviation_law, linear_solver, eps=eps)

#         # результаты
#         self._s_values = None
#         self._uv_states = None
#         self._tangents = None
#         self._points_3d = None
#         self._success = False
#         self._diagnostics = {}
    
#     def set_rhs(self, rhs: Callable[[float, np.ndarray], np.ndarray]):
#         """Заменить функцию правых частей (например, для подключения другого решателя)."""
#         self._rhs = rhs
# #     def __init__(
# #         self,
# #         surface: AnalyticalSurface,
# #         deviation_law: DeviationLaw,
# #         solver: ODESolver,
# #         normalize_tangent: bool = True,
# #         eps: float = 1e-12
# #     ):
# #         self._surface = surface
# #         self._law = deviation_law
# #         self._solver = solver
# #         self._normalize = normalize_tangent
# #         self.eps = eps
# #         self._diagnostics = {}

# #         # Внутренний вычислитель правых частей
# #         # self._rhs = ForwardRHS(surface, deviation_law, normalize_tangent, eps)
# # #         self._rhs = QRegularizedForwardRHS(
# # #     surface, deviation_law,
# # #     q_param=1.1, adaptive_q=True, eps=1e-12
# # # )
# #         # if linear_solver is None:
# #         #     linear_solver = GMRESSolver()  # по умолчанию
# #         linear_solver = GMRESSolver()
# #         self._rhs = DelegatingForwardRHS(surface, deviation_law, linear_solver)

# #         # Результаты последнего расчёта
# #         self._s_values: Optional[np.ndarray] = None
# #         self._uv_states: Optional[np.ndarray] = None
# #         self._tangents: Optional[np.ndarray] = None
# #         self._points_3d: Optional[np.ndarray] = None
# #         self._success: bool = False

#     def build(
#     self,
#     initial_point: Tuple[float, float],
#     initial_tangent: Optional[Tuple[float, float]] = None,
#     end_param: Optional[float] = None,
#     eval_points: Optional[np.ndarray] = None,
#     **kwargs
#         ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Запускает построение линии укладки для прямой задачи.

#         Параметры
#         ---------
#         initial_point : (u0, v0)
#             Криволинейные координаты начальной точки на поверхности.
#         initial_tangent : tuple, optional
#             Начальное направление. Может быть задано двумя способами:
#             - (alpha,) : угол намотки α (в радианах), отсчитываемый от ru.
#             - (u_prime0, v_prime0) : явные производные du/ds, dv/ds.
#             Если None, возбуждается исключение.
#         end_param : float, optional
#             Конечная длина линии укладки s_end. Если не указана, требуется задать eval_points.
#         eval_points : np.ndarray, optional
#             Явный массив значений s, в которых необходимо вычислить решение.
#             Если указан, end_param игнорируется, интервал берётся по границам eval_points.
#         **kwargs
#             Дополнительные аргументы, передаваемые в solver.solve (rtol, atol, max_step и т.д.).

#         Возвращает
#         ----------
#         s_values : np.ndarray
#             Значения натурального параметра s в точках вывода.
#         points_3d : np.ndarray
#             Соответствующие 3D-точки линии укладки на поверхности, форма (N, 3).

#         Исключения
#         ----------
#         ValueError
#             Если не задано начальное направление, или не указан интервал интегрирования,
#             или поверхность вырождена в начальной точке.
#         """
#         u0, v0 = initial_point

#         # 1. Определение начальных производных u', v'
#         if initial_tangent is None:
#             raise ValueError("Для прямой задачи необходимо задать initial_tangent")

#         if len(initial_tangent) == 1:
#             # Задан угол намотки α
#             alpha = initial_tangent[0]
#             u_prime0, v_prime0 = self._angle_to_derivatives(u0, v0, alpha)
#         elif len(initial_tangent) == 2:
#             u_prime0, v_prime0 = initial_tangent
#         else:
#             raise ValueError(
#                 f"initial_tangent должен быть длины 1 (угол) или 2 (u', v'), "
#                 f"получено {len(initial_tangent)}"
#             )

#         # 2. Нормировка начального касательного вектора (опционально)
#         if self._normalize:
#             E, F, G = self._surface.first_fundamental_form(u0, v0)
#             norm2 = E * u_prime0**2 + 2.0 * F * u_prime0 * v_prime0 + G * v_prime0**2
#             if norm2 < self.eps:
#                 raise ValueError(f"Вырожденная метрика в начальной точке (u={u0}, v={v0})")
#             scale = 1.0 / np.sqrt(norm2)
#             u_prime0 *= scale
#             v_prime0 *= scale

#         # 3. Формирование начального вектора состояния
#         y0 = np.array([u0, u_prime0, v0, v_prime0], dtype=float)

#         # 4. Определение интервала интегрирования
#         if eval_points is not None:
#             s_span = (float(eval_points[0]), float(eval_points[-1]))
#             s_eval = eval_points
#         elif end_param is not None:
#             s_span = (0.0, float(end_param))
#             s_eval = None
#         else:
#             raise ValueError("Необходимо задать либо end_param, либо eval_points")

#         # 5. Оборачиваем правые части для решателя
#         def rhs_wrapper(s: float, y: np.ndarray) -> np.ndarray:
#             return self._rhs(s, y)

#         # # 6. Запуск численного интегрирования
#         # s_vals, y_vals = self._solver.solve(
#         #     fun=rhs_wrapper,
#         #     t_span=s_span,
#         #     y0=y0,
#         #     t_eval=s_eval,
#         #     **kwargs
#         # )
#         try:
#             # s_vals, y_vals, diag = self._solver.solve_with_diagnostics(
#             #     rhs_wrapper, s_span, y0, t_eval=s_eval, **kwargs
#             # )
#             s_vals, y_vals = self._solver.solve(rhs_wrapper, s_span, y0, t_eval=s_eval, **kwargs)
# # Успех или ошибка будет в исключении, если status!=0 (мы кидаем RuntimeError)
# # Тогда блок try/except уже есть в build
#         except Exception as e:
#             self._diagnostics = {
#                 'success': False,
#                 'message': f'Исключение: {str(e)}',
#                 'num_points': 0,
#                 'final_param': s_span[0]
#             }
#             self._success = False
#             return np.array([]), np.array([])

#         self._diagnostics = {
#             'success': diag['success'],
#             'message': diag['message'],
#             'num_points': len(s_vals),
#             'final_param': diag['final_t'],
#             'solver_message': diag['solver_message']
#         }
#         if not diag['success']:
#             self._success = False
#             return s_vals, np.array([])  # или возвращаем частичные точки


#         # 7. Сохранение результатов
#         self._s_values = s_vals
#         self._uv_states = y_vals[:, [0, 2]]      # колонки u, v
#         self._tangents = y_vals[:, [1, 3]]       # колонки u', v'
#         self._points_3d = np.array([
#             self._surface.position(u, v) for u, v in self._uv_states
#         ])
#         self._success = True

#         return self._s_values, self._points_3d
#     def _angle_to_derivatives(self, u: float, v: float, alpha: float) -> Tuple[float, float]:
#         """
#         Преобразует угол намотки α в производные криволинейных координат u', v'
#         согласно формулам (2.21) диссертации.

#         Угол α отсчитывается от координатного вектора ru в касательной плоскости.
#         """
#         E, F, G = self._surface.first_fundamental_form(u, v)
#         det = E * G - F * F
#         sqrt_det = np.sqrt(max(det, self.eps))

#         inv_sqrt_E = 1.0 / np.sqrt(E) if E > self.eps else 0.0
#         F_over_sqrt_det = F / sqrt_det

#         u_prime = inv_sqrt_E * (np.cos(alpha) - F_over_sqrt_det * np.sin(alpha))
#         v_prime = (np.sqrt(E) / sqrt_det) * np.sin(alpha)

#         return u_prime, v_prime
    
#     def get_uv_states(self) -> Optional[np.ndarray]:
#         return self._uv_states

#     def get_tangents(self) -> Optional[np.ndarray]:
#         return self._tangents

#     def get_3d_points(self) -> Optional[np.ndarray]:
#         return self._points_3d
#     def get_diagnostics(self) -> dict:
#         return self._diagnostics

#     @property
#     def last_run_successful(self) -> bool:
#         return self._success