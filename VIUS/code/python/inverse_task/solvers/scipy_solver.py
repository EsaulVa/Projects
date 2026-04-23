# winding/scipy_solver.py
from scipy.integrate import solve_ivp
import numpy as np
from .base_solver import ODESolver

# solvers/scipy_solver.py
from typing import Optional, Tuple, Dict, Any, List, Callable
import numpy as np
from scipy.integrate import solve_ivp
from .base_solver import ODESolver


# solvers/scipy_solver.py

from typing import Optional, Tuple, Dict, Any, List, Callable
import numpy as np
from scipy.integrate import solve_ivp
from .base_solver import ODESolver


class SciPySolver(ODESolver):
    def __init__(
        self,
        method: str = 'RK45',
        events: Optional[List[Callable]] = None,
        **kwargs
    ):
        self.method = method
        self.events = events
        self.kwargs = kwargs
        self._last_diagnostics: Optional[Dict[str, Any]] = None

    def solve(
        self,
        fun: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        t, y, diag = self.solve_with_diagnostics(fun, t_span, y0, t_eval)

        if not diag['success']:
            # Безопасное форматирование: final_t может быть числом или строкой
            final_t = diag.get('final_t')
            if final_t is None:
                final_t_str = "None"
            else:
                final_t_str = f"{final_t:.6e}"
            raise RuntimeError(
                f"Интегрирование не удалось: {diag['message']}\n"
                f"Достигнуто t = {final_t_str} из {diag['t_bound']:.6e}\n"
                f"Количество шагов: {diag['num_steps']}"
            )

        if t_eval is not None and len(t) < len(t_eval):
            import warnings
            warnings.warn(
                f"Решатель остановился досрочно: получено {len(t)} точек из {len(t_eval)} запрошенных. "
                f"Последний t = {t[-1]:.6f}"
            )

        return t, y

    def solve_with_diagnostics(
        self,
        fun: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        diagnostics = {
            'success': False,
            'message': '',
            'status': None,
            'num_steps': 0,
            'final_t': t_span[0],  # начальное значение по умолчанию
            't_bound': t_span[1],
            'solver_message': ''
        }

        try:
            sol = solve_ivp(
                fun,
                t_span,
                y0,
                method=self.method,
                t_eval=t_eval,
                events=self.events,
                **self.kwargs
            )
        except Exception as e:
            diagnostics['message'] = f"Исключение в solve_ivp: {str(e)}"
            diagnostics['final_t'] = t_span[0]  # не продвинулись
            self._last_diagnostics = diagnostics
            return (
                np.array([]),
                np.array([]).reshape(0, len(y0)),
                diagnostics
            )

        diagnostics['status'] = sol.status
        diagnostics['num_steps'] = len(sol.t)
        diagnostics['final_t'] = sol.t[-1] if len(sol.t) > 0 else t_span[0]
        diagnostics['solver_message'] = sol.message

        if sol.status == 0:
            diagnostics['success'] = True
            diagnostics['message'] = 'Успешно достигнут конец интервала'
        elif sol.status == 1:
            diagnostics['success'] = False
            diagnostics['message'] = (
                f"Остановка по терминальному событию. "
                f"Сообщение решателя: {sol.message}"
            )
        elif sol.status == -1:
            diagnostics['success'] = False
            diagnostics['message'] = (
                "Интегрирование прервано: шаг стал слишком мал или "
                "не удалось достичь требуемой точности. "
                "Возможные причины: жёсткость системы, сингулярность, "
                "выход за область определения."
            )
        else:
            diagnostics['success'] = False
            diagnostics['message'] = f"Неизвестный статус {sol.status}: {sol.message}"

        self._last_diagnostics = diagnostics
        return sol.t, sol.y.T, diagnostics
    # def get_diagnostics(self) -> dict:
    #         return self._diagnostics
    @property
    def last_diagnostics(self) -> Optional[Dict[str, Any]]:
        return self._last_diagnostics
# class SciPySolver(ODESolver):
#     def __init__(self, method: str = 'RK45', **kwargs):
#         """
#         method: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
#         **kwargs: дополнительные параметры для solve_ivp (rtol, atol, ...)
#         """
#         self.method = method
#         self.kwargs = kwargs

#     def solve(self, fun, t_span, y0, t_eval=None):
#         # SciPy ожидает порядок (t, y), а не (z, state)!
#         def scipy_fun(t, y):
#             return fun(t, y)
        
#         sol = solve_ivp(
#             scipy_fun, t_span, y0,
#             method=self.method,
#             t_eval=t_eval,
#             **self.kwargs
#         )
#         if sol.status == 1:
#             raise RuntimeError(f"Решатель остановлен событием: {sol.message}")
#         elif sol.status != 0:
#          raise RuntimeError(f"Решатель завершился некорректно (status={sol.status}): {sol.message}")
#     # Если t_eval не None, может быть, что sol.t не содержит всю сетку
#         if t_eval is not None and len(sol.t) < len(t_eval):
#             print(f"Предупреждение: решатель остановился досрочно, длина t_eval={len(t_eval)}, получено точек {len(sol.t)}")
#         return sol.t, sol.y.T  # Транспонируем для единообразия