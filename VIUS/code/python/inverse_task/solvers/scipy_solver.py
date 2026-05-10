# # winding/scipy_solver.py
# from scipy.integrate import solve_ivp
# import numpy as np
# from .base_solver import ODESolver

# # solvers/scipy_solver.py
# from typing import Optional, Tuple, Dict, Any, List, Callable
# import numpy as np
# from scipy.integrate import solve_ivp
# from .base_solver import ODESolver


# solvers/scipy_solver.py

# from typing import Optional, Tuple, Dict, Any, List, Callable
# import numpy as np
# from scipy.integrate import solve_ivp
# from .base_solver import ODESolver

# solvers/scipy_solver.py
from typing import Optional, Tuple, Dict, Any, List, Callable
import numpy as np
from scipy.integrate import solve_ivp
from .base_solver import ODESolver
from observers.base_observer import Observer

class SciPySolver(ODESolver):
    def __init__(
        self,
        method: str = 'RK45',
        events: Optional[List[Callable]] = None,
        **kwargs
    ):
        super().__init__()   # если ODESolver имеет свой __init__, иначе просто self._observers = []
        self.method = method
        self.events = events
        self.kwargs = kwargs
        self._observers: List[Observer] = []

    def add_observer(self, observer: Observer) -> None:
        self._observers.append(observer)

    def solve(
        self,
        fun: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Колбэк для передачи данных наблюдателям после каждого шага
        def callback(t: float, y: np.ndarray):
            u, v = y[0], y[2]  # для системы (2.10)
            data = {
                's': t,
                'u': u,
                'v': v,
                'step_size': 0.0   # точный размер шага в колбэке недоступен, можно заменить на None
            }
            for obs in self._observers:
                obs.update('step', data)
            return 0

        sol = solve_ivp(
            fun,
            t_span,
            y0,
            method=self.method,
            t_eval=t_eval,
            events=self.events,
            callback=callback,
            **self.kwargs
        )

        # Обработка событий
        if sol.t_events is not None:
            for i, event_times in enumerate(sol.t_events):
                if event_times is not None:
                    for t in event_times:
                        for obs in self._observers:
                            obs.update('event_triggered', {
                                'event_name': f'event_{i}',
                                's': t
                            })

        if sol.status != 0:
            raise RuntimeError(
                f"Интегрирование завершилось с ошибкой "
                f"(status={sol.status}): {sol.message}"
            )

        return sol.t, sol.y.T

