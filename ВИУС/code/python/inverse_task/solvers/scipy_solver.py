# winding/scipy_solver.py
from scipy.integrate import solve_ivp
import numpy as np
from .base_solver import ODESolver

class SciPySolver(ODESolver):
    def __init__(self, method: str = 'RK45', **kwargs):
        """
        method: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        **kwargs: дополнительные параметры для solve_ivp (rtol, atol, ...)
        """
        self.method = method
        self.kwargs = kwargs

    def solve(self, fun, t_span, y0, t_eval=None):
        # SciPy ожидает порядок (t, y), а не (z, state)!
        def scipy_fun(t, y):
            return fun(t, y)
        
        sol = solve_ivp(
            scipy_fun, t_span, y0,
            method=self.method,
            t_eval=t_eval,
            **self.kwargs
        )
        return sol.t, sol.y.T  # Транспонируем для единообразия