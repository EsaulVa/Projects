
# predictors/dae_predictor.py
from .inverse_method import *
from .predictor_base import *
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple
from solvers.base_solver import ODESolver

class DAEPredictor(Predictor):
    """Предиктор на основе DAE-дифференцирования связи с встроенным решателем ОДУ."""
    def __init__(self, solver: ODESolver):
        self.solver = solver

    def predict(self, z_k, z_next, u_cur, v_cur, surface, traj):
        # Функция правых частей
        def rhs(z, uv):
            du, dv = compute_dr_dz(surface, traj, uv[0], uv[1], z)
            return np.array([du, dv])

        try:
            # Используем переданный решатель (он сам разберётся с методом и точностями)
            t_vals, y_vals = self.solver.solve(
                rhs, (z_k, z_next), np.array([u_cur, v_cur])
            )
            if len(t_vals) < 2:   # решатель не смог продвинуться
                return None
            u_pred, v_pred = y_vals[-1]   # последняя точка соответствует z_next
            # u_pred, v_pred = y_vals[-1]            
            # Проверка: шаг не должен быть катастрофическим
            du = u_pred - u_cur
            dv = v_pred - v_cur
            if abs(du) > 50.0 or abs(dv) > 5.0:  # разумные границы для вашей геометрии
                return None  # отклоняем, пойдём на fallback/оптику
        except Exception:
            return None

        # Принудительное соблюдение границ (если они определены)
        if hasattr(surface, 'u_min'):
            u_pred = np.clip(u_pred, surface.u_min, surface.u_max)
        if hasattr(surface, 'v_min'):
            v_pred = np.clip(v_pred, surface.v_min, surface.v_max)
        return u_pred, v_pred