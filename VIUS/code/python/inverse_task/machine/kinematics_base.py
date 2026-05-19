# # machine/kinematics_base.py
# import numpy as np
# from abc import ABC, abstractmethod

# class MachineState:
#     def __init__(self, coords: np.ndarray):
#         self.coords = np.asarray(coords, dtype=float)

#     @property
#     def size(self):
#         return len(self.coords)
# kinematics_base.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class MachineState:
    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
    @property
    def size(self):
        return len(self.coords)

class IMachineKinematics(ABC):
    @abstractmethod
    def forward(self, state: MachineState) -> dict:
        """Прямая кинематика: возвращает {'point': ..., 'd': ..., 'n': ...}"""
        pass

    @abstractmethod
    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
        """Невязки уравнений (4 уравнения для 3-х координатного станка)"""
        pass

    @abstractmethod
    def get_ode_data(self, s: float, q: np.ndarray,
                     target_point: np.ndarray, r_mandrel: np.ndarray,
                     d_target_ds: np.ndarray, d_mandrel_ds: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Возвращает (J, dF_ds), где J = dF/dq (матрица m×n),
        dF_ds – частная производная F по явной зависимости от s.
        """
        pass

    def inverse(self, target_data: dict, initial_guess: MachineState) -> MachineState:
        """Обратная кинематика (итерационная) – может быть переопределена."""
        from scipy.optimize import least_squares
        def func(x):
            return self.residuals(target_data, MachineState(x))
        res = least_squares(func, initial_guess.coords, method='lm',
                            max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if res.success and np.linalg.norm(res.fun) < 1e-9:
            return MachineState(res.x)
        else:
            raise RuntimeError(f"Inverse failed: {res.message}, norm={np.linalg.norm(res.fun)}")

# class IMachineKinematics(ABC):
#     @abstractmethod
#     def forward(self, state: MachineState) -> dict:
#         pass

#     @abstractmethod
#     def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
#         pass

#     def inverse(self, target_data: dict, initial_guess: MachineState) -> MachineState:
#         from scipy.optimize import root
#         def func(x):
#             state_temp = MachineState(x)
#             return self.residuals(target_data, state_temp)
#         sol = root(func, initial_guess.coords, method='hybr')
#         if sol.success:
#             return MachineState(sol.x)
#         else:
#             raise RuntimeError(f"Inverse failed: {sol.message}")
#     def get_ode_data(self, s, q, target_point, r_mandrel, d_target_ds, d_mandrel_ds):
#         # Реализация по умолчанию: численное дифференцирование
#         eps = 1e-7
#         n = len(q)
#         F0 = self.residuals({'point': target_point, 'r_mandrel': r_mandrel}, MachineState(q))
#         J = np.zeros((len(F0), n))
#         for i in range(n):
#             q_plus = q.copy(); q_plus[i] += eps
#             q_minus = q.copy(); q_minus[i] -= eps
#             F_plus = self.residuals({'point': target_point, 'r_mandrel': r_mandrel}, MachineState(q_plus))
#             F_minus = self.residuals({'point': target_point, 'r_mandrel': r_mandrel}, MachineState(q_minus))
#             J[:, i] = (F_plus - F_minus) / (2*eps)
#         # dF_ds численно:
#         eps_s = 1e-6
#         F_s_plus = self.residuals({'point': target_point + d_target_ds*eps_s,
#                                    'r_mandrel': r_mandrel + d_mandrel_ds*eps_s}, MachineState(q))
#         F_s_minus = self.residuals({'point': target_point - d_target_ds*eps_s,
#                                     'r_mandrel': r_mandrel - d_mandrel_ds*eps_s}, MachineState(q))
#         dF_ds = (F_s_plus - F_s_minus) / (2*eps_s)
#         return J, dF_ds