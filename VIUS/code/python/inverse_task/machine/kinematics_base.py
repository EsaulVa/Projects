import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import root

class MachineState:
    """Универсальное состояние станка для любого числа осей."""
    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        
    @property
    def size(self):
        return len(self.coords)

from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import root

class MachineState:
    """Универсальное состояние станка для любого числа осей."""
    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)

class IMachineKinematics(ABC):
    """Абстрактный класс для кинематики намоточных станков."""
    
    @abstractmethod
    def forward(self, state: MachineState) -> dict:
        """
        Прямая задача.
        Возвращает словарь с 3D позицией и ориентацией выходного ролика.
        """
        pass

    @abstractmethod
    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
        """
        Система нелинейных уравнений (невязки).
        """
        pass

    def inverse(self, target_data: dict, initial_guess: MachineState) -> MachineState:
        """Обратная задача."""
        # Эту часть менять не нужно, она универсальна
        def func_to_solve(x_opt):
            state_temp = MachineState(x_opt)
            return self.residuals(target_data, state_temp)
            
        sol = root(func_to_solve, initial_guess.coords, method='hybr')
        if sol.success:
            return MachineState(sol.x)
        else:
            raise RuntimeError(f"Кинематика не сошлась: {sol.message}")

import numpy as np
from core.trajectory import Trajectory
from machine.kinematics_base import IMachineKinematics, MachineState

class TrajectoryDeployer:
    """
    Проецирует пространственную траекторию ТСН на обобщенные координаты станка.
    Адаптирован под полные кинематические модели (по Приложению I),
    требующие как точки, так и вектора касательной для ориентации ролика.
    """
    def __init__(self, machine: IMachineKinematics):
        self.machine = machine

    def deploy(self, tsn_trajectory: Trajectory, theta_array: np.ndarray) -> dict:
        N = len(theta_array)
        
        # Расширяем историю: теперь храним 4 координаты [theta, Z, R, phi]
        history_theta = np.zeros(N)
        history_Z = np.zeros(N)
        history_R = np.zeros(N)
        history_phi = np.zeros(N) # НОВАЯ КООРДИНАТА
        history_3d = np.zeros((N, 3))
        success_flags = np.ones(N, dtype=bool)
        
        # 1. Точное решение для нулевой точки (пользователь должен задать хороший guess)
        p0 = tsn_trajectory.R(0.0)
        tau0 = tsn_trajectory.R_deriv(0.0)
        
        # Начальный guess должен содержать 4 элемента (включая начальный phi=0)
        guess_0 = MachineState(np.array([theta_array[0], 150.0, 250.0, 0.0])) 
        
        try:
            state_0 = self.machine.inverse({'point': p0, 'tau': tau0}, guess_0)
            history_theta[0], history_Z[0], history_R[0], history_phi[0] = state_0.coords
            history_3d[0] = self.machine.forward(state_0)['point']
        except RuntimeError:
            success_flags[0] = False

        # 2. Идем по шагам
        for i in range(1, N):
            s_val = tsn_trajectory.total_length * (i / (N - 1))
            
            # Формируем целевые данные (точка + касательная)
            target_3d = tsn_trajectory.R(s_val)
            target_tau = tsn_trajectory.R_deriv(s_val)
            target_data = {'point': target_3d, 'tau': target_tau}
            
            # Начальное приближение: берем ВСЕ 4 координаты с предыдущего шага
            # (включая угол theta, так как для сложных поверхностей он может немного "плавать")
            guess = MachineState(np.array([
                history_theta[i-1], 
                history_Z[i-1], 
                history_R[i-1], 
                history_phi[i-1]
            ]))
            
            try:
                state_i = self.machine.inverse(target_data, guess)
                history_theta[i], history_Z[i], history_R[i], history_phi[i] = state_i.coords
                history_3d[i] = self.machine.forward(state_i)['point']
            except RuntimeError as e:
                # print(f"Ошибка на шаге {i}: {e}")
                success_flags[i] = False
                history_theta[i] = history_theta[i-1]
                history_Z[i] = history_Z[i-1]
                history_R[i] = history_R[i-1]
                history_phi[i] = history_phi[i-1]
                history_3d[i] = history_3d[i-1]

        return {
            's_array': np.linspace(0, tsn_trajectory.total_length, N),
            'theta': history_theta, # Фактический угол, с которым решалась система
            'Z': history_Z,
            'R': history_R,
            'phi': history_phi,     # Динамика угла по кольцу раскладчика
            'tsn_actual_3d': history_3d, 
            'success': success_flags
        }