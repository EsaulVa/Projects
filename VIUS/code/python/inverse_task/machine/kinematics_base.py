# machine/kinematics_base.py
import numpy as np
from abc import ABC, abstractmethod

class MachineState:
    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)

    @property
    def size(self):
        return len(self.coords)

class IMachineKinematics(ABC):
    @abstractmethod
    def forward(self, state: MachineState) -> dict:
        pass

    @abstractmethod
    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
        pass

    def inverse(self, target_data: dict, initial_guess: MachineState) -> MachineState:
        from scipy.optimize import root
        def func(x):
            state_temp = MachineState(x)
            return self.residuals(target_data, state_temp)
        sol = root(func, initial_guess.coords, method='hybr')
        if sol.success:
            return MachineState(sol.x)
        else:
            raise RuntimeError(f"Inverse failed: {sol.message}")