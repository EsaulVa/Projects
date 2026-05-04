# solvers/local_linear_solver.py
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from observers.base_observer import Observer

class LocalLinearSolver(ABC):
    def __init__(self):
        self._observers: List[Observer] = []
    
    def add_observer(self, observer: Observer):
        self._observers.append(observer)
    
    def _notify_observers(self, event: str, data: Dict):
        for obs in self._observers:
            obs.update(event, data)
    
    @abstractmethod
    def solve(self, A, b) -> tuple[float, float]:
        pass

from scipy.sparse.linalg import gmres

# solvers/local_linear_solver.py (фрагмент)
from scipy.sparse.linalg import gmres
import numpy as np

class GMRESSolver(LocalLinearSolver):
    def __init__(self, rtol=1e-8, atol=0.0):
        super().__init__()          # инициализирует список наблюдателей
        self.rtol = rtol            # сохраняем параметры
        self.atol = atol

    def solve(self, A, b):
        # ATA = A.T @ A
        # ATb = A.T @ b
        x, info = gmres(A, b, rtol=self.rtol, atol=self.atol)
        if info != 0:
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        residual = np.linalg.norm(A @ x - b)
        det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        cond_est = np.linalg.norm(A, 'fro') / max(abs(det_A), 1e-12) if abs(det_A) > 0 else np.inf

        self._notify_observers('solve', {
            'det_A': det_A,
            'cond_A': cond_est,
            'q_used': None,
            'residual': residual,
            'solution_norm': np.linalg.norm(x)
        })
        return x[0], x[1]
