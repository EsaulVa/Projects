# observers/linear_solver_observer.py
from .base_observer import Observer

class LinearSolverObserver(Observer):
    """Собирает информацию о каждом решении локальной системы A·x = b."""
    
    def __init__(self):
        self.reset()
    
    def update(self, event: str, data: Dict):
        if event == 'solve':
            self.det_A.append(data.get('det_A', float('nan')))
            self.cond_A.append(data.get('cond_A', float('nan')))
            self.q_used.append(data.get('q_used', None))
            self.residual.append(data.get('residual', float('nan')))
            self.solution_norm.append(data.get('solution_norm', float('nan')))
    
    def reset(self):
        self.det_A = []
        self.cond_A = []
        self.q_used = []
        self.residual = []
        self.solution_norm = []
    
    def report(self):
        if not self.det_A:
            return {}
        import numpy as np
        return {
            'min_det_A': np.min(self.det_A),
            'max_cond_A': np.max(self.cond_A),
            'mean_q': np.mean([q for q in self.q_used if q is not None]) if self.q_used else None,
            'max_residual': np.max(self.residual),
            'max_solution_norm': np.max(self.solution_norm),
        }