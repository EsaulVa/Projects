# observers/ode_observer.py
from .base_observer import Observer

class ODEObserver(Observer):
    """Собирает статистику по шагам ОДУ."""
    
    def __init__(self):
        self.reset()
    
    def update(self, event: str, data: Dict):
        if event == 'step':
            self.s.append(data.get('s', 0.0))
            self.u.append(data.get('u', 0.0))
            self.v.append(data.get('v', 0.0))
            self.tau_err.append(data.get('tau_norm_err', 0.0))
            self.tg_theta.append(data.get('tg_theta', 0.0))
            self.kN.append(data.get('normal_curvature', 0.0))
            self.step_size.append(data.get('step_size', 0.0))
        elif event == 'event_triggered':
            self.events_hit.append(data.get('event_name', 'unknown'))
    
    def reset(self):
        self.s = []
        self.u = []
        self.v = []
        self.tau_err = []
        self.tg_theta = []
        self.kN = []
        self.step_size = []
        self.events_hit = []
    
    def report(self):
        if not self.s:
            return {}
        import numpy as np
        return {
            'n_steps': len(self.s),
            'final_s': self.s[-1],
            'max_tau_err': np.max(np.abs(self.tau_err)),
            'min_step': np.min(self.step_size) if self.step_size else None,
            'max_step': np.max(self.step_size) if self.step_size else None,
            'events_triggered': self.events_hit,
        }