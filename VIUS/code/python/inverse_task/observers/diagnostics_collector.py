# observers/diagnostics_collector.py
from .base_observer import Observer
from .linear_solver_observer import LinearSolverObserver
from .ode_observer import ODEObserver

class DiagnosticsCollector(Observer):
    """
    Агрегирует данные от LinearSolverObserver и ODEObserver.
    Может использоваться как единый наблюдатель для всего процесса.
    """
    def __init__(self):
        self.lin_observer = LinearSolverObserver()
        self.ode_observer = ODEObserver()
    
    def update(self, event, data):
        # маршрутизация в зависимости от источника
        if event == 'solve':
            self.lin_observer.update(event, data)
        elif event in ('step', 'event_triggered'):
            self.ode_observer.update(event, data)
    
    def reset(self):
        self.lin_observer.reset()
        self.ode_observer.reset()
    
    def report(self):
        return {
            'linear_solver': self.lin_observer.report(),
            'ode_solver': self.ode_observer.report(),
        }