from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple
import numpy as np

class ODEIntegrator(ABC):
    """Абстрактный интегратор обыкновенных дифференциальных уравнений."""
    
    @abstractmethod
    def solve(
        self,
        z_span: Tuple[float, float],
        initial_state: np.ndarray,
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Решает задачу Коши на интервале z_span.
        
        Параметры
        ---------
        z_span : (z_start, z_end)
        initial_state : начальный вектор состояния (например, [u0, v0])
        callback : функция, вызываемая после каждого шага; может использоваться
                   для сохранения промежуточных данных или досрочной остановки.
        
        Возвращает
        ----------
        z_values : массив значений независимой переменной
        states : массив состояний в соответствующих точках
        """
        pass

# winding/ode_solver.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Tuple

class ODESolver(ABC):
    """Абстрактный интерфейс для всех решателей ОДУ."""
    
    @abstractmethod
    def solve(
        self,
        fun: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        t_eval: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Решить задачу Коши dy/dt = fun(t, y).
        
        Параметры
        ---------
        fun : callable
            Функция правых частей, принимающая t (float) и y (np.ndarray)
        t_span : (t_start, t_end)
            Интервал интегрирования
        y0 : np.ndarray
            Начальное состояние
        t_eval : np.ndarray, optional
            Точки, в которых нужно вычислить решение
            
        Возвращает
        ----------
        t : np.ndarray
            Массив времен
        y : np.ndarray
            Массив состояний (shape: (len(t), len(y0)))
        """
        pass