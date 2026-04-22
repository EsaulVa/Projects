# curve_factory.py
from typing import Callable, Dict, Any
import numpy as np
from .curve_interface import Curve1D

class CurveFactory:
    _builders: Dict[str, Callable[..., Curve1D]] = {}

    @classmethod
    def register(cls, method: str):
        """Декоратор для регистрации строителя кривой."""
        def decorator(func: Callable[..., Curve1D]):
            cls._builders[method] = func
            return func
        return decorator

    @classmethod
    def create(cls, u: np.ndarray, values: np.ndarray,
               method: str = 'cubic', **kwargs) -> Curve1D:
        if method not in cls._builders:
            available = list(cls._builders.keys())
            raise ValueError(f"Unknown curve method '{method}'. Available: {available}")
        builder = cls._builders[method]
        return builder(u, values, **kwargs)

    @classmethod
    def available_methods(cls):
        return list(cls._builders.keys())