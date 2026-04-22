# geometry/cylinder.py
import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple
from .tsurfaces import AnalyticalSurface

class CylinderAnalytical(AnalyticalSurface):
    """Базовый цилиндр с аналитическими формами."""
    def __init__(self, radius: float = 1.0):
        self.R = radius

    def position(self, u, v):
        return jnp.array([self.R * jnp.cos(u), self.R * jnp.sin(u), v])

    def normal(self, u, v):
        return jnp.array([jnp.cos(u), jnp.sin(u), 0.0])

    def first_fundamental_form(self, u, v) -> Tuple[float, float, float]:
        return self.R**2, 0.0, 1.0

    def second_fundamental_form(self, u, v) -> Tuple[float, float, float]:
        return -self.R, 0.0, 0.0

    def metric_derivatives(self, u, v):
        # Все производные метрики равны нулю, т.к. E, F, G константы
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def derivatives(self, u, v) -> Dict[str, jnp.ndarray]:
        cos_u = jnp.cos(u)
        sin_u = jnp.sin(u)
        r = jnp.array([self.R * cos_u, self.R * sin_u, v])
        ru = jnp.array([-self.R * sin_u, self.R * cos_u, 0.0])
        rv = jnp.array([0.0, 0.0, 1.0])
        normal = jnp.array([cos_u, sin_u, 0.0])
        return {'r': r, 'ru': ru, 'rv': rv, 'normal': normal}


class CylinderWithDerivatives(AnalyticalSurface):
    """Цилиндр, расширенный методом derivatives, использующий композицию."""
    def __init__(self, radius: float = 1.0):
        self._cylinder = CylinderAnalytical(radius)
        self.R = radius

    def position(self, u, v):
        return self._cylinder.position(u, v)

    def normal(self, u, v):
        return self._cylinder.normal(u, v)

    def first_fundamental_form(self, u, v):
        return self._cylinder.first_fundamental_form(u, v)

    def second_fundamental_form(self, u, v):
        return self._cylinder.second_fundamental_form(u, v)

    def metric_derivatives(self, u, v):
        return self._cylinder.metric_derivatives(u, v)

    def derivatives(self, u, v) -> Dict[str, jnp.ndarray]:
        return self._cylinder.derivatives(u, v)