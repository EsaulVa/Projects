# trajectory.py
import numpy as np
# from scipy.integrate import cumtrapz
# Замените строку:
# from scipy.integrate import cumtrapz
# на следующий условный импорт:
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from .curve_interface import Curve1D
from .curve_factory import CurveFactory
from .register_factory import *

class Trajectory:
    def __init__(self, curve_x: Curve1D, curve_y: Curve1D, curve_z: Curve1D):
        self._curve_x = curve_x
        self._curve_y = curve_y
        self._curve_z = curve_z
        self._u_min = max(curve_x.domain[0], curve_y.domain[0], curve_z.domain[0])
        self._u_max = min(curve_x.domain[1], curve_y.domain[1], curve_z.domain[1])
        self._build_arc_length_mapping()

    @classmethod
    def from_points(cls, points: np.ndarray, method: str = 'cubic', **kwargs):
        """
        Создаёт траекторию из массива точек (N, 3), используя указанный метод.
        Дополнительные kwargs передаются в фабрику кривых.
        """
        u = cls._chord_parameterization(points)
        curve_x = CurveFactory.create(u, points[:, 0], method, **kwargs)
        curve_y = CurveFactory.create(u, points[:, 1], method, **kwargs)
        curve_z = CurveFactory.create(u, points[:, 2], method, **kwargs)
        return cls(curve_x, curve_y, curve_z)

    @staticmethod
    def _chord_parameterization(points: np.ndarray) -> np.ndarray:
        diffs = np.diff(points, axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        u = np.zeros(points.shape[0])
        u[1:] = np.cumsum(dists)
        return u

    def _build_arc_length_mapping(self, num_samples: int = 5000):
        u_fine = np.linspace(self._u_min, self._u_max, num_samples)
        dx_du = np.array([self._curve_x.evaluate(ui, 1) for ui in u_fine])
        dy_du = np.array([self._curve_y.evaluate(ui, 1) for ui in u_fine])
        dz_du = np.array([self._curve_z.evaluate(ui, 1) for ui in u_fine])
        speed = np.sqrt(dx_du**2 + dy_du**2 + dz_du**2)
        s_fine = np.zeros_like(u_fine)
        s_fine[1:] = cumtrapz(speed, u_fine)
        self._s_func = lambda u: np.interp(u, u_fine, s_fine)
        self._u_func = lambda s: np.interp(s, s_fine, u_fine)
        self._total_length = s_fine[-1]

    def R(self, s: float) -> np.ndarray:
        z = np.clip(s, 0.0, self._total_length)
        u = self._u_func(z)
        return np.array([
            self._curve_x.evaluate(u),
            self._curve_y.evaluate(u),
            self._curve_z.evaluate(u)
        ])

    def R_deriv(self, s: float) -> np.ndarray:
        z = np.clip(s, 0.0, self._total_length)
        u = self._u_func(z)
        dx_du = self._curve_x.evaluate(u, 1)
        dy_du = self._curve_y.evaluate(u, 1)
        dz_du = self._curve_z.evaluate(u, 1)
        dR_du = np.array([dx_du, dy_du, dz_du])
        ds_du = np.linalg.norm(dR_du)
        return dR_du / ds_du

    @property
    def total_length(self) -> float:
        return self._total_length