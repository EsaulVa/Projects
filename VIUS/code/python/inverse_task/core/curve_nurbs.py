import numpy as np
from geomdl import BSpline
from geomdl import utilities
from .curve_interface import Curve1D

import numpy as np
from geomdl import BSpline
from geomdl import utilities
from core.curve_interface import Curve1D

class NURBSCurve(Curve1D):
    def __init__(self, u, values, degree=3, **kwargs):
        self._u_original = np.asarray(u).flatten()
        self._values = np.asarray(values).flatten()
        self._degree = degree
        
        # Нормализуем параметр u к диапазону [0, 1]
        self._u_min = self._u_original[0]
        self._u_max = self._u_original[-1]
        self._u_range = self._u_max - self._u_min
        
        # Нормализованные параметры для контрольных точек
        u_norm = (self._u_original - self._u_min) / self._u_range
        
        # Создаём B-сплайн кривую
        self._curve = BSpline.Curve()
        self._curve.degree = degree
        
        # Контрольные точки: (u_norm, value)
        ctrlpts = [[float(ui), float(vi)] for ui, vi in zip(u_norm, self._values)]
        self._curve.ctrlpts = ctrlpts
        
        # Узловой вектор (равномерный)
        self._curve.knotvector = utilities.generate_knot_vector(degree, len(ctrlpts))
        
    def _normalize_u(self, u: float) -> float:
        return (u - self._u_min) / self._u_range
    
    def evaluate(self, u: float, der: int = 0) -> float:
        if der > 2:
            raise ValueError("NURBSCurve supports derivatives up to order 2")
        # Ограничиваем u областью определения
        u_clamped = max(self._u_original[0], min(self._u_original[-1], u))
        u_norm = self._normalize_u(u_clamped)
        
        if der == 0:
            pt = self._curve.evaluate_single(u_norm)
            return float(pt[1])
        else:
            # Получаем производные по нормализованному параметру
            ders = self._curve.derivatives(u_norm, order=der)
            # ders[k] — k-я производная вектора (x(u_norm), y(u_norm))
            # Нам нужна производная y по u: dy/du = (dy/du_norm) * (du_norm/du)
            dy_du_norm = float(ders[der][1])
            du_norm_du = 1.0 / self._u_range
            if der == 1:
                return dy_du_norm * du_norm_du
            elif der == 2:
                # Вторая производная: d²y/du² = (d²y/du_norm²) * (du_norm/du)²
                d2y_du_norm2 = float(ders[2][1])
                return d2y_du_norm2 * (du_norm_du ** 2)
    
    @property
    def domain(self) -> tuple[float, float]:
        return (self._u_original[0], self._u_original[-1])