# predictors/optical_predictor.py
import numpy as np
from .predictor_base import Predictor
from .intersection import RayTracer

class OpticalPredictor(Predictor):
    """
    Оптический предиктор: находит следующую точку укладки трассировкой луча
    от точки схода к текущей точке на поверхности оправки.
    Использует RayTracer для пересечения луча с поверхностью и uv_from_point
    для обратного проецирования.
    """
    def __init__(self, ray_tracer: RayTracer):
        self.ray_tracer = ray_tracer

    def predict(self, z_k, z_next, u_cur, v_cur, surface, traj):
        # 1. Текущая точка укладки на оправке
        r_cur = surface.position(u_cur, v_cur)
        
        # 2. Следующая точка траектории (раскладчика)
        R_next = traj.R(z_next)
        
        # 3. Направление луча от R_next к текущей точке укладки
        delta = r_cur - R_next
        length = np.linalg.norm(delta)
        if length < 1e-12:
            # Точки совпадают — невозможно построить луч
            return None
        direction = delta / length
        
        # 4. Трассировка луча: найдём ближайшее пересечение с оправкой
        t, r_opt = self.ray_tracer.trace(surface, R_next, direction)
        if t is None:
            return None
        
        # 5. Обратное проецирование 3D-точки на параметрические координаты
        try:
            u_pred, v_pred = surface.uv_from_point(r_opt)
        except ValueError:
            return None
        
        return u_pred, v_pred