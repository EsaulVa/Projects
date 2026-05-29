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
        r_cur = surface.position(u_cur, v_cur)
        R_next = traj.R(z_next)
        delta = r_cur - R_next
        length = np.linalg.norm(delta)
        if length < 1e-12:
            return None
        direction = delta / length
        
        # ИСПРАВЛЕНИЕ: ищем ДАЛЬНЕЕ пересечение (выход луча), а не ближайшее.
        # Ожидаемая длина нити ~ length. Пропускаем входное пересечение,
        # начиная поиск с 0.7*length.
        # t, r_opt = self.ray_tracer.trace(
        #     surface, R_next, direction,
        #     t_min=1e-6,
        #     t_max=1.3 * length
        # )
        t, r_opt = self.ray_tracer.trace(
            surface, R_next, direction,
            t_min=1e-6,          # начинаем сразу от раскладчика
            t_max=1.3 * length
        )
# Если результат слишком далеко от текущего (u_cur, v_cur) — отбросить
        if t is None:
            # Fallback: если не нашли дальнее, ищем любое
            t, r_opt = self.ray_tracer.trace(
                surface, R_next, direction,
                t_min=1e-6,
                t_max=2.0 * length
            )
        if t is None:
            return None
        
        try:
            u_pred, v_pred = surface.uv_from_point(r_opt)
        except ValueError:
            return None
        
        return u_pred, v_pred

    # def predict(self, z_k, z_next, u_cur, v_cur, surface, traj):
    #     # 1. Текущая точка укладки на оправке
    #     r_cur = surface.position(u_cur, v_cur)
        
    #     # 2. Следующая точка траектории (раскладчика)
    #     R_next = traj.R(z_next)
        
    #     # 3. Направление луча от R_next к текущей точке укладки
    #     delta = r_cur - R_next
    #     length = np.linalg.norm(delta)
    #     if length < 1e-12:
    #         # Точки совпадают — невозможно построить луч
    #         return None
    #     direction = delta / length
        
    #     # 4. Трассировка луча: найдём ближайшее пересечение с оправкой
    #     t, r_opt = self.ray_tracer.trace(surface, R_next, direction)
    #     if t is None:
    #         return None
        
    #     # 5. Обратное проецирование 3D-точки на параметрические координаты
    #     try:
    #         u_pred, v_pred = surface.uv_from_point(r_opt)
    #     except ValueError:
    #         return None
        
    #     return u_pred, v_pred