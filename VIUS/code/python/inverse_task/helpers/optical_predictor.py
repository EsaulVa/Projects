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
        R_next = traj.R(z_next)
        
        # 1. Попытка локальной трассировки (от R_next к r_cur)
        try:
            r_cur = surface.position(u_cur, v_cur)
            direction_local = r_cur - R_next
            if np.linalg.norm(direction_local) > 1e-6:
                direction_local /= np.linalg.norm(direction_local)
                t, r_opt = self.ray_tracer.trace(surface, R_next, direction_local, 
                                                 t_min=1e-3, t_max=2000.0)
                if t is not None:
                    u_p, v_p = surface.uv_from_point(r_opt)
                    dv = v_p - v_cur
                    while dv > np.pi: dv -= 2*np.pi
                    while dv < -np.pi: dv += 2*np.pi
                    # Строгий фильтр для локального поиска
                    if abs(dv) < 1.0 and abs(u_p - u_cur) < 50.0:
                        return u_p, v_p
        except Exception:
            pass

        # 2. ГЛОБАЛЬНЫЙ ПОИСК ПО КАСАТЕЛЬНОЙ (Вместо луча в центр!)
        # Используем производную траектории ТСН как направление "вперед"
        try:
            R_deriv = traj.R_deriv(z_k)
            # Проецируем скорость раскладчика на плоскость, перпендикулярную радиус-вектору
            # чтобы получить приблизительное направление намотки
            if np.linalg.norm(R_deriv) > 1e-6:
                dir_global = R_deriv / np.linalg.norm(R_deriv)
                
                # Трассируем луч в направлении движения раскладчика
                t, r_opt = self.ray_tracer.trace(surface, R_next, dir_global, 
                                                 t_min=1e-3, t_max=2000.0)
                
                if t is not None:
                    u_p, v_p = surface.uv_from_point(r_opt)
                    
                    # ФИЛЬТР ПРАВДОПОДОБИЯ:
                    # Новая точка не должна быть слишком далеко от предыдущей
                    # (максимальный шаг ~ dz * 2)
                    dz = z_next - z_k
                    max_step_u = dz * 3.0  # Допуск по высоте
                    
                    if abs(u_p - u_cur) < max_step_u:
                        return u_p, v_p
        except Exception:
            pass
            
        # 3. Fallback: Луч в центр оси (только если ничего не помогло)
        try:
            center_axis = np.array([0.0, 0.0, R_next[2]])
            direction_fallback = center_axis - R_next
            if np.linalg.norm(direction_fallback) > 1e-6:
                direction_fallback /= np.linalg.norm(direction_fallback)
                t, r_opt = self.ray_tracer.trace(surface, R_next, direction_fallback, 
                                                 t_min=1e-3, t_max=2000.0)
                if t is not None:
                    u_p, v_p = surface.uv_from_point(r_opt)
                    # Очень строгий фильтр для fallback
                    if abs(u_p - u_cur) < dz * 2.0:
                        return u_p, v_p
        except Exception:
            pass
            
        return None
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