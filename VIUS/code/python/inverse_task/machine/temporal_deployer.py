# # Для точки схода нити (уже есть в forward)
# def tsn_selector(state: MachineState) -> np.ndarray:
#     return machine.forward(state)['point']

# # Для центра кольца (если forward его не возвращает – добавим)
# def ring_center_selector(state: MachineState) -> np.ndarray:
#     theta, Z, R, phi = state.coords
#     # Центр кольца лежит на оси оправки? Нет, он смещён.
#     # Упрощённо: центр кольца в локальной системе кольца – это (0,0,0) в системе кольца.
#     # Но проще добавить в forward отдельный ключ 'center'.
#     # Пока допустим, что мы модифицировали forward:
#     return machine.forward(state)['center']

# # Использование
# result = kinematics.compute_kinematics(t_eval, machine, point_selector=tsn_selector)
# temporal_deployer.py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from core.curve_factory import CurveFactory
from machine.kinematics_base import MachineState

class TemporalDeployer:
    """
    Адаптивное масштабирование скорости с использованием кривых из core.
    """
    def __init__(self, curve_factory: CurveFactory, method: str = 'cubic', **curve_kwargs):
        self.factory = curve_factory
        self.method = method
        self.curve_kwargs = curve_kwargs

    def deploy(self, s_array: np.ndarray, axes_data: dict, limits: dict,
               mode: str = 'const_speed', speed_param: float = 100.0,
               n_iter: int = 5, relax: float = 0.3):
        # Построение кривых q(s)
        curves = {}
        dq_ds = {}
        ddq_ds = {}
        for name, values in axes_data.items():
            curve = self.factory.create(s_array, values, self.method, **self.curve_kwargs)
            curves[name] = curve
            dq = np.array([curve.evaluate(s, der=1) for s in s_array])
            ddq = np.array([curve.evaluate(s, der=2) for s in s_array])
            dq_ds[name] = dq
            ddq_ds[name] = ddq

        # Начальный профиль скорости
        if mode == 'const_speed':
            V = np.full(len(s_array), speed_param)
        elif mode == 'const_omega':
            if 'theta' not in axes_data:
                raise ValueError("Для const_omega нужна ось 'theta'")
            dtheta = dq_ds['theta']
            V = speed_param / (np.abs(dtheta) + 1e-12)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        ds = np.diff(s_array)
        ds = np.insert(ds, 0, ds[0])

        # Итеративная коррекция
        for _ in range(n_iter):
            V_new = V.copy()
            for i in range(1, len(s_array)-1):
                dV_ds = (V[i] - V[i-1]) / (s_array[i] - s_array[i-1]) if i>0 else 0.0
                scale = 1.0
                for name in axes_data.keys():
                    if name not in limits:
                        continue
                    a_max = limits[name]['max_accel']
                    dq = dq_ds[name][i]
                    ddq = ddq_ds[name][i]
                    accel = ddq * V[i]**2 + dq * dV_ds * V[i]
                    if abs(accel) > a_max:
                        scale = min(scale, np.sqrt(a_max / (abs(accel) + 1e-12)))
                V_new[i] = V[i] * scale
            V = relax * V_new + (1 - relax) * V

        # Интегрирование времени
        t = np.zeros(len(s_array))
        for i in range(1, len(s_array)):
            dt = (s_array[i] - s_array[i-1]) / (0.5 * (V[i] + V[i-1]))
            t[i] = t[i-1] + dt

        # Построение кривых q(t) с помощью той же фабрики
        curves_t = {}
        for name, values in axes_data.items():
            # гарантия монотонности t
            if np.any(np.diff(t) <= 0):
                idx = np.argsort(t)
                t_sorted = t[idx]
                values_sorted = values[idx]
            else:
                t_sorted = t
                values_sorted = values
            curve_t = self.factory.create(t_sorted, values_sorted, self.method, **self.curve_kwargs)
            curves_t[name] = curve_t

        return {
            't_array': t,
            'V_s': V,
            'curves_t': curves_t,
            'total_time': t[-1]
        }