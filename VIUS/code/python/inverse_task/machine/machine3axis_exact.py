import numpy as np
from machine.kinematics_base import IMachineKinematics, MachineState

from machine.kinematics_base import IMachineKinematics, MachineState
import numpy as np

class Machine3AxisExact(IMachineKinematics): # 1. Явное наследование
    
    def __init__(self, ring_radius: float, d_offset: float):
        self.r_ring = ring_radius
        self.d_off = d_offset

    def forward(self, state: MachineState) -> dict: # 2. Логика без изменений
        theta, Z, R, phi = state.coords[0], state.coords[1], state.coords[2], state.coords[3]
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        
        X_tsn = (self.r_ring * cp - R) * ct
        Y_tsn = (self.r_ring * cp - R) * st
        Z_tsn = self.r_ring * sp + Z + self.d_off
        point_3d = np.array([X_tsn, Y_tsn, Z_tsn])
        
        tau = np.array([
            -self.r_ring * sp * ct - R * st,
            -self.r_ring * sp * st + R * ct,
             self.r_ring * cp
        ])
        n = np.array([cp * ct, cp * st, sp])
        m = np.array([-sp * ct, -sp * st, cp])
        
        return {'point': point_3d, 'tau': tau, 'n': n, 'm': m}

    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray: # 3. Логика без изменений
        res = self.forward(state)
        F = np.zeros(4)
        
        target_point = target_data['point']
        F[0] = res['point'][0] - target_point[0]
        F[1] = res['point'][1] - target_point[1]
        F[2] = res['point'][2] - target_point[2]
        
        target_tau = target_data['tau']
        F[3] = np.dot(res['tau'] - target_tau, res['n']) 
        
        return F

    # 4. МЕТОД inverse() ЗДЕСЬ ОТСУТСТВУЕТ!
    # Он автоматически унаследуется от IMachineKinematics и будет работать
    # под капотом, вызывая residuals(), который вы только что написали выше.