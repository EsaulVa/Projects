# machine/machine5axis_exact_ode.py
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from machine.kinematics_base import IMachineKinematics, MachineState

class Machine5AxisExact_ODE(IMachineKinematics):
    def __init__(self, params: dict):
        self.params = params
        # Задаём параметры со значениями по умолчанию (из ваших оценок)
        self.a2 = params.get('a2', 250.0)
        self.S4 = params.get('S4', 150.0)
        self.S6 = params.get('S6', 120.0)
        self.S7 = params.get('S7', 0.0)
        self.S8 = params.get('S8', 80.0)
        self.S9 = params.get('S9', 80.0)
        self.S10 = params.get('S10', 60.0)
        self.S11 = params.get('S11', 60.0)
        self.a12 = params.get('a12', 0.0)

    def _dh(self, theta, d, a, alpha):
        """Матрица Денавита-Хартенберга (4x4)."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,       sa,     ca,    d],
            [0,        0,      0,    1]
        ])

    def forward(self, state: MachineState) -> dict:
        """
        Обобщённые координаты (управляемые):
            x1 = S1   – продольное смещение каретки
            x2 = S3   – поперечное смещение раскладчика
            x3 = Θ5   – поворот головки (в диссертации Θ5)
            x4 = Θ6   – поворот выходного ролика
            x5 = Θ13  – поворот оправки (не используется для положения ролика)
        """
        x1, x2, x3, x4, x5 = state.coords

        # Матрицы звеньев по таблице 2.1 (звенья 0..11)
        # A1: 0→1, поступательная, theta=π/2, d=S1, a=0, alpha=0
        A1 = self._dh(np.pi/2, x1, 0.0, 0.0)
        # A2: 1→2, вращательная, theta=0, d=0, a=a2, alpha=-π/2
        A2 = self._dh(0.0, 0.0, self.a2, -np.pi/2)
        # A3: 2→3, поступательная, theta=0, d=S3, a=0, alpha=0
        A3 = self._dh(0.0, x2, 0.0, 0.0)
        # A4: 3→4, поступательная, theta=π/2, d=S4, a=0, alpha=π/2
        A4 = self._dh(np.pi/2, self.S4, 0.0, np.pi/2)
        # A5: 4→5, вращательная, theta=Θ5, d=0, a=0, alpha=-π/2
        A5 = self._dh(x3, 0.0, 0.0, -np.pi/2)
        # A6: 5→6, вращательная, theta=Θ6, d=S6, a=0, alpha=0
        A6 = self._dh(x4, self.S6, 0.0, 0.0)
        # A7: 6→7, поступательная, theta=π/2, d=S7, a=0, alpha=-π/2
        A7 = self._dh(np.pi/2, self.S7, 0.0, -np.pi/2)
        # A8: 7→8, вращательная, theta=0 (или Θ8), d=S8, a=0, alpha=π/2
        A8 = self._dh(0.0, self.S8, 0.0, np.pi/2)
        # A9: 7→9, вращательная, theta=0, d=S9, a=0, alpha=-π/2
        A9 = self._dh(0.0, self.S9, 0.0, -np.pi/2)
        # A10: 8→10, поступательная, d=S10
        A10 = self._dh(0.0, self.S10, 0.0, 0.0)
        # A11: 9→11, поступательная, d=S11
        A11 = self._dh(0.0, self.S11, 0.0, 0.0)

        # Полные преобразования для двух крайних точек ролика
        T1 = A1 @ A2 @ A3 @ A4 @ A5 @ A6 @ A7 @ A8 @ A10
        T2 = A1 @ A2 @ A3 @ A4 @ A5 @ A6 @ A7 @ A9 @ A11

        p1 = T1 @ np.array([0,0,0,1])
        p2 = T2 @ np.array([0,0,0,1])
        p_center = (p1[:3] + p2[:3]) / 2
        axis = (p2[:3] - p1[:3])
        axis_norm = axis / (np.linalg.norm(axis) + 1e-12)

        return {'point': p_center, 'axis': axis_norm}

    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
        res = self.forward(state)
        F = np.zeros(5)
        F[0:3] = res['point'] - target_data['point']
        tau = target_data.get('tau', np.zeros(3))
        F[3] = np.dot(res['axis'], tau)
        m = target_data.get('m', np.zeros(3))
        F[4] = np.dot(res['axis'], m)
        return F

    def get_ode_data(self, s, q, target_point, tau, m, d_target_ds, d_tau_ds, d_m_ds):
        eps = 1e-7
        J = np.zeros((5,5))
        for i in range(5):
            q_plus = q.copy(); q_plus[i] += eps
            q_minus = q.copy(); q_minus[i] -= eps
            F_plus = self.residuals({'point': target_point, 'tau': tau, 'm': m}, MachineState(q_plus))
            F_minus = self.residuals({'point': target_point, 'tau': tau, 'm': m}, MachineState(q_minus))
            J[:, i] = (F_plus - F_minus) / (2*eps)
        eps_s = 1e-6
        F_s_plus = self.residuals({'point': target_point + d_target_ds*eps_s,
                                   'tau': tau + d_tau_ds*eps_s,
                                   'm': m + d_m_ds*eps_s}, MachineState(q))
        F_s_minus = self.residuals({'point': target_point - d_target_ds*eps_s,
                                    'tau': tau - d_tau_ds*eps_s,
                                    'm': m - d_m_ds*eps_s}, MachineState(q))
        dF_ds = (F_s_plus - F_s_minus) / (2*eps_s)
        return J, dF_ds

    def inverse_first_point(self, target0: dict, initial_guess: MachineState) -> MachineState:
        # Этап 1: только позиция (3 уравнения, 5 неизвестных)
        def func_pos(x):
            return self.forward(MachineState(x))['point'] - target0['point']
        res_pos = least_squares(func_pos, initial_guess.coords, method='trf',
                                max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if not (res_pos.success and np.linalg.norm(res_pos.fun) < 1e-9):
            raise RuntimeError(f"Position only failed: {res_pos.message}, norm={np.linalg.norm(res_pos.fun)}")
        q_pos = res_pos.x
        # Этап 2: полная система (5 уравнений)
        def func_full(x):
            return self.residuals(target0, MachineState(x))
        res_full = least_squares(func_full, q_pos, method='lm',
                                 max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if res_full.success and np.linalg.norm(res_full.fun) < 1e-9:
            return MachineState(res_full.x)
        else:
            raise RuntimeError(f"Full inverse failed: {res_full.message}, norm={np.linalg.norm(res_full.fun)}")

    def integrate(self, s_span, q0, tsn_func, tau_func, m_func,
                  d_tsn_func, d_tau_func, d_m_func,
                  s_eval=None, alpha=2.0, rtol=1e-10, atol=1e-12):
        def rhs(s, q):
            R_t = tsn_func(s)
            tau = tau_func(s)
            m = m_func(s)
            dR_t = d_tsn_func(s)
            d_tau = d_tau_func(s)
            d_m = d_m_func(s)
            J, dF_ds = self.get_ode_data(s, q, R_t, tau, m, dR_t, d_tau, d_m)
            F_curr = self.residuals({'point': R_t, 'tau': tau, 'm': m}, MachineState(q))
            b = -dF_ds - alpha * F_curr
            dq = np.linalg.lstsq(J, b, rcond=None)[0]
            return dq
        sol = solve_ivp(rhs, s_span, q0, method='RK45', t_eval=s_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return {'s_array': sol.t, 'coords': sol.y.T}
