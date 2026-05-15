import numpy as np
from scipy.optimize import least_squares
from machine.kinematics_base import IMachineKinematics, MachineState

class Machine3AxisExact(IMachineKinematics):
    def __init__(self, ring_radius: float, d_offset: float):
        self.r_ring = ring_radius
        self.d_off = d_offset

    def forward(self, state: MachineState) -> dict:
        theta, Z, R, phi = state.coords
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        rr = self.r_ring

        X = (rr * cp - R) * ct - rr * sp * st
        Y = (rr * cp - R) * st + rr * sp * ct
        Z_coord = rr * sp + Z + self.d_off
        point = np.array([X, Y, Z_coord])

        dX_dphi = -rr * sp * ct - rr * cp * st
        dY_dphi = -rr * sp * st + rr * cp * ct
        dZ_dphi = rr * cp
        d = np.array([dX_dphi, dY_dphi, dZ_dphi])
        d_norm = d / (np.linalg.norm(d) + 1e-12)

        n = np.array([cp*ct, cp*st, sp])
        return {'point': point, 'd': d_norm, 'n': n}

    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
        res = self.forward(state)
        R_tsn = res['point']
        d = res['d']
        r_mandrel = target_data['r_mandrel']
        F = np.zeros(4)
        # Масштабируем геометрическую ошибку (делим на 1000)
        F[0:3] = (R_tsn - target_data['point']) / 1000.0
        delta = r_mandrel - R_tsn
        length = np.linalg.norm(delta)
        if length > 1e-9:
            tau = delta / length
        else:
            tau = delta
        F[3] = np.dot(tau, d)
        return F

    def inverse(self, target_data: dict, initial_guess: MachineState) -> MachineState:
        def func(x):
            return self.residuals(target_data, MachineState(x))
        # Используем метод 'lm' – теперь уравнений 4, переменных 4
        res = least_squares(func, initial_guess.coords, method='lm',
                            max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if res.success and np.linalg.norm(res.fun) < 1e-9:
            return MachineState(res.x)
        else:
            # Резервный метод 'trf' на случай проблем
            res2 = least_squares(func, initial_guess.coords, method='trf',
                                 max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if res2.success and np.linalg.norm(res2.fun) < 1e-9:
                return MachineState(res2.x)
            else:
                raise RuntimeError(f"Inverse failed: {res.message}, fun_norm={np.linalg.norm(res.fun)}")

# machine/machine3axis_exact_ode.py
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import gmres
from machine.kinematics_base import MachineState

class Machine3AxisExact_ODE:
    def __init__(self, ring_radius: float, d_offset: float):
        self.r_ring = ring_radius
        self.d_off = d_offset

    def forward(self, state: MachineState) -> dict:
        theta, Z, R, phi = state.coords
        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        rr = self.r_ring

        X = (rr * cp - R) * ct - rr * sp * st
        Y = (rr * cp - R) * st + rr * sp * ct
        Z_coord = rr * sp + Z + self.d_off
        point = np.array([X, Y, Z_coord])

        dX_dphi = -rr * sp * ct - rr * cp * st
        dY_dphi = -rr * sp * st + rr * cp * ct
        dZ_dphi = rr * cp
        d = np.array([dX_dphi, dY_dphi, dZ_dphi])
        d_norm = d / (np.linalg.norm(d) + 1e-12)

        n = np.array([cp*ct, cp*st, sp])
        return {'point': point, 'd': d_norm, 'n': n}

    def residuals(self, target_data: dict, state: MachineState) -> np.ndarray:
        res = self.forward(state)
        R_tsn = res['point']
        d = res['d']
        r_mandrel = target_data['r_mandrel']
        F = np.zeros(4)
        # Масштабируем геометрические ошибки
        F[0:3] = (R_tsn - target_data['point']) / 1000.0
        delta = r_mandrel - R_tsn
        length = np.linalg.norm(delta)
        tau = delta / length if length > 1e-9 else delta
        F[3] = np.dot(tau, d)
        return F

    def inverse_first_point(self, target0: dict, initial_guess: MachineState) -> MachineState:
        """Находит точное решение для первой точки (итерационно)."""
        def func(x):
            return self.residuals(target0, MachineState(x))
        res = least_squares(func, initial_guess.coords, method='lm',
                            max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if res.success and np.linalg.norm(res.fun) < 1e-9:
            return MachineState(res.x)
        else:
            raise RuntimeError(f"First point inverse failed: {res.message}, norm={np.linalg.norm(res.fun)}")

    def integrate(self, s_span, q0, tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
                  s_eval=None, alpha=2, rtol=1e-10, atol=1e-12):
        """
        Интегрирование стабилизированной ОДУ: dq/ds = -J^{-1}(dF/ds + α F)
        Решение линейной системы через GMRES.
        """
        def rhs(s, q):
            state = MachineState(q)
            R_t = tsn_func(s)
            r_m = mandrel_func(s)

            F_current = self.residuals({'point': R_t, 'r_mandrel': r_m}, state)

            # Численный якобиан (4x4)
            eps = 1e-7
            J = np.zeros((4,4))
            for i in range(4):
                q_plus = q.copy(); q_plus[i] += eps
                q_minus = q.copy(); q_minus[i] -= eps
                F_plus = self.residuals({'point': R_t, 'r_mandrel': r_m}, MachineState(q_plus))
                F_minus = self.residuals({'point': R_t, 'r_mandrel': r_m}, MachineState(q_minus))
                J[:, i] = (F_plus - F_minus) / (2*eps)

            # Производная F по s (численно)
            eps_s = 1e-6
            F_s_plus = self.residuals({'point': tsn_func(s+eps_s), 'r_mandrel': mandrel_func(s+eps_s)}, state)
            F_s_minus = self.residuals({'point': tsn_func(s-eps_s), 'r_mandrel': mandrel_func(s-eps_s)}, state)
            dF_ds = (F_s_plus - F_s_minus) / (2*eps_s)

            b = -dF_ds - alpha * F_current

            # Решаем J * dq = b методом GMRES
            # Плотная матрица 4x4, но GMRES устойчив к плохой обусловленности
            dq, info = gmres(J, b, atol=1e-12, rtol=1e-12)
            if info != 0:
                # fallback на прямое решение (если GMRES не сошёлся)
                print('lol')
                dq = np.linalg.solve(J, b)

            # Ограничение величины производной (защита от выбросов)
            max_dq = 10.0
            dq = np.clip(dq, -max_dq, max_dq)
            return dq

        sol = solve_ivp(rhs, s_span, q0, method='RK45', t_eval=s_eval,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        return {'s_array': sol.t, 'coords': sol.y.T}