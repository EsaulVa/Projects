# kinematic_model.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.sparse.linalg import gmres
from machine.kinematics_base import IMachineKinematics, MachineState

class KinematicModel:
    """
    Обёртка над IMachineKinematics, добавляющая методы для работы
    с фиксированными осями (совместное редактирование).
    """
    def __init__(self, machine: IMachineKinematics):
        self.machine = machine

    def inverse_fixed(self, target_data: dict, initial_guess: MachineState,
                      fixed_indices: list, fixed_values: np.ndarray) -> MachineState:
        """
        Решает обратную задачу, фиксируя координаты с индексами fixed_indices.
        fixed_values – массив значений для этих координат (той же длины, что и fixed_indices).
        """
        n = self.machine.residuals(target_data, initial_guess).shape[0]  # число уравнений
        n_free = initial_guess.size - len(fixed_indices)
        # Преобразуем задачу: удаляем фиксированные переменные
        def func(x_free):
            q = initial_guess.coords.copy()
            q[fixed_indices] = fixed_values
            q[free_indices] = x_free
            return self.machine.residuals(target_data, MachineState(q))
        free_indices = [i for i in range(initial_guess.size) if i not in fixed_indices]
        x0 = initial_guess.coords[free_indices]
        res = least_squares(func, x0, method='lm', max_nfev=5000,
                            ftol=1e-14, xtol=1e-14, gtol=1e-14)
        if res.success and np.linalg.norm(res.fun) < 1e-9:
            q_new = initial_guess.coords.copy()
            q_new[free_indices] = res.x
            q_new[fixed_indices] = fixed_values
            return MachineState(q_new)
        else:
            raise RuntimeError(f"Inverse fixed failed: {res.message}")

    def integrate_fixed_step(self, s_span, q0_free, fixed_funcs, fixed_indices,
                         tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
                         step=1.0, s_eval=None, alpha=2.0):
        """
        Интегрирование с постоянным шагом (RK4).
        step – шаг по s (мм).
        s_eval – опциональная сетка для вывода (результаты интерполируются на неё).
        """
        s0, s_end = s_span
        n_steps = max(1, int(np.ceil((s_end - s0) / step)))
        step = (s_end - s0) / n_steps  # корректируем, чтобы попасть точно в конец

        n_total = q0_free.size + len(fixed_indices)
        free_indices = [i for i in range(n_total) if i not in fixed_indices]

        def rhs(s, q_free):
            # полный вектор q
            q_full = np.zeros(n_total)
            for idx, f in zip(fixed_indices, fixed_funcs):
                q_full[idx] = f(s)
            q_full[free_indices] = q_free
            # производные фиксированных
            dq_fixed_ds = np.zeros(len(fixed_indices))
            for j, f in enumerate(fixed_funcs):
                eps_s = 1e-6
                dq_fixed_ds[j] = (f(s+eps_s) - f(s-eps_s)) / (2*eps_s)
            R_t = tsn_func(s)
            r_m = mandrel_func(s)
            dR_t = d_tsn_func(s)
            dr_m = d_mandrel_func(s)
            J, dF_ds = self.machine.get_ode_data(s, q_full, R_t, r_m, dR_t, dr_m)
            J_free = J[:, free_indices]
            b = -dF_ds - J[:, fixed_indices] @ dq_fixed_ds - alpha * self.machine.residuals(
                {'point': R_t, 'r_mandrel': r_m}, MachineState(q_full))
            # решение переопределённой системы
            dq_free, _, _, _ = np.linalg.lstsq(J_free, b, rcond=None)
            return dq_free

        # Интегрирование RK4
        s_curr = s0
        q_curr = q0_free.copy()
        s_vals = [s_curr]
        q_vals = [q_curr.copy()]
        while s_curr < s_end - 1e-12:
            h = min(step, s_end - s_curr)
            k1 = rhs(s_curr, q_curr)
            k2 = rhs(s_curr + h/2, q_curr + h/2 * k1)
            k3 = rhs(s_curr + h/2, q_curr + h/2 * k2)
            k4 = rhs(s_curr + h, q_curr + h * k3)
            q_next = q_curr + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            s_curr += h
            s_vals.append(s_curr)
            q_vals.append(q_next.copy())
            q_curr = q_next

        s_array = np.array(s_vals)
        q_free_array = np.array(q_vals)

        # Восстанавливаем полные координаты
        full_coords = []
        for i, s in enumerate(s_array):
            q_full = np.zeros(n_total)
            for idx, f in zip(fixed_indices, fixed_funcs):
                q_full[idx] = f(s)
            q_full[free_indices] = q_free_array[i]
            full_coords.append(q_full)
        full_coords = np.array(full_coords)

        # Интерполяция на заданную сетку s_eval (если нужна)
        if s_eval is not None:
            from scipy.interpolate import interp1d
            interp_coords = []
            for j in range(n_total):
                interp_func = interp1d(s_array, full_coords[:, j], kind='linear',
                                    fill_value='extrapolate')
                interp_coords.append(interp_func(s_eval))
            full_coords = np.array(interp_coords).T
            s_array = s_eval

        return {'s_array': s_array, 'coords': full_coords}
    def integrate_fixed(self, s_span, q0_free, fixed_funcs, fixed_indices,
                    tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
                    s_eval=None, alpha=0, rtol=1e-6, atol=1e-8):
        """
        Интегрирует редуцированную ОДУ для свободных координат.
        fixed_funcs: список функций fi(s), возвращающих значение фиксированной координаты.
        fixed_indices: список индексов фиксированных осей.
        q0_free – начальные значения свободных координат.
        """
        n_total = q0_free.size + len(fixed_indices)
        free_indices = [i for i in range(n_total) if i not in fixed_indices]

        def rhs(s, q_free):
            # Собираем полный вектор q
            q_full = np.zeros(n_total)
            for idx, f in zip(fixed_indices, fixed_funcs):
                q_full[idx] = f(s)
            q_full[free_indices] = q_free
            # Вычисляем dq_fixed/ds
            dq_fixed_ds = np.zeros(len(fixed_indices))
            for j, f in enumerate(fixed_funcs):
                eps_s = 1e-6
                dq_fixed_ds[j] = (f(s+eps_s) - f(s-eps_s)) / (2*eps_s)
            # Получаем J и dF_ds от станка
            R_t = tsn_func(s)
            r_m = mandrel_func(s)
            dR_t = d_tsn_func(s)
            dr_m = d_mandrel_func(s)
            import time
            t0 = time.perf_counter()
            J, dF_ds = self.machine.get_ode_data(s, q_full, R_t, r_m, dR_t, dr_m)
            t1 = time.perf_counter()
            print(f"get_ode_data time: {t1-t0:.3e} s")
            # Подматрица для свободных осей (4 × n_free)
            J_free = J[:, free_indices]
            # Правая часть с учётом фиксированных
            b = -dF_ds - J[:, fixed_indices] @ dq_fixed_ds - alpha * self.machine.residuals(
                {'point': R_t, 'r_mandrel': r_m}, MachineState(q_full))
            # Решаем переопределённую систему J_free * dq_free = b (метод наименьших квадратов)
            dq_free, _, _, _ = np.linalg.lstsq(J_free, b, rcond=None)
            # Альтернативный вариант с lsmr (для разреженных)
            # from scipy.sparse.linalg import lsmr
            # dq_free = lsmr(J_free, b)[0]
            return dq_free

        sol = solve_ivp(rhs, s_span, q0_free, method='RK23', t_eval=s_eval,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Integration fixed failed: {sol.message}")
        # Восстанавливаем полные координаты на выходе
        full_coords = []
        for i, t in enumerate(sol.t):
            q_full = np.zeros(n_total)
            for idx, f in zip(fixed_indices, fixed_funcs):
                q_full[idx] = f(t)
            q_full[free_indices] = sol.y[:, i]
            full_coords.append(q_full)
        return {'s_array': sol.t, 'coords': np.array(full_coords)}