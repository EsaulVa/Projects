import numpy as np
from machine.kinematics_base import MachineState

class TrajectoryDeployer:
    def __init__(self, machine):
        self.machine = machine

    def deploy(self, tsn_trajectory, theta_array, lu_points_on_mandrel):
        N = len(theta_array)
        history = np.zeros((N, 4))
        success = np.ones(N, dtype=bool)

        # Первая точка
        r0 = lu_points_on_mandrel[0]
        init_coords = np.array([theta_array[0], r0[2], np.linalg.norm(r0[:2]), 0.0])
        target0 = {'point': tsn_trajectory.R(0.0), 'r_mandrel': r0}
        try:
            state0 = self.machine.inverse(target0, MachineState(init_coords))
            history[0] = state0.coords
        except Exception as e:
            print(f"Initial point failed: {e}")
            success[0] = False
            history[0] = init_coords

        # Последующие точки
        for i in range(1, N):
            s_val = tsn_trajectory.total_length * (i / (N - 1))
            target_data = {
                'point': tsn_trajectory.R(s_val),
                'r_mandrel': lu_points_on_mandrel[i]
            }
            guess = MachineState(history[i-1])
            try:
                state_i = self.machine.inverse(target_data, guess)
                history[i] = state_i.coords
                # Контроль точности прямой задачи
                calc_point = self.machine.forward(state_i)['point']
                err = np.linalg.norm(calc_point - target_data['point'])
                if err > 1e-3:
                    print(f"Step {i}: forward error = {err:.2e} mm")
            except Exception as e:
                print(f"Step {i} failed: {e}")
                success[i] = False
                history[i] = history[i-1]

        return {
            's_array': np.linspace(0, tsn_trajectory.total_length, N),
            'coords': history,
            'success': success
        }