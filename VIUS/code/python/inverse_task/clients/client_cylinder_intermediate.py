# client_cylinder_intermediate.py
import numpy as np
import matplotlib.pyplot as plt
from geometry.cylinder import CylinderAnalytical
from core.trajectory import Trajectory
from helpers.inverse_winding_intermediate import inverse_winding_intermediate
from helpers.dae_predictor import DAEPredictor
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method import newton_corrector
from geometry.tsurfaces import FixedPointTrajectory

# 1. Поверхности
R_ext, R_int = 4.0, 3.0
outer_cyl = CylinderAnalytical(R_ext)
inner_cyl = CylinderAnalytical(R_int)

# 2. Винтовая траектория на внешнем цилиндре
def helical(R, height_start, height_end, turns, n):
    total_h = height_end - height_start
    pitch = total_h / turns
    theta = np.linspace(0, 2*np.pi*turns, n)
    z = height_start + pitch * theta / (2*np.pi)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return np.column_stack([x, y, z])

traj_pts = helical(R_ext, 1.0, 12.0, 3, 300)
traj = Trajectory.from_points(traj_pts, method='cubic')

# 3. Начальное приближение для внутреннего цилиндра (касательный луч)
R0 = traj.R(0.0)
theta = np.arccos(R_int / R_ext)  # угол касания для соосных цилиндров
u_guess = theta if R0[0] >= 0 else -theta
v_guess = R0[2]

# Коррекция начальной точки до Φ=0
dummy_traj = FixedPointTrajectory(R0)
u0, v0, Phi0, _, conv = newton_corrector(
    inner_cyl, dummy_traj, u_guess, v_guess, 0.0, eps_Phi=1e-12, max_iter=20)
print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}, сходимость={conv}")

# 4. Настройка предиктора и запуск
solver = SciPySolver(method='Radau', rtol=1e-8, atol=1e-10)
predictor = DAEPredictor(solver)

result = inverse_winding_intermediate(
    inner_cyl, traj, u0, v0,
    count_points=100,
    eps_Phi=1e-10, max_newton=7, max_bisect=4,
    predictor=predictor
)

# 5. Визуализация
line_E2 = result['points_3d']
print(f"Максимальная невязка |Φ| = {np.max(np.abs(result['Phi'])):.2e}")

# ... (графики как в fnc_2_1_1.py)