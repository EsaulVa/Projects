# client_cylinder_hybrid.py
import sys
from pathlib import Path

# Добавляем корневую директорию проекта (родительскую по отношению к папке gui)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
import numpy as np
from geometry.cylinder import CylinderAnalytical
from core.trajectory import Trajectory
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor
from solvers.scipy_solver import SciPySolver
from helpers.optical_predictor import RayTracer
from helpers.intersection import CylinderIntersection, SphereIntersection
from helpers.inverse_method import newton_corrector
from geometry.tsurfaces import FixedPointTrajectory

# 1. Поверхности
R_ext, R_int = 3.0, 2.0
outer = CylinderAnalytical(R_ext)
inner = CylinderAnalytical(R_int)

# 2. Траектория (винтовая на внешнем цилиндре)
def helical(R, h_start, h_end, turns, n):
    total_h = h_end - h_start
    pitch = total_h / turns
    theta = np.linspace(0, 2*np.pi*turns, n)
    z = h_start + pitch * theta / (2*np.pi)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return np.column_stack([x, y, z])

traj_pts = helical(R_ext, 1.0, 12.0, 3, 300)
traj = Trajectory.from_points(traj_pts, method='cubic')

# 3. Начальная точка на внутреннем цилиндре (касательный луч)
R0 = traj.R(0.0)
theta = np.arccos(R_int / R_ext)
u_guess = theta if R0[0] >= 0 else -theta
v_guess = R0[2]
dummy = FixedPointTrajectory(R0)
u0, v0, Phi0, _, conv = newton_corrector(
    inner, dummy, u_guess, v_guess, 0.0, eps_Phi=1e-12, max_iter=20)
print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}")

# 4. Настройка предикторов
# DAE-предиктор
solver_dae = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
dae_pred = DAEPredictor(solver_dae)

# Оптический предиктор
ray_tracer = RayTracer()
ray_tracer.register(CylinderAnalytical, CylinderIntersection())
# Если используем сегменты, нужно зарегистрировать и их
optical_pred = OpticalPredictor(ray_tracer)

# 5. Запуск гибридного алгоритма
result = inverse_winding_hybrid(
    inner, traj, u0, v0,
    count_points=20,
    eps_Phi=1e-10, max_newton=7, max_bisect=4,
    predictor_dae=dae_pred,
    predictor_optical=optical_pred,
    eps_kappa=1e-4,
    u_margin=0.05
)

line_E2 = result['points_3d']
print(f"Максимальная невязка |Φ| = {np.max(np.abs(result['Phi'])):.2e}")

# 6. Визуализация (аналогично предыдущим примерам)
# ...