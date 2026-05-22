import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io
import sys
from pathlib import Path

# Добавляем корневую директорию проекта (родительскую по отношению к папке gui)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method import newton_corrector
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor, RayTracer
from helpers.intersection import RevolutionIntersection,RobustRevolutionIntersection
from geometry.tsurfaces import FixedPointTrajectory

import sys
from pathlib import Path

# Добавляем корневую директорию проекта (родительскую по отношению к папке gui)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# ---------- Коэффициенты из surface_r.m (оправка) ----------
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392,
                 -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

E2 = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

# ---------- Коэффициенты из surface_r_b.m (безопасность) ----------
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366,
              -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387

E1 = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)

# 1. Загружаем ТСН из .mat
data = scipy.io.loadmat('winding_trajectory_result.mat')
z_offset = (bound_safe[3] - bound_opravka[3]) / 2  # (955.956 - 768.54)/2 ≈ 93.708
X, Y, Z = data['X_tsn'].flatten(), data['Y_tsn'].flatten(), data['Z_tsn'].flatten()
Z_local = Z - z_offset
points_tsn = np.column_stack([X, Y, Z_local])
count_points = len(X)

print(f"z_offset = {z_offset:.3f} мм")
traj = Trajectory.from_points(points_tsn, method='nurbs', degree=4)

# Загружаем эталонную линию укладки из LU_data.mat
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']  # массив 545x3
    print(f"Эталонная линия укладки загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    print("Файл LU_data.mat не найден – эталонная линия не будет показана.")
    r_etalon = None

# Начальная точка из эталонной линии (локальные координаты оправки)
if r_etalon is not None:
    u0 = r_etalon[0, 2]          # z-координата
    v0 = np.arctan2(r_etalon[0, 1], r_etalon[0, 0])
else:
    # fallback: первая точка траектории, обрезанная по границам оправки
    R0 = traj.R(0.0)
    u0 = np.clip(R0[2], E2.v_min, E2.v_max)
    v0 = np.arctan2(R0[1], R0[0])

# Коррекция начальной точки до касания Φ = 0
r0 = E2.position(u0, v0)
R0 = traj.R(0.0)
m0 = E2.normal(u0, v0)
Phi0 = np.dot(R0 - r0, m0)
if abs(Phi0) > 1e-8:
    print("Корректировка начальной точки...")
    u0, v0, Phi0_corr, _, conv0 = newton_corrector(
        E2, traj, u0, v0, 0.0, eps_Phi=1e-12, max_iter=2000
    )
    print(f"После коррекции: Φ₀ = {Phi0_corr:.6e}, сошёлся: {conv0}")

print("\n===== Обратная задача: восстановление линии укладки на E2 (гибрид) =====")

# Настройка предикторов
solver_dae = SciPySolver(method='BDF', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)

ray_tracer = RayTracer()
# ray_tracer.register(PiecewisePolynomialRevolution, RevolutionIntersection())
ray_tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
optical_predictor = OpticalPredictor(ray_tracer)

# Гибридный решатель
result = inverse_winding_hybrid(
    E2, traj, u0, v0,
    count_points=30000,
    eps_Phi=1e-10, max_newton=2, max_bisect=10, jump_threshold=3.0,
    predictor_dae=dae_predictor,
    predictor_optical=optical_predictor,
    eps_kappa=1e-2,
    u_margin=0.05,
    force_optical_after_fail=True
)

# Извлечение результатов
z_vals = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
lam_hist = result['lam']
flags = result['flags']
line_E2 = result['points_3d']

n_bisected = np.sum(flags == 1)
print(f"Шагов с бисекцией: {n_bisected} из {len(z_vals)-1}")
print(f"Максимальная невязка |Φ| = {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя невязка |Φ|      = {np.mean(np.abs(Phi_hist)):.2e}")
print(f"Среднее итераций Ньютона  = {np.mean(newton_iters_hist[1:]):.2f}")
print(f"Максимум итераций Ньютона = {np.max(newton_iters_hist[1:])}")
print(f"Минимальная невязка Phi: {np.min(result['Phi']):.4f}")
print(f"Максимальная невязка Phi: {np.max(result['Phi']):.4f}")
print(f"Средняя невязка Phi: {np.mean(np.abs(result['Phi'])):.4f}")

# Визуализация (базовый вариант)
fig = go.Figure()
# Сетки поверхностей
u_grid = np.linspace(0, 2*np.pi, 80)
v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 60)
X2, Y2, Z2 = np.zeros((len(v_grid_E2), len(u_grid))), np.zeros_like(X2), np.zeros_like(X2)
for i, v in enumerate(v_grid_E2):
    for j, u in enumerate(u_grid):
        p = E2.position(u, v)
        X2[i,j], Y2[i,j], Z2[i,j] = p
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds', showscale=False, name='Оправка'))

# Траектория R(z)
fig.add_trace(go.Scatter3d(x=points_tsn[:,0], y=points_tsn[:,1], z=points_tsn[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='ТСН'))

# Восстановленная линия укладки
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                           mode='lines', line=dict(color='red', width=4), name='Восстановленная ЛУ'))

# Эталонная линия
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(x=r_etalon[:,0], y=r_etalon[:,1], z=r_etalon[:,2],
                               mode='lines', line=dict(color='orange', width=2, dash='dot'),
                               name='Эталонная ЛУ'))

fig.update_layout(title='Гибридная обратная задача (полиномиальный баллон)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)
fig.write_html('fnc_ballon_hybrid.html')
print("График сохранён в fnc_ballon_hybrid.html")

# Диагностические графики
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.semilogy(z_vals[1:], np.abs(Phi_hist[1:])+1e-16)
plt.title('Невязка |Φ|')
plt.subplot(2,2,2)
plt.plot(z_vals[1:], newton_iters_hist[1:], '.')
plt.title('Итерации Ньютона')
plt.subplot(2,2,3)
plt.plot(z_vals, kappa_n_hist)
plt.title('κ_n')
plt.subplot(2,2,4)
plt.plot(z_vals, u_hist, label='u(z)')
plt.plot(z_vals, v_hist, label='v(z)')
plt.legend()
plt.tight_layout()
plt.show()