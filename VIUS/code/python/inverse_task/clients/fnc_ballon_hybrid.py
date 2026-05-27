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

# E2 = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)
# пишем:
from geometry.fixed_surfaces import FixedPiecewisePolynomialRevolution, safe_initial_point
E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)


# ---------- Коэффициенты из surface_r_b.m (безопасность) ----------
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366,
              -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387

# E1 = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)
E1 = FixedPiecewisePolynomialRevolution(phi_c_safe,   R_c_safe,   bound_safe,   cyl_r_safe)

# 1. Загружаем ТСН из .mat
data = scipy.io.loadmat('winding_trajectory_result.mat')
z_offset = (bound_safe[3] - bound_opravka[3]) / 2  # (955.956 - 768.54)/2 ≈ 93.708
X, Y, Z = data['X_tsn'].flatten(), data['Y_tsn'].flatten(), data['Z_tsn'].flatten()
Z_local = Z - z_offset
points_tsn = np.column_stack([X, Y, Z_local])
count_points = len(X)

print(f"z_offset = {z_offset:.3f} мм")
traj = Trajectory.from_points(points_tsn, method='cubic')  # cubic для стабильности
# traj = Trajectory.from_points(points_tsn, method='nurbs', degree=4)
print(f"Z начала: {points_tsn[0,2]:.3f}, Z конца: {points_tsn[-1,2]:.3f}")
print(f"R'(0) = {traj.R_deriv(0.0)}")
if points_tsn[0, 2] > points_tsn[-1, 2]:
    points_tsn = points_tsn[::-1].copy()
    print("Инвертировано: ТСН шла сверху вниз")
# Загружаем эталонную линию укладки из LU_data.mat
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']  # массив 545x3
    print(f"Эталонная линия укладки загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    print("Файл LU_data.mat не найден – эталонная линия не будет показана.")
    r_etalon = None

if r_etalon is not None:
    u0 = r_etalon[0, 2]
    v0 = np.arctan2(r_etalon[0, 1], r_etalon[0, 0])
else:
    R0 = traj.R(0.0)
    u0, v0 = safe_initial_point(E2, R0)


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

# --- Диагностика нормали и корректора на одной точке ---
z_test = 0.0
R_test = traj.R(z_test)
u_test, v_test = u0, v0

# Проверим нормаль
r_test = E2.position(u_test, v_test)
m_test = E2.normal(u_test, v_test)
Phi_test = np.dot(R_test - r_test, m_test)
print(f"Φ на старте: {Phi_test:.4f}")
print(f"Нормаль: {m_test}")
print(f"R-r (направление нити): {R_test - r_test}")

# Запустим корректор вручную
u_corr, v_corr, Phi_corr, nit, conv = newton_corrector(
    E2, traj, u_test, v_test, z_test, eps_Phi=1e-10, max_iter=20
)
print(f"Корректор: сошёлся={conv}, итераций={nit}, Φ={Phi_corr:.2e}")
print(f"Исправленная точка: u={u_corr:.3f}, v={v_corr:.3f}")

from helpers.inverse_method import compute_dr_dz

# --- Тест 1: Проверка compute_dr_dz в стартовой точке ---
z0 = 0.0
dz = traj.total_length / 10**6  # шаг, соответствующий count_points=300
du, dv = compute_dr_dz(E2, traj, u0, v0, z0)

u1_euler = u0 + du * dz
v1_euler = v0 + dv * dz

r1 = E2.position(u1_euler, v1_euler)
R1 = traj.R(dz)
m1 = E2.normal(u1_euler, v1_euler)
Phi_euler = np.dot(R1 - r1, m1)

print(f"\n=== Тест предиктора (явный Эйлер) ===")
print(f"du/dz={du:.6f}, dv/dz={dv:.6f}")
print(f"u1_euler={u1_euler:.3f}, v1_euler={v1_euler:.3f}")
print(f"Φ после шага Эйлера: {Phi_euler:.4e}")

# --- Тест 2: Корректор на предсказании ---
u1_c, v1_c, Phi_c, nit_c, conv_c = newton_corrector(
    E2, traj, u1_euler, v1_euler, dz, eps_Phi=1e-10, max_iter=50
)
print(f"Корректор: сошёлся={conv_c}, итераций={nit_c}, Φ={Phi_c:.4e}")
print(f"Исправлено: u={u1_c:.3f}, v={v1_c:.3f}")
# Настройка предикторов
solver_dae = SciPySolver(method='Radau', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)
# ray_tracer = RayTracer()
# # ray_tracer.register(PiecewisePolynomialRevolution, RevolutionIntersection())
# ray_tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
from helpers.fixed_intersections import FixedPiecewisePolynomialIntersection, FixedRobustRevolutionIntersection

ray_tracer = RayTracer()
ray_tracer.register(FixedPiecewisePolynomialRevolution, FixedRobustRevolutionIntersection())
# или:
# ray_tracer.register(PiecewisePolynomialRevolution, FixedPiecewisePolynomialIntersection())
optical_predictor = OpticalPredictor(ray_tracer)
# --- Тест 3: Предиктор DAE (интегратор) ---
pred = dae_predictor.predict(0.0, dz, u0, v0, E2, traj)
if pred:
    u1_dae, v1_dae = pred
    r1_dae = E2.position(u1_dae, v1_dae)
    m1_dae = E2.normal(u1_dae, v1_dae)
    Phi_dae_pred = np.dot(R1 - r1_dae, m1_dae)
    print(f"DAE-предиктор: u={u1_dae:.3f}, v={v1_dae:.3f}, Φ предсказания={Phi_dae_pred:.4e}")
    
    u1_dae_c, v1_dae_c, Phi_dae_c, nit_d, conv_d = newton_corrector(
        E2, traj, u1_dae, v1_dae, dz, eps_Phi=1e-10, max_iter=50
    )
    print(f"DAE+корректор: сошёлся={conv_d}, итераций={nit_d}, Φ={Phi_dae_c:.4e}")
else:
    print("DAE-предиктор вернул None!")

print("\n===== Обратная задача: восстановление линии укладки на E2 (гибрид) =====")



# # ray_tracer = RayTracer()
# # # ray_tracer.register(PiecewisePolynomialRevolution, RevolutionIntersection())
# # ray_tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
# from helpers.fixed_intersections import FixedPiecewisePolynomialIntersection, FixedRobustRevolutionIntersection

# ray_tracer = RayTracer()
# ray_tracer.register(FixedPiecewisePolynomialRevolution, FixedRobustRevolutionIntersection())
# # или:
# # ray_tracer.register(PiecewisePolynomialRevolution, FixedPiecewisePolynomialIntersection())
# optical_predictor = OpticalPredictor(ray_tracer)

# Гибридный решатель
result = inverse_winding_hybrid(
    E2, traj, u0, v0,
    count_points=300,
    eps_Phi=1e-10, max_newton=7, max_bisect=10, jump_threshold=3.0,
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
v_grid = np.linspace(0, 2*np.pi, 80)              # азимут
u_grid_E2 = np.linspace(E2.u_min, E2.u_max, 60)  # высота (аксиальная)
X2 = np.zeros((len(v_grid), len(u_grid_E2)))
Y2, Z2 = np.zeros_like(X2), np.zeros_like(X2)
for i, v in enumerate(v_grid):
    for j, u in enumerate(u_grid_E2):
        p = E2.position(u, v)
        X2[i,j], Y2[i,j], Z2[i,j] = p
# # Сетки поверхностей
# u_grid = np.linspace(0, 2*np.pi, 80)
# v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 60)
# X2, Y2, Z2 = np.zeros((len(v_grid_E2), len(u_grid))), np.zeros_like(X2), np.zeros_like(X2)
# for i, v in enumerate(v_grid_E2):
#     for j, u in enumerate(u_grid):
#         p = E2.position(u, v)
#         X2[i,j], Y2[i,j], Z2[i,j] = p
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