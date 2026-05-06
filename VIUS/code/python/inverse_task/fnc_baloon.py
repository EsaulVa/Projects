import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io
from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.scipy_solver import SciPySolver
from inverse_winding.inverse_winding_builder import InvWindingLineBuilder, InverseWindingLineBuilder
# from core.trajectory_io import TrajectoryIO
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from helpers.inverse_method import *
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
# Смещение центров (как в align_surface_centers.m)
z_offset = (bound_safe[3] - bound_opravka[3]) / 2  # (955.956 - 768.54)/2 ≈ 93.708
X, Y, Z = data['X_tsn'].flatten(), data['Y_tsn'].flatten(), data['Z_tsn'].flatten()
Z_local = Z - z_offset
points_tsn_ = np.column_stack([X, Y, Z])
# traj = Trajectory.from_points(points_tsn_local, method='cubic')
points_tsn = np.column_stack([X, Y, Z_local])
count_points=len(X)

print(f"z_offset = {z_offset:.3f} мм")
traj = Trajectory.from_points(points_tsn, method='cubic')
import scipy.io as sio

# Загружаем эталонную линию укладки из l.mat
try:
    data_l = sio.loadmat('LU_data.mat')
    r_etalon = data_l['r']  # массив 545x3
    print(f"Эталонная линия укладки загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    print("Файл l.mat не найден – эталонная линия не будет показана.")
    r_etalon = None

# # Начальное приближение: высота из первой точки ТСН (обрезанная по границам оправки) и азимут
# u0_guess = np.clip(Z_local[0], E2.u_min, E2.u_max)
# v0_guess = np.arctan2(Y[0], X[0])

# # Коррекция начальной точки методом Ньютона
# r0 = E2.position(u0_guess, v0_guess)
# R0 = traj.R(0.0)
# m0 = E2.normal(u0_guess, v0_guess)
# Phi0 = np.dot(R0 - r0, m0)
# if abs(Phi0) > 1e-8:
#     print("Корректировка начальной точки...")
#     u0, v0, Phi0_corr, _, conv0 = newton_corrector(
#         E2, traj, u0_guess, v0_guess, 0.0, eps_Phi=1e-12, max_iter=20
#     )
#     print(f"После коррекции: Φ₀ = {Phi0_corr:.6e}, сошёлся: {conv0}")
# else:
#     u0, v0 = u0_guess, v0_guess

# Далее используем (u0, v0) в inverse_winding_v3
u0=0
v0=0
# Коррекция начальной точки
r0 = E2.position(u0, v0)
R0 = traj.R(0.0)
m0 = E2.normal(u0, v0)
Phi0 = np.dot(R0 - r0, m0)
# if abs(Phi0) > 1e-8:
#     print("Корректировка начальной точки...")
#     u0, v0, Phi0_corr, _, conv0 = newton_corrector(
#         E2, traj, u0, v0, 0.0, eps_Phi=1e-12, max_iter=20
#     )
#     print(f"После коррекции: Φ₀ = {Phi0_corr:.6e}, сошёлся: {conv0}")


print("\n===== Обратная задача: восстановление линии укладки на E2 =====")

# ======================================================================
# 8. ЗАПУСК ОБРАТНОЙ ЗАДАЧИ
# ======================================================================

count_points = 30

print(f"\n===== Обратная задача v3 ({count_points} точек) =====")
u0_mandrel=u0
v0_mandrel=v0
line_E1=points_tsn
result = inverse_winding_v3(
    E2, traj, u0_mandrel, v0_mandrel,
    count_points=count_points,
    eps_Phi=1e-10,
    max_newton=7,
    max_bisect=4,
    jump_threshold=3.0
)
pts_geod=line_E1
z_eval = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
lam_hist = result['lam']
flags = result['flags']
line_E2 = result['points_3d']

n_bisected = np.sum(flags == 1)
print(f"\nШагов с бисекцией: {n_bisected} из {count_points - 1}")
print(f"Максимальная невязка |Φ|: {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя невязка |Φ|:      {np.mean(np.abs(Phi_hist)):.2e}")
print(f"Среднее итераций Ньютона:  {np.mean(newton_iters_hist[1:]):.2f}")
print(f"Максимум итераций Ньютона: {np.max(newton_iters_hist[1:])}")
# ----------------------------------------------------------------------
# 5. Визуализация
# ----------------------------------------------------------------------
print("\n===== Построение 3D-графика =====")

# def surface_grid(surf, u_arr, v_arr):
#     U, V = np.meshgrid(u_arr, v_arr)
#     X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
#     for i in range(U.shape[0]):
#         for j in range(U.shape[1]):
#             p = surf.position(U[i,j], V[i,j])
#             X[i,j], Y[i,j], Z[i,j] = np.array(p)
#     return X, Y, Z

# u_grid = np.linspace(0, 2*np.pi, 80)
# v_grid_E1 = np.linspace(-np.pi/2, np.pi/2, 50)
# v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 80)

# X1, Y1, Z1 = surface_grid(E1, u_grid, v_grid_E1)
# X2, Y2, Z2 = surface_grid(E2, u_grid, v_grid_E2)

fig = go.Figure()
# Построение сеток
u_opr = np.linspace(0, 768.54, 80)
v_opr = np.linspace(0, 2*np.pi, 60)
Uo, Vo = np.meshgrid(u_opr, v_opr)
Xo, Yo, Zo = np.zeros_like(Uo), np.zeros_like(Uo), np.zeros_like(Uo)
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = E2.position(Uo[i,j], Vo[i,j])
        Xo[i,j] = p[0]
        Yo[i,j] = p[1]
        Zo[i,j] = p[2] + z_offset   # переход в глобальную систему

u_safe = np.linspace(0, 955.956, 100)
v_safe = np.linspace(0, 2*np.pi, 60)
Us, Vs = np.meshgrid(u_safe, v_safe)
Xs, Ys, Zs = np.zeros_like(Us), np.zeros_like(Us), np.zeros_like(Us)
for i in range(Us.shape[0]):
    for j in range(Us.shape[1]):
        p = E1.position(Us[i,j], Vs[i,j])
        Xs[i,j] = p[0]
        Ys[i,j] = p[1]
        Zs[i,j] = p[2]

fig = go.Figure()
fig.add_trace(go.Surface(x=Xo, y=Yo, z=Zo, opacity=0.4, colorscale='Blues', name='Оправка'))
fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds', name='Безопасность'))
# fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
# fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='Баллон E2'))

# fig.add_trace(go.Scatter3d(x=line_E1[:,0], y=line_E1[:,1], z=line_E1[:,2],
#                            mode='lines', line=dict(color='blue', width=4), name='Траектория R(z)'))
# fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
#                            mode='lines', line=dict(color='red', width=4), name='Линия укладки на E2'))
from scipy.interpolate import CubicSpline
# line_E1 – массив точек (N,3)
# создаём сплайн по накопленной длине дуги вдоль линии
line_E1=points_tsn.copy()
dist = np.zeros(len(line_E1))
dist[1:] = np.linalg.norm(np.diff(line_E1, axis=0), axis=1).cumsum()

cs_x = CubicSpline(dist, line_E1[:,0])
cs_y = CubicSpline(dist, line_E1[:,1])
cs_z = CubicSpline(dist, line_E1[:,2])

# генерируем в 5-10 раз больше точек для плавной картинки
dense_dist = np.linspace(dist[0], dist[-1], len(line_E1)*10)
smooth_x = cs_x(dense_dist)
smooth_y = cs_y(dense_dist)
smooth_z = cs_z(dense_dist)

# теперь рисуем сглаженную версию
fig.add_trace(go.Scatter3d(
    x=smooth_x, y=smooth_y, z=smooth_z,
   mode='lines', line=dict(color='blue', width=4), name='Траектория R(z)'))
if r_etalon is not None:
    if r_etalon is not None:
        r_etalon_global = r_etalon.copy()
        r_etalon_global[:, 2] += z_offset
    fig.add_trace(go.Scatter3d(
        x=r_etalon[:, 0], y=r_etalon_global[:, 1], z=r_etalon_global[:, 2],
        mode='lines',
        line=dict(color='green', width=4, dash='solid'),
        name='Эталонная ЛУ'
    ))

def surface_grid(surf, u_arr, v_arr):
    U, V = np.meshgrid(u_arr, v_arr)
    X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surf.position(U[i,j], V[i,j])
            X[i,j], Y[i,j], Z[i,j] = np.array(p)
    return X, Y, Z

# u_grid = np.linspace(0, 2*np.pi, 80)
# v_grid_E1 = np.linspace(E1.v_min, E1.v_max, 50)
# v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 80)

# X1, Y1, Z1 = surface_grid(E1, u_grid, v_grid_E1)
# X2, Y2, Z2 = surface_grid(E2, u_grid, v_grid_E2)

# fig = go.Figure()
# fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='Безопасность (E1)'))
# fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='Оправка (E2)'))
# line_E2 – массив точек (N,3)
# # создаём сплайн по накопленной длине дуги вдоль линии
# dist = np.zeros(len(line_E2))
# dist[1:] = np.linalg.norm(np.diff(line_E2, axis=0), axis=1).cumsum()

# cs_x = CubicSpline(dist, line_E2[:,0])
# cs_y = CubicSpline(dist, line_E2[:,1])
# cs_z = CubicSpline(dist, line_E2[:,2])

# # генерируем в 5-10 раз больше точек для плавной картинки
# dense_dist = np.linspace(dist[0], dist[-1], len(line_E2)*10)
# smooth_x = cs_x(dense_dist)
# smooth_y = cs_y(dense_dist)
# smooth_z = cs_z(dense_dist)

# # теперь рисуем сглаженную версию
# fig.add_trace(go.Scatter3d(
#     x=smooth_x, y=smooth_y, z=smooth_z,
#     mode='lines',
#     line=dict(color='red', width=4),
#     name='Линия укладки на E2'
# ))

# Начальные и конечные точки
# E1 (траектория)
start_E1 = line_E1[0]
end_E1   = line_E1[-1]
fig.add_trace(go.Scatter3d(x=[start_E1[0]], y=[start_E1[1]], z=[start_E1[2]],
                           mode='markers', marker=dict(color='green', size=6, symbol='diamond'),
                           name='Начало R(z)'))
fig.add_trace(go.Scatter3d(x=[end_E1[0]], y=[end_E1[1]], z=[end_E1[2]],
                           mode='markers', marker=dict(color='black', size=6, symbol='diamond'),
                           name='Конец R(z)'))

# # E2 (линия укладки)
# start_E2 = line_E2[0]
# end_E2   = line_E2[-1]
# fig.add_trace(go.Scatter3d(x=[start_E2[0]], y=[start_E2[1]], z=[start_E2[2]],
#                            mode='markers', marker=dict(color='lime', size=6, symbol='diamond'),
#                            name='Старт укладки'))
# fig.add_trace(go.Scatter3d(x=[end_E2[0]], y=[end_E2[1]], z=[end_E2[2]],
#                            mode='markers', marker=dict(color='orange', size=6, symbol='diamond'),
#                            name='Финиш укладки'))

# step = 5
# for i in range(0, len(z_vals), step):
#     fig.add_trace(go.Scatter3d(x=[line_E1[i,0], line_E2[i,0]],
#                                y=[line_E1[i,1], line_E2[i,1]],
#                                z=[line_E1[i,2], line_E2[i,2]],
#                                mode='lines', line=dict(color='green', width=2), showlegend=False))

# fig.update_layout(title='Эллипсоид → баллон (прямая + обратная задача)',
#                   scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
#                   width=1000, height=800)
# fig.show()
fig.write_html('winding_analytic.html')