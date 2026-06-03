#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_cylinder_machine.py
==========================
Гибрид client_cylinder_hybrid.py (геометрия) + client_machine.py (кинематика станка).

1. Геометрия: концентрические цилиндры (outer R=3, inner R=2).
2. ТСН: винтовая линия на outer (как в client_cylinder_hybrid).
3. Обратная задача: восстановление ЛУ на inner через inverse_winding_hybrid.
4. Кинематика: пространственная развёртка на 3-осном станке (Machine3AxisExact_ODE).
5. Верификация + визуализация + сохранение (в той же логике, что client_machine).
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
from scipy.interpolate import CubicSpline
from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.cylinder import CylinderAnalytical
from core.trajectory import Trajectory
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor
from helpers.inverse_method import newton_corrector
from solvers.scipy_solver import SciPySolver
from helpers.optical_predictor import RayTracer
from helpers.intersection import CylinderIntersection
from geometry.tsurfaces import FixedPointTrajectory
from machine.machine3axis_exact import Machine3AxisExact_ODE, MachineState


# ======================================================================
# 1. ГЕОМЕТРИЯ (из client_cylinder_hybrid.py)
# ======================================================================
R_ext, R_int = 3.0, 2.0
outer = CylinderAnalytical(R_ext)
inner = CylinderAnalytical(R_int)

# Винтовая траектория на внешнем цилиндре
def helical(R, h_start, h_end, turns, n):
    total_h = h_end - h_start
    pitch = total_h / turns
    theta = np.linspace(0, 2 * np.pi * turns, n)
    z = h_start + pitch * theta / (2 * np.pi)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return np.column_stack([x, y, z])

traj_pts = helical(R_ext, 1.0, 12.0, 3, 300)
traj = Trajectory.from_points(traj_pts, method='cubic')

# ======================================================================
# 2. ОБРАТНАЯ ЗАДАЧА: восстановление ЛУ на inner
# ======================================================================
R0 = traj.R(0.0)
theta0 = np.arccos(R_int / R_ext)
u_guess = theta0 if R0[0] >= 0 else -theta0
v_guess = R0[2]
dummy = FixedPointTrajectory(R0)
u0, v0, Phi0, _, conv = newton_corrector(
    inner, dummy, u_guess, v_guess, 0.0, eps_Phi=1e-12, max_iter=20
)
print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}, conv={conv}")

solver_dae = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
dae_pred = DAEPredictor(solver_dae)

ray_tracer = RayTracer()
ray_tracer.register(CylinderAnalytical, CylinderIntersection())
optical_pred = OpticalPredictor(ray_tracer)

result = inverse_winding_hybrid(
    inner, traj, u0, v0,
    count_points=200,
    eps_Phi=1e-10, max_newton=20, max_bisect=6,
    predictor_dae=dae_pred,
    predictor_optical=optical_pred,
    eps_kappa=1e-4,
    u_margin=0.05
)

lu_points_local = result['points_3d']  # ЛУ на inner (локальная система)
print(f"Обратная задача: max |Φ| = {np.max(np.abs(result['Phi'])):.2e}")

# ======================================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ СТАНКА (логика client_machine.py)
# ======================================================================
# Для цилиндров z_offset = 0 (коаксиальные)
z_offset = 0.0

# ТСН = винтовая на outer (уже есть как traj)
s_vals = np.linspace(0, traj.total_length, len(lu_points_local))
ts_n_pts = np.array([traj.R(s) for s in s_vals])

# ЛУ в глобальной системе (для цилиндра совпадает с локальной)
lu_points_global = lu_points_local.copy()

# theta_array — азимут точек касания
alpha = np.arctan2(lu_points_global[:, 1], lu_points_global[:, 0])
theta_array = np.unwrap(alpha)

# Сплайны для станка (как в client_machine.py)
tsn_spline = CubicSpline(s_vals, ts_n_pts, axis=0, bc_type='natural')
mandrel_spline = CubicSpline(s_vals, lu_points_global, axis=0, bc_type='natural')
tsn_func = lambda s: tsn_spline(s)
mandrel_func = lambda s: mandrel_spline(s)
d_tsn_func = lambda s: tsn_spline(s, nu=1)
d_mandrel_func = lambda s: mandrel_spline(s, nu=1)

# ======================================================================
# 4. ОБРАТНАЯ ЗАДАЧА КИНЕМАТИКИ (логика client_machine.py)
# ======================================================================
print("\n===== Пространственная развёртка (станок) =====")

# Параметры станка (масштабированы под размер цилиндра)
# В client_machine.py: ring=50, d=100 для баллона R~250.
# Для цилиндра R=2-3 берём пропорционально: ring=0.5, d=1.0
RING_RADIUS = 0.5
D_OFFSET = 1.0

machine_ode = Machine3AxisExact_ODE(ring_radius=RING_RADIUS, d_offset=D_OFFSET)

target0 = {
    'point': tsn_func(s_vals[0]),
    'r_mandrel': mandrel_func(s_vals[0])
}
initial_guess = MachineState([
    theta_array[0],
    mandrel_pts[0, 2] if 'mandrel_pts' in dir() else lu_points_global[0, 2],
    np.linalg.norm(lu_points_global[0, :2]),
    0.0
])

# Корректная инициализация начального состояния
q0 = machine_ode.inverse_first_point(target0, initial_guess)

deploy_result = machine_ode.integrate(
    (s_vals[0], s_vals[-1]),
    q0.coords,
    tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
    s_eval=s_vals
)

coords = deploy_result['coords']
s_array = deploy_result['s_array']

# ======================================================================
# 5. ИЗВЛЕЧЕНИЕ КООРДИНАТ СТАНКА
# ======================================================================
theta_actual = coords[:, 0]
Z_actual = coords[:, 1] + z_offset
R_actual = coords[:, 2]
phi_actual_deg = np.degrees(coords[:, 3])

# ======================================================================
# 6. ВЕРИФИКАЦИЯ (прямая задача кинематики)
# ======================================================================
R_tsn_reconstructed = np.zeros_like(ts_n_pts)
for i in range(len(s_array)):
    state = MachineState(coords[i])
    R_tsn_reconstructed[i] = machine_ode.forward(state)['point']

error = np.linalg.norm(ts_n_pts - R_tsn_reconstructed, axis=1)
print(f"Средняя ошибка прямой задачи: {np.mean(error):.3e}")
print(f"Максимальная ошибка: {np.max(error):.3e}")

# ======================================================================
# 7. ВИЗУАЛИЗАЦИЯ (логика client_machine.py)
# ======================================================================
print("\n===== Построение графиков =====")

# --- 3D ---
fig3d = go.Figure()

# Внутренний цилиндр (оправка)
v_cyl = np.linspace(0, 2*np.pi, 60)
z_cyl = np.linspace(0, 13, 40)
V, Z = np.meshgrid(v_cyl, z_cyl)
X_inner = R_int * np.cos(V)
Y_inner = R_int * np.sin(V)
fig3d.add_trace(go.Surface(
    x=X_inner, y=Y_inner, z=Z,
    opacity=0.4, colorscale='Blues', showscale=False, name='Оправка (inner)'
))

# Внешний цилиндр (безопасность)
X_outer = R_ext * np.cos(V)
Y_outer = R_ext * np.sin(V)
fig3d.add_trace(go.Surface(
    x=X_outer, y=Y_outer, z=Z,
    opacity=0.2, colorscale='Reds', showscale=False, name='Безопасность (outer)'
))

# ТСН
fig3d.add_trace(go.Scatter3d(
    x=ts_n_pts[:,0], y=ts_n_pts[:,1], z=ts_n_pts[:,2],
    mode='lines', line=dict(color='red', width=3), name='ТСН (outer)'
))

# ЛУ
fig3d.add_trace(go.Scatter3d(
    x=lu_points_global[:,0], y=lu_points_global[:,1], z=lu_points_global[:,2],
    mode='lines', line=dict(color='green', width=3), name='ЛУ (inner)'
))

# Центр кольца станка
X_center = R_actual * np.cos(theta_actual)
Y_center = R_actual * np.sin(theta_actual)
fig3d.add_trace(go.Scatter3d(
    x=X_center, y=Y_center, z=Z_actual,
    mode='lines', line=dict(color='orange', width=3, dash='dot'),
    name='Центр кольца'
))

fig3d.update_layout(
    title='Развёртка кинематики: Цилиндры -> Станок',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=30)
)
fig3d.write_html("cylinder_kinematics_3d.html")

# --- 2D законы движения ---
fig_2d = make_subplots(rows=4, cols=1,
    subplot_titles=('θ(s), рад', 'Z(s), мм', 'R(s), мм', 'φ(s), град'))

fig_2d.add_trace(go.Scatter(x=s_array, y=theta_actual, mode='lines+markers',
    name='θ(s)', line=dict(color='black', width=2)), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=Z_actual, mode='lines+markers',
    name='Z(s)', line=dict(color='blue', width=2)), row=2, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=R_actual, mode='lines+markers',
    name='R(s)', line=dict(color='green', width=2)), row=3, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=phi_actual_deg, mode='lines+markers',
    name='φ(s)', line=dict(color='purple', width=2)), row=4, col=1)

fig_2d.update_xaxes(title_text="s, мм", row=4, col=1)
fig_2d.update_yaxes(title_text="рад", row=1, col=1)
fig_2d.update_yaxes(title_text="мм", row=2, col=1)
fig_2d.update_yaxes(title_text="мм", row=3, col=1)
fig_2d.update_yaxes(title_text="град", row=4, col=1)
fig_2d.update_layout(height=1000, title_text="Законы движения (цилиндр)", showlegend=True)
fig_2d.write_html("cylinder_kinematics_2d.html")

# ======================================================================
# 8. СОХРАНЕНИЕ
# ======================================================================
results = {
    's': s_array,
    'theta': theta_actual,
    'Z': Z_actual,
    'R': R_actual,
    'phi': coords[:, 3],
    'z_offset': z_offset,
    'tsn_pts': ts_n_pts,
    'mandrel_pts': lu_points_global,
    'ring_radius': RING_RADIUS,
    'd_offset': D_OFFSET,
}
scipy.io.savemat('cylinder_kinematics_results.mat', results)
print("\nРезультаты сохранены: cylinder_kinematics_results.mat")
print("Графики: cylinder_kinematics_3d.html, cylinder_kinematics_2d.html")
