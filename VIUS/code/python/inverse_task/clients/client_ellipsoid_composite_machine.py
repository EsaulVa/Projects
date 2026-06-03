#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_ellipsoid_composite_machine.py
=====================================
Интеграция fnc_2_1_2.py (эллипсоид + композитная оправка) 
с пространственной развёрткой станка (логика client_machine.py).

1. Прямая задача: геодезическая на эллипсоиде E1.
2. Обратная задача: восстановление ЛУ на композитной E2 (сферы+цилиндр).
3. Кинематика: Machine3AxisExact_ODE — развёртка законов движения.
4. Верификация + визуализация + сохранение.
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

from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from machine.machine3axis_exact import Machine3AxisExact_ODE, MachineState


# ======================================================================
# 1. ГЕОМЕТРИЯ (из fnc_2_1_2.py)
# ======================================================================
a1, b1, c1 = 2.0, 2.5, 5.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)

R_cyl = 1.0
L_cyl = 6.0
z_cyl_min = -L_cyl / 2
z_cyl_max = L_cyl / 2

E2 = CompositeSurface([
    SphereSegment(R_cyl, z_cyl_min, is_upper=False),
    CylinderSegment(R_cyl, z_cyl_min, z_cyl_max),
    SphereSegment(R_cyl, z_cyl_max, is_upper=True)
])

print("===== 1. Прямая задача (геодезическая на E1) =====")
deviation_law = ConstantDeviation(tan_theta=0.0)
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
forward_builder = ForwardWindingBuilder(
    surface=E1, deviation_law=deviation_law,
    solver=solver_forward, normalize_tangent=True, eps=1e-12
)

u0, v0 = 70.0 * np.pi / 180.0, -np.pi / 6.0
alpha = np.pi / 6.0
s_end = 30.0
count_points = 200
s_eval = np.linspace(0.0, s_end, count_points)

s_vals_fwd, line_E1 = forward_builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
    eval_points=s_eval
)
if not forward_builder.last_run_successful:
    raise RuntimeError("Прямая задача завершилась с ошибкой")

print(f"Построено {len(s_vals_fwd)} точек на E1, длина s = {s_vals_fwd[-1]:.3f}")

# Траектория раскладчика
traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории (сплайн): {traj.total_length:.4f}")


# ======================================================================
# 2. ОБРАТНАЯ ЗАДАЧА (восстановление ЛУ на E2)
# ======================================================================
print("\n===== 2. Обратная задача (ЛУ на композите E2) =====")

# Начальная точка на E2
u0_m = u0
v0_m = v0
r0 = E2.position(u0_m, v0_m)
R0 = traj.R(0.0)
m0 = E2.normal(u0_m, v0_m)
Phi0 = np.dot(R0 - r0, m0)
print(f"Начальная невязка Φ₀ = {Phi0:.6e}")

# Корректировка (если нужна) — используем newton_corrector из fnc_2_1_2
from helpers.inverse_method import newton_corrector

if abs(Phi0) > 1e-8:
    u0_m, v0_m, Phi0_c, _, conv0 = newton_corrector(
        E2, traj, u0_m, v0_m, 0.0, eps_Phi=1e-12, max_iter=20
    )
    print(f"После коррекции: Φ₀ = {Phi0_c:.6e}, conv = {conv0}")

# Запускаем обратную задачу через inverse_winding_hybrid (ядро проекта)
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor
from helpers.optical_predictor import RayTracer
from helpers.intersection import SphereIntersection, CylinderIntersection

solver_dae = SciPySolver(method='DBF', rtol=1e-8, atol=1e-10)
dae_pred = DAEPredictor(solver_dae)

ray_tracer = RayTracer()
ray_tracer.register(SphereSegment, SphereIntersection())
ray_tracer.register(CylinderSegment, CylinderIntersection())
optical_pred = OpticalPredictor(ray_tracer)

result_inv = inverse_winding_hybrid(
    E2, traj, u0_m, v0_m,
    count_points=300,
    eps_Phi=1e-10, max_newton=20, max_bisect=6,
    predictor_dae=dae_pred,
    predictor_optical=optical_pred,
    eps_kappa=1e-2,
    u_margin=0.1
)

line_E2 = result_inv['points_3d']
print(f"Обратная задача: max |Φ| = {np.max(np.abs(result_inv['Phi'])):.2e}")
print(f"  Средняя |Φ| = {np.mean(np.abs(result_inv['Phi'])):.2e}")


# ======================================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ СТАНКА (логика client_machine.py)
# ======================================================================
print("\n===== 3. Подготовка данных для станка =====")

z_offset = 0.0  # коаксиальные

# Точки ЛУ и ТСН как функции от s
n_pts = len(line_E2)
s_array = np.linspace(0, traj.total_length, n_pts)

# Сплайны для станка
tsn_pts = np.array([traj.R(s) for s in s_array])
mandrel_pts = line_E2.copy()  # уже в глобальной системе (z_offset = 0)

tsn_spline = CubicSpline(s_array, tsn_pts, axis=0, bc_type='natural')
mandrel_spline = CubicSpline(s_array, mandrel_pts, axis=0, bc_type='natural')
tsn_func = lambda s: tsn_spline(s)
mandrel_func = lambda s: mandrel_spline(s)
d_tsn_func = lambda s: tsn_spline(s, nu=1)
d_mandrel_func = lambda s: mandrel_spline(s, nu=1)

# theta_array — азимут точек касания
alpha_az = np.arctan2(mandrel_pts[:, 1], mandrel_pts[:, 0])
theta_array = np.unwrap(alpha_az)


# ======================================================================
# 4. ОБРАТНАЯ ЗАДАЧА КИНЕМАТИКИ
# ======================================================================
print("\n===== 4. Пространственная развёртка (станок) =====")

# Параметры станка (масштабированы под размеры эллипсоида)
# Для баллона R~250 использовались ring=50, d=100.
# Здесь R_cyl=1, a=2, b=2.5 — берём пропорционально.
RING_RADIUS = 0.3
D_OFFSET = 0.6

machine_ode = Machine3AxisExact_ODE(ring_radius=RING_RADIUS, d_offset=D_OFFSET)

target0 = {
    'point': tsn_func(s_array[0]),
    'r_mandrel': mandrel_func(s_array[0])
}
initial_guess = MachineState([
    theta_array[0],
    mandrel_pts[0, 2],
    np.linalg.norm(mandrel_pts[0, :2]),
    0.0
])

q0 = machine_ode.inverse_first_point(target0, initial_guess)

deploy_result = machine_ode.integrate(
    (s_array[0], s_array[-1]),
    q0.coords,
    tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
    s_eval=s_array
)

coords = deploy_result['coords']
s_out = deploy_result['s_array']

# Извлечение координат станка
theta_actual = coords[:, 0]
Z_actual = coords[:, 1] + z_offset
R_actual = coords[:, 2]
phi_actual_deg = np.degrees(coords[:, 3])

print(f"Развёртка: {len(s_out)} точек")


# ======================================================================
# 5. ВЕРИФИКАЦИЯ
# ======================================================================
print("\n===== 5. Верификация =====")

R_tsn_reconstructed = np.zeros_like(tsn_pts)
for i in range(len(s_out)):
    state = MachineState(coords[i])
    R_tsn_reconstructed[i] = machine_ode.forward(state)['point']

error = np.linalg.norm(tsn_pts - R_tsn_reconstructed, axis=1)
print(f"Средняя ошибка прямой задачи: {np.mean(error):.3e}")
print(f"Максимальная ошибка:          {np.max(error):.3e}")


# ======================================================================
# 6. ВИЗУАЛИЗАЦИЯ 3D
# ======================================================================
print("\n===== 6. Визуализация =====")

fig3d = go.Figure()

# --- Эллипсоид E1 ---
u_e = np.linspace(0, 2*np.pi, 80)
v_e = np.linspace(-np.pi/2, np.pi/2, 50)
Ue, Ve = np.meshgrid(u_e, v_e)
X1 = a1 * np.cos(Ue) * np.cos(Ve)
Y1 = b1 * np.sin(Ue) * np.cos(Ve)
Z1 = c1 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1, opacity=0.15, colorscale='Blues',
    showscale=False, name='E1 (эллипсоид)'
))

# --- Композит E2 ---
v_c = np.linspace(0, 2*np.pi, 80)
# Цилиндр
z_cyl = np.linspace(z_cyl_min, z_cyl_max, 40)
Vc, Zc = np.meshgrid(v_c, z_cyl)
Xc = R_cyl * np.cos(Vc)
Yc = R_cyl * np.sin(Vc)
fig3d.add_trace(go.Surface(
    x=Xc, y=Yc, z=Zc, opacity=0.3, colorscale='Reds',
    showscale=False, name='E2 (композит)'
))

# ТСН
fig3d.add_trace(go.Scatter3d(
    x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
    mode='lines', line=dict(color='blue', width=3), name='ТСН (E1)'
))

# ЛУ
fig3d.add_trace(go.Scatter3d(
    x=mandrel_pts[:, 0], y=mandrel_pts[:, 1], z=mandrel_pts[:, 2],
    mode='lines', line=dict(color='green', width=3), name='ЛУ (E2)'
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
    title='Развёртка: Эллипсоид + Композит -> Станок',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=30)
)
fig3d.write_html('ellipsoid_composite_kinematics_3d.html')


# ======================================================================
# 7. 2D-графики законов движения
# ======================================================================
fig_2d = make_subplots(rows=4, cols=1,
    subplot_titles=('θ(s), рад', 'Z(s)', 'R(s)', 'φ(s), град'))

fig_2d.add_trace(go.Scatter(x=s_out, y=theta_actual, mode='lines+markers',
    name='θ(s)', line=dict(color='black', width=2)), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=s_out, y=Z_actual, mode='lines+markers',
    name='Z(s)', line=dict(color='blue', width=2)), row=2, col=1)
fig_2d.add_trace(go.Scatter(x=s_out, y=R_actual, mode='lines+markers',
    name='R(s)', line=dict(color='green', width=2)), row=3, col=1)
fig_2d.add_trace(go.Scatter(x=s_out, y=phi_actual_deg, mode='lines+markers',
    name='φ(s)', line=dict(color='purple', width=2)), row=4, col=1)

fig_2d.update_xaxes(title_text='s', row=4, col=1)
fig_2d.update_yaxes(title_text='рад', row=1, col=1)
fig_2d.update_yaxes(title_text='мм', row=2, col=1)
fig_2d.update_yaxes(title_text='мм', row=3, col=1)
fig_2d.update_yaxes(title_text='град', row=4, col=1)
fig_2d.update_layout(height=1000, title_text='Законы движения', showlegend=True)
fig_2d.write_html('ellipsoid_composite_kinematics_2d.html')


# ======================================================================
# 8. СОХРАНЕНИЕ
# ======================================================================
results = {
    's': s_out,
    'theta': theta_actual,
    'Z': Z_actual,
    'R': R_actual,
    'phi': coords[:, 3],
    'z_offset': z_offset,
    'tsn_pts': tsn_pts,
    'mandrel_pts': mandrel_pts,
    'ring_radius': RING_RADIUS,
    'd_offset': D_OFFSET,
    'error_mean': float(np.mean(error)),
    'error_max': float(np.max(error))
}
scipy.io.savemat('ellipsoid_composite_kinematics.mat', results)

print("\n===== Готово =====")
print("Графики: ellipsoid_composite_kinematics_3d.html, ellipsoid_composite_kinematics_2d.html")
print("Данные: ellipsoid_composite_kinematics.mat")
