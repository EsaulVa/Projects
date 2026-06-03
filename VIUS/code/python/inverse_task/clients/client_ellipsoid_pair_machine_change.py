#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_ellipsoid_pair_machine_change.py
========================================
Аналог machine_change.py для данных client_ellipsoid_pair_machine.py.

Логика:
  1. Загружает развёртку из ellipsoid_pair_kinematics.mat.
  2. Создаёт KinematicModel на базе Machine3AxisExact_ODE.
  3. Фиксирует и сглаживает координату Z(s) (наиболее гладкая на скрине).
  4. Интегрирует с фиксированной Z — получает скорректированные θ, R, φ.
  5. Верификация прямой задачей + визуализация.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
import plotly.graph_objects as go
from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from machine.machine3axis_exact import Machine3AxisExact_ODE
from machine.kinematic_model import KinematicModel
from machine.kinematics_base import MachineState


# ======================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ======================================================================
data = scipy.io.loadmat('ellipsoid_pair_kinematics.mat')

s_array = data['s'].flatten()
theta_orig = data['theta'].flatten()
Z_global_orig = data['Z'].flatten()
R_orig = data['R'].flatten()
phi_orig = data['phi'].flatten()
z_offset = float(data['z_offset'].flatten()[0])

tsn_pts = data['tsn_pts']          # (N,3) — точки ТСН
mandrel_pts = data['mandrel_pts']  # (N,3) — точки ЛУ

# Параметры станка
RING_RADIUS = float(data['ring_radius'].flatten()[0])
D_OFFSET = float(data['d_offset'].flatten()[0])

print(f"Загружено {len(s_array)} точек")
print(f"Параметры станка: ring={RING_RADIUS:.3f}, d={D_OFFSET:.3f}")
print(f"z_offset = {z_offset:.3f}")

# Переход к обобщённой координате каретки (без z_offset)
Z_carriage_orig = Z_global_orig - z_offset

# Сплайны для интеграции
tsn_traj = CubicSpline(s_array, tsn_pts, axis=0, bc_type='natural')
mandrel_traj = CubicSpline(s_array, mandrel_pts, axis=0, bc_type='natural')
d_tsn = tsn_traj.derivative(1)
d_mandrel = mandrel_traj.derivative(1)


# ======================================================================
# 2. ПРОВЕРКА ИСХОДНЫХ КООРДИНАТ
# ======================================================================
machine = Machine3AxisExact_ODE(ring_radius=RING_RADIUS, d_offset=D_OFFSET)

for i in [0, len(s_array)//2, -1]:
    state = MachineState([theta_orig[i], Z_carriage_orig[i], R_orig[i], phi_orig[i]])
    target_data = {
        'point': tsn_traj(s_array[i]),
        'r_mandrel': mandrel_traj(s_array[i])
    }
    F = machine.residuals(target_data, state)
    print(f"i={i}, s={s_array[i]:.1f}, ||F||={np.linalg.norm(F):.2e}")


# ======================================================================
# 3. СОЗДАНИЕ KINEMATICMODEL
# ======================================================================
kin_model = KinematicModel(machine)


# ======================================================================
# 4. ФИКСАЦИЯ ОСИ Z (сглаживание UnivariateSpline)
# ======================================================================
# Z(s) — наиболее гладкая координата (см. скрин), амплитуда ~±1.5 мм.
# Сглаживаем с параметром s — чем больше, тем сильнее сглаживание.
# Для эллипсоидов (масштаб ~1-3 мм) берём s=1e-2.

Z_spline = UnivariateSpline(s_array, Z_carriage_orig, s=2)
Z_fixed = Z_spline(s_array)

fixed_indices = [1]           # индекс Z в coords [theta, Z, R, phi]
fixed_funcs = [lambda s: Z_spline(s)]

# Начальные свободные координаты: theta, R, phi
q0_free = np.array([theta_orig[0], R_orig[0], phi_orig[0]])

print(f"\nФиксация Z: сглаживание s=1e-2")
print(f"  max |Z_orig - Z_fixed| = {np.max(np.abs(Z_carriage_orig - Z_fixed)):.4e}")


# ======================================================================
# 5. ИНТЕГРИРОВАНИЕ С ФИКСАЦИЕЙ
# ======================================================================
result = kin_model.integrate_fixed_step(
    s_span=(s_array[0], s_array[-1]),
    q0_free=q0_free,
    fixed_funcs=fixed_funcs,
    fixed_indices=fixed_indices,
    tsn_func=tsn_traj,
    mandrel_func=mandrel_traj,
    d_tsn_func=d_tsn,
    d_mandrel_func=d_mandrel,
    step=0.1,          # шаг 0.1 мм (мелкий для точности)
    s_eval=s_array,    # вернуть на исходной сетке
    alpha=2.0
)

s_new = result['s_array']
coords_new = result['coords']   # (N,4): [theta, Z_carriage, R, phi]

theta_new = coords_new[:, 0]
Z_carriage_new = coords_new[:, 1]
R_new = coords_new[:, 2]
phi_new = coords_new[:, 3]

Z_global_new = Z_carriage_new + z_offset

print(f"\nИнтегрирование завершено: {len(s_new)} точек")


# ======================================================================
# 6. ВЕРИФИКАЦИЯ ПРЯМОЙ ЗАДАЧИ
# ======================================================================
R_tsn_reconstructed = np.zeros_like(tsn_pts)
for i in range(len(s_new)):
    state = MachineState(coords_new[i])
    R_tsn_reconstructed[i] = machine.forward(state)['point']

target_points = np.array([tsn_traj(s) for s in s_new])
error = np.linalg.norm(target_points - R_tsn_reconstructed, axis=1)

print(f"Средняя ошибка после коррекции: {np.mean(error):.3e}")
print(f"Максимальная ошибка:            {np.max(error):.3e}")


# ======================================================================
# 7. ВИЗУАЛИЗАЦИЯ 2D (сравнение координат)
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(s_array, theta_orig, 'b-', label='исходная')
axes[0,0].plot(s_new, theta_new, 'r--', label='скорректированная')
axes[0,0].set_ylabel('θ, рад')
axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(s_array, Z_carriage_orig, 'b-', label='исходная')
axes[0,1].plot(s_new, Z_carriage_new, 'r--', label='скорректированная (фиксирована)')
axes[0,1].set_ylabel('Z_carriage, мм')
axes[0,1].legend(); axes[0,1].grid(True)

axes[1,0].plot(s_array, R_orig, 'b-', label='исходная')
axes[1,0].plot(s_new, R_new, 'r--', label='скорректированная')
axes[1,0].set_ylabel('R, мм')
axes[1,0].legend(); axes[1,0].grid(True)

axes[1,1].plot(s_array, np.degrees(phi_orig), 'b-', label='исходная')
axes[1,1].plot(s_new, np.degrees(phi_new), 'r--', label='скорректированная')
axes[1,1].set_ylabel('φ, град')
axes[1,1].legend(); axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('ellipsoid_pair_comparison_fixed_Z.png', dpi=150)
plt.show()
print("График сохранён: ellipsoid_pair_comparison_fixed_Z.png")


# ======================================================================
# 8. 3D-ВИЗУАЛИЗАЦИЯ
# ======================================================================
fig3d = go.Figure()

# Исходная ТСН
fig3d.add_trace(go.Scatter3d(
    x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
    mode='lines', line=dict(color='red', width=4), name='Исходная ТСН'
))

# ТСН после коррекции
fig3d.add_trace(go.Scatter3d(
    x=R_tsn_reconstructed[:,0], y=R_tsn_reconstructed[:,1], z=R_tsn_reconstructed[:,2],
    mode='lines', line=dict(color='black', width=3), name='ТСН после коррекции'
))

# Центр кольца (новый)
X_center = R_new * np.cos(theta_new)
Y_center = R_new * np.sin(theta_new)
Z_center = Z_global_new
fig3d.add_trace(go.Scatter3d(
    x=X_center, y=Y_center, z=Z_center,
    mode='lines', line=dict(color='orange', width=3, dash='dot'),
    name='Центр кольца (новый)'
))

# Линия укладки
fig3d.add_trace(go.Scatter3d(
    x=mandrel_pts[:,0], y=mandrel_pts[:,1], z=mandrel_pts[:,2],
    mode='lines', line=dict(color='green', width=2), name='Линия укладки'
))

fig3d.update_layout(
    title='Сравнение до/после коррекции (фиксация Z) — пара эллипсоидов',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data')
)
fig3d.write_html("ellipsoid_pair_comparison_3d.html")
print("3D-сцена сохранена: ellipsoid_pair_comparison_3d.html")


# ======================================================================
# 9. СОХРАНЕНИЕ
# ======================================================================
scipy.io.savemat('ellipsoid_pair_refined_kinematics.mat', {
    's': s_new,
    'theta': theta_new,
    'Z_carriage': Z_carriage_new,
    'Z_global': Z_global_new,
    'R': R_new,
    'phi': phi_new,
    'z_offset': z_offset,
    'tsn_pts': tsn_pts,
    'mandrel_pts': mandrel_pts,
    'error_mean': float(np.mean(error)),
    'error_max': float(np.max(error))
})
print("\nСкорректированные данные сохранены: ellipsoid_pair_refined_kinematics.mat")
