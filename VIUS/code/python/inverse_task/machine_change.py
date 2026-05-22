# refine_with_kinematic_model.py
import numpy as np
import pickle
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline
from machine.machine3axis_exact import Machine3AxisExact_ODE
from machine.kinematic_model import KinematicModel
from machine.kinematics_base import MachineState
import plotly.graph_objects as go

# # ============================================================================
# # 1. ЗАГРУЗКА ДАННЫХ И ПАРАМЕТРОВ СТАНКА
# # ============================================================================
# with open('machine_params.pkl', 'rb') as f:
#     params = pickle.load(f)
# print(f"Загружены параметры станка: {params}")

# data = scipy.io.loadmat('kinematics_results_full.mat')
# s_array = data['s'].flatten()
# theta_orig = data['theta'].flatten()
# Z_global_orig = data['Z'].flatten()      # глобальная Z (с z_offset)
# R_orig = data['R'].flatten()
# phi_orig = data['phi'].flatten()
# points_wall = data['points_wall']
# lu_points_global = data['lu_points_global']
# # z_offset = float(data['z_offset'])
# z_offset = float(data['z_offset'].flatten()[0])

# # Переход к обобщённой координате каретки
# Z_carriage_orig = Z_global_orig - z_offset

# # Траектории как функции s (для интеграции)
# tsn_traj = CubicSpline(s_array, points_wall, axis=0, bc_type='natural')
# mandrel_traj = CubicSpline(s_array, lu_points_global, axis=0, bc_type='natural')
# d_tsn = tsn_traj.derivative(1)
# d_mandrel = mandrel_traj.derivative(1)

# machine_change.py (начало, до раздела 2)
import numpy as np
import pickle
import scipy.io
from scipy.interpolate import CubicSpline
# from machine.machine3axis_exact_ode import Machine3AxisExact_ODE
from machine.kinematic_model import KinematicModel
from machine.kinematics_base import MachineState

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ И ПАРАМЕТРОВ СТАНКА
# ============================================================================
# Загрузка параметров станка
with open('machine_params.pkl', 'rb') as f:
    params = pickle.load(f)
print(f"Загружены параметры станка: {params}")

# Загрузка данных из файла, сохранённого client_machine.py
data = scipy.io.loadmat('kinematics_results_full.mat')

# Извлекаем массивы
s_array = data['s'].flatten()                      # натуральный параметр (длина дуги)
theta_orig = data['theta'].flatten()               # угол поворота оправки, рад
Z_global_orig = data['Z'].flatten()                # глобальная координата каретки (с z_offset)
R_orig = data['R'].flatten()                       # радиальное смещение кольца, мм
phi_orig = data['phi'].flatten()                   # угол на кольце, рад

# Точки, по которым строились сплайны в клиенте
tsn_pts = data['tsn_pts']                          # массив (N,3) – точки ТСН на равномерной сетке s
mandrel_pts = data['mandrel_pts']                  # массив (N,3) – точки касания на оправке (глобальные)
z_offset = float(data['z_offset'].flatten()[0])    # смещение для приведения оправки к глобальной системе

# Переход к обобщённой координате каретки (без z_offset)
Z_carriage_orig = Z_global_orig - z_offset

# Построение сплайнов, идентичных использованным в клиенте
# Используем CubicSpline с bc_type='natural' (как в Trajectory и в клиенте)
tsn_traj = CubicSpline(s_array, tsn_pts, axis=0, bc_type='natural')
mandrel_traj = CubicSpline(s_array, mandrel_pts, axis=0, bc_type='natural')
d_tsn = tsn_traj.derivative(1)                     # производная ТСН по s
d_mandrel = mandrel_traj.derivative(1)             # производная точки касания по s

# Проверка невязки исходных координат (для самоконтроля)
# from machine.machine3axis_exact_ode import Machine3AxisExact_ODE
machine_test = Machine3AxisExact_ODE(ring_radius=params['ring_radius'],
                                     d_offset=params['d_offset'])
for i in [0, len(s_array)//2, -1]:
    state = MachineState([theta_orig[i], Z_carriage_orig[i], R_orig[i], phi_orig[i]])
    target_data = {'point': tsn_traj(s_array[i]), 'r_mandrel': mandrel_traj(s_array[i])}
    F = machine_test.residuals(target_data, state)
    print(f"i={i}, s={s_array[i]:.1f}, |F|={np.linalg.norm(F):.2e}")

# ============================================================================
# 2. СОЗДАНИЕ KINEMATICMODEL (продолжение следует)
# ============================================================================

# ============================================================================
# 2. СОЗДАНИЕ KINEMATICMODEL
# ============================================================================
machine = Machine3AxisExact_ODE(ring_radius=params['ring_radius'],
                                d_offset=params['d_offset'])
kin_model = KinematicModel(machine)

# ============================================================================
# 3. ФИКСАЦИЯ ОСИ Z С НЕБОЛЬШИМ СГЛАЖИВАНИЕМ
# ============================================================================
# Сглаживаем Z_carriage с помощью UnivariateSpline
# Z_spline = UnivariateSpline(s_array, Z_carriage_orig, s=1e3)  # подберите параметр s
# for i in range(len(s_array)):
#     state = MachineState([theta_orig[i], Z_carriage_orig[i], R_orig[i], phi_orig[i]])
#     F = machine.residuals({'point': tsn_traj(s_array[i]), 'r_mandrel': mandrel_traj(s_array[i])}, state)
#     print(i, np.linalg.norm(F))
for i in [0, len(s_array)//2, -1]:
    state = MachineState([theta_orig[i], Z_carriage_orig[i], R_orig[i], phi_orig[i]])
    F = machine.residuals({'point': tsn_traj(s_array[i]), 'r_mandrel': mandrel_traj(s_array[i])}, state)
    print(f"i={i}, s={s_array[i]:.1f}, |F|={np.linalg.norm(F):.2e}")
# Z_spline=CubicSpline(s_array,Z_carriage_orig,axis=0, bc_type='natural')
# Z_fixed = Z_spline(s_array)               # сглаженные значения Z_carriage
# Используйте сглаживающий сплайн с параметром s (чем больше s, тем сильнее сглаживание)
from scipy.interpolate import UnivariateSpline
# Z_spline = UnivariateSpline(s_array, Z_carriage_orig, s=1e3)   # подберите s под свой выброс
# Z_fixed = Z_spline(s_array)

# fixed_indices = [1]                       # индекс оси Z
# fixed_funcs = [lambda s: Z_spline(s)]     # функция, возвращающая Z_carriage(s)

# # Начальные значения свободных координат (theta, R, phi) из первой точки
# q0_free = np.array([theta_orig[0], R_orig[0], phi_orig[0]])
# Coord_T=Z_carriage_orig
Coord_T=R_orig
Coord_spline = UnivariateSpline(s_array, Coord_T, s=1e3)   # подберите s под свой выброс
Coord_fixed = Coord_spline(s_array)

fixed_indices = [2]                       # индекс оси Z
fixed_funcs = [lambda s: Coord_spline(s)]     # функция, возвращающая Z_carriage(s)

# Начальные значения свободных координат (theta, R, phi) из первой точки
q0_free = np.array([theta_orig[0], Z_carriage_orig[0], R_orig[0],phi_orig[0]])
q0_free=np.delete(q0_free,fixed_indices[0])

# ============================================================================
# 4. ИНТЕГРИРОВАНИЕ С ФИКСАЦИЕЙ
# ============================================================================
# print("Интегрирование с фиксацией оси Z...")
# result = kin_model.integrate_fixed(
#     s_span=(s_array[0], s_array[-1]),
#     q0_free=q0_free,
#     fixed_funcs=fixed_funcs,
#     fixed_indices=fixed_indices,
#     tsn_func=tsn_traj,
#     mandrel_func=mandrel_traj,
#     d_tsn_func=d_tsn,
#     d_mandrel_func=d_mandrel,
#     s_eval=s_array,
#     alpha=2.0
# )
result = kin_model.integrate_fixed_step(
    s_span=(s_array[0], s_array[-1]),
    q0_free=q0_free,
    fixed_funcs=fixed_funcs,
    fixed_indices=fixed_indices,
    tsn_func=tsn_traj,
    mandrel_func=mandrel_traj,
    d_tsn_func=d_tsn,
    d_mandrel_func=d_mandrel,
    step=0.1,                # шаг 10 мм (можно менять)
    s_eval=s_array,           # вернуть на исходной сетке
    alpha=2
)

s_new = result['s_array']
coords_new = result['coords']            # (N,4): [theta, Z_carriage, R, phi]
theta_new = coords_new[:, 0]
Z_carriage_new = coords_new[:, 1]
R_new = coords_new[:, 2]
phi_new = coords_new[:, 3]

# Глобальная Z для визуализации
Z_global_new = Z_carriage_new + z_offset

# ============================================================================
# 5. ВЕРИФИКАЦИЯ ПРЯМОЙ ЗАДАЧИ
# ============================================================================
# R_tsn_reconstructed = np.zeros_like(points_wall)
# используйте
R_tsn_reconstructed = np.zeros_like(tsn_pts)
for i in range(len(s_new)):
    state = MachineState(coords_new[i])
    R_tsn_reconstructed[i] = machine.forward(state)['point']
target_points = np.array([tsn_traj(s) for s in s_new])
error = np.linalg.norm(target_points - R_tsn_reconstructed, axis=1)
print(f"Средняя ошибка прямой задачи после коррекции: {np.mean(error):.3e} мм, макс: {np.max(error):.3e} мм")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ (сравнение координат)
# ============================================================================
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

axes[1,1].plot(s_array, phi_orig, 'b-', label='исходная')
axes[1,1].plot(s_new, phi_new, 'r--', label='скорректированная')
axes[1,1].set_ylabel('φ, рад')
axes[1,1].legend(); axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('comparison_fixed_Z.png')
plt.show()

# ============================================================================
# 7. 3D-ВИЗУАЛИЗАЦИЯ (как в клиенте)
# ============================================================================
# tsn_pts и mandrel_pts
points_wall=tsn_pts
lu_points_global=mandrel_pts
fig3d = go.Figure()
# Исходная ТСН (красная)
fig3d.add_trace(go.Scatter3d(x=points_wall[:,0], y=points_wall[:,1], z=points_wall[:,2],
                             mode='lines', line=dict(color='red', width=4), name='Исходная ТСН'))
# ТСН после коррекции (голубая)
fig3d.add_trace(go.Scatter3d(x=R_tsn_reconstructed[:,0], y=R_tsn_reconstructed[:,1], z=R_tsn_reconstructed[:,2],
                             mode='lines', line=dict(color='black', width=3, dash='solid'), name='ТСН после коррекции'))
# Центр кольца после коррекции (оранжевая)
X_center = R_new * np.cos(theta_new)
Y_center = R_new * np.sin(theta_new)
Z_center = Z_global_new
fig3d.add_trace(go.Scatter3d(x=X_center, y=Y_center, z=Z_center,
                             mode='lines', line=dict(color='orange', width=3, dash='dot'), name='Центр кольца (новый)'))
# Линия укладки (зелёная)
fig3d.add_trace(go.Scatter3d(x=lu_points_global[:,0], y=lu_points_global[:,1], z=lu_points_global[:,2],
                             mode='lines', line=dict(color='green', width=2), name='Линия укладки'))
fig3d.update_layout(title='Сравнение до/после коррекции (фиксация Z)',
                    scene=dict(aspectmode='data'))
fig3d.write_html("comparison_3d.html")
print("3D-сцена сохранена в comparison_3d.html")

# ============================================================================
# 8. СОХРАНЕНИЕ СКОРРЕКТИРОВАННЫХ ДАННЫХ
# ============================================================================
scipy.io.savemat('refined_kinematics.mat', {
    's': s_new,
    'theta': theta_new,
    'Z_carriage': Z_carriage_new,
    'R': R_new,
    'phi': phi_new,
    'z_offset': z_offset,
    'points_wall': points_wall,
    'lu_points_global': lu_points_global
})
print("Скорректированные данные сохранены в refined_kinematics.mat")