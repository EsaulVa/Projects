import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
from scipy.interpolate import interp1d

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.intersection import RayTracer, PiecewisePolynomialIntersection
from constraints.corridor_max_calculator import CorridorMaxCalculator
from constraints.corridor_min_calculator import CorridorMinCalculator
from machine.machine3axis_exact import Machine3AxisExact, Machine3AxisExact_ODE,MachineState
from machine.deployer import TrajectoryDeployer

# ======================================================================
# 1. ИСХОДНЫЕ ДАННЫЕ И РАСЧЕТ КОРРИДОРА
# ======================================================================
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705
E2_opravka = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366, -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387
E1_safety = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)
z_offset = (bound_safe[3] - bound_opravka[3]) / 2

data_l = scipy.io.loadmat('LU_data.mat')
r_etalon = data_l['r']   # локальная линия укладки (N,3)
lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')

print("===== 1. Расчет коридоров =====")
tracer = RayTracer()
Num_points=200
tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
result_max = CorridorMaxCalculator(lu_trajectory, E1_safety, tracer, safe_distance=15.0).calculate(num_points=Num_points)
result_min = CorridorMinCalculator(lu_trajectory, E2_opravka, safe_margin=10.0).calculate(num_points=Num_points)

valid_mask = result_max.valid_mask & result_min.valid_mask
print(f"Точек после двойной фильтрации: {np.sum(valid_mask)} из 200")

points_wall = result_max.safety_points[valid_mask]          # глобальные ТСН
lu_points_local = result_max.lu_points[valid_mask]          # локальная линия укладки

tsn_trajectory = Trajectory.from_points(points_wall, method='cubic')

# ======================================================================
# 2. ПОДГОТОВКА РЕАЛЬНОГО theta_array (азимут точек касания в глобальной системе)
# ======================================================================
lu_points_global = lu_points_local.copy()
lu_points_global[:, 2] += z_offset
alpha = np.arctan2(lu_points_global[:, 1], lu_points_global[:, 0])
theta_array = np.unwrap(alpha)

# ======================================================================
# 3. ОБРАТНАЯ ЗАДАЧА КИНЕМАТИКИ
# ======================================================================
print("\n===== 2. Пространственная развертка =====")
# Создаём станок-интегратор
machine_ode = Machine3AxisExact_ODE(ring_radius=50.0, d_offset=100.0)

# Подготавливаем траектории как функции от s
# Допустим, у вас есть tsn_trajectory (объект Trajectory) и mandrel_points (массив Nx3)
# Строим сплайн-интерполяцию и её производную
from scipy.interpolate import CubicSpline
s_vals = np.linspace(0, tsn_trajectory.total_length, len(lu_points_global))
tsn_pts = np.array([tsn_trajectory.R(s) for s in s_vals])
mandrel_pts = lu_points_global  # уже глобальные

tsn_spline = CubicSpline(s_vals, tsn_pts, axis=0, bc_type='natural')
mandrel_spline = CubicSpline(s_vals, mandrel_pts, axis=0, bc_type='natural')
tsn_func = lambda s: tsn_spline(s)
mandrel_func = lambda s: mandrel_spline(s)
d_tsn_func = lambda s: tsn_spline(s, nu=1)
d_mandrel_func = lambda s: mandrel_spline(s, nu=1)

# Начальные условия для первой точки
target0 = {'point': tsn_func(s_vals[0]), 'r_mandrel': mandrel_func(s_vals[0])}
initial_guess = MachineState([theta_array[0], mandrel_pts[0,2], np.linalg.norm(mandrel_pts[0,:2]), 0.0])
q0 = machine_ode.inverse_first_point(target0, initial_guess)

# Интегрирование
deploy_result = machine_ode.integrate((s_vals[0], s_vals[-1]), q0.coords,
                               tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
                               s_eval=s_vals)

# result['coords'] – массив (N,4)

# success_rate = np.sum(deploy_result['success'])
# print(f"Успешно решено: {success_rate} из {len(theta_array)} точек ({100*success_rate/len(theta_array):.0f}%)")
# if success_rate < len(theta_array):
#     print("Примечание: некоторые точки не сошлись. Будет выполнена интерполяция.")

# ======================================================================
# 4. ИНТЕРПОЛЯЦИЯ НЕУДАЧНЫХ ТОЧЕК
# ======================================================================
coords = deploy_result['coords'].copy()
s_array = deploy_result['s_array']
# success = deploy_result['success']

# if not np.all(success):
#     for col in range(4):
#         valid_idx = np.where(success)[0]
#         if len(valid_idx) >= 2:
#             interp_func = interp1d(s_array[valid_idx], coords[valid_idx, col],
#                                    kind='linear', fill_value='extrapolate')
#             coords[:, col] = interp_func(s_array)
#     print("Неудачные точки интерполированы.")

# ===== Уточнение полученных координат =====
print("Уточнение координат итерационным методом...")
from machine.machine3axis_exact import Machine3AxisExact
machine_iter = Machine3AxisExact(ring_radius=50.0, d_offset=100.0)
coords_refined = np.zeros_like(coords)
for i, s in enumerate(s_array):
    target_data = {'point': tsn_func(s), 'r_mandrel': mandrel_func(s)}
    guess = MachineState(coords[i])
    try:
        refined = machine_iter.inverse(target_data, guess)
        coords_refined[i] = refined.coords
    except Exception as e:
        print(f"Refinement failed at i={i}: {e}, keeping original")
        coords_refined[i] = coords[i]
coords = coords_refined.copy()
theta_actual = coords[:, 0]
Z_actual = coords[:, 1] + z_offset
R_actual = coords[:, 2]
phi_actual_deg = np.degrees(coords[:, 3])

# theta_actual = coords[:, 0]
# Z_actual = coords[:, 1] + z_offset       # глобальная Z каретки
# R_actual = coords[:, 2]
# phi_actual_deg = np.degrees(coords[:, 3])

# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ======================================================================
print("\n===== Построение графиков =====")

# --- 3D график ---
fig3d = go.Figure()

# Оправка (со сдвигом)
u_opr = np.linspace(0, 768.54, 40)
v_opr = np.linspace(0, 2*np.pi, 30)
Uo, Vo = np.meshgrid(u_opr, v_opr)
Xo, Yo = np.zeros_like(Uo), np.zeros_like(Uo)
Zo = Uo.copy()
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = E2_opravka.position(Uo[i,j], Vo[i,j])
        Xo[i,j], Yo[i,j] = p[0], p[1]
fig3d.add_trace(go.Surface(x=Xo, y=Yo, z=Zo + z_offset, opacity=0.4,
                           colorscale='Blues', showscale=False, name='Оправка'))

# Стена безопасности
u_safe = np.linspace(0, 955.956, 60)
v_safe = np.linspace(0, 2*np.pi, 30)
Us, Vs = np.meshgrid(u_safe, v_safe)
Xs, Ys = np.zeros_like(Us), np.zeros_like(Us)
Zs = Us.copy()
for i in range(Us.shape[0]):
    for j in range(Us.shape[1]):
        p = E1_safety.position(Us[i,j], Vs[i,j])
        Xs[i,j], Ys[i,j] = p[0], p[1]
fig3d.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds',
                           showscale=False, name='Стена безопасности'))

# Траектория ТСН на стене (красная)
fig3d.add_trace(go.Scatter3d(x=points_wall[:,0], y=points_wall[:,1], z=points_wall[:,2],
                             mode='lines', line=dict(color='red', width=3),
                             name='ТСН (на стене)'))

# Линия укладки на оправке (зелёная, в глобальной системе)
lu_global = lu_points_local.copy()
lu_global[:, 2] += z_offset
fig3d.add_trace(go.Scatter3d(x=lu_global[:,0], y=lu_global[:,1], z=lu_global[:,2],
                             mode='lines', line=dict(color='green', width=3),
                             name='Линия укладки (оправка)'))

# Траектория центра кольца (оранжевая)
X_center = R_actual * np.cos(theta_actual)
Y_center = R_actual * np.sin(theta_actual)
fig3d.add_trace(go.Scatter3d(x=X_center, y=Y_center, z=Z_actual,
                             mode='lines', line=dict(color='orange', width=3, dash='dot'),
                             name='Центр кольца (рабочие органы)'))

fig3d.update_layout(title='Развертка кинематики: Стена -> Станок',
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                    margin=dict(l=0, r=0, b=0, t=30))
fig3d.write_html("kinematics_3d.html")

# --- 2D графики законов движения ---
fig_2d = make_subplots(rows=4, cols=1,
                       subplot_titles=('Угол поворота оправки θ(s), рад',
                                       'Координата каретки Z(s), мм',
                                       'Радиальное смещение R(s), мм',
                                       'Угол на кольце φ(s), град'))

fig_2d.add_trace(go.Scatter(x=s_array, y=theta_actual, mode='lines+markers',
                            name='θ(s)', line=dict(color='black', width=2)), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=Z_actual, mode='lines+markers',
                            name='Z(s)', line=dict(color='blue', width=2)), row=2, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=R_actual, mode='lines+markers',
                            name='R(s)', line=dict(color='green', width=2)), row=3, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=phi_actual_deg, mode='lines+markers',
                            name='φ(s)', line=dict(color='purple', width=2)), row=4, col=1)

fig_2d.update_xaxes(title_text="Натуральный параметр s, мм", row=4, col=1)
fig_2d.update_yaxes(title_text="рад", row=1, col=1)
fig_2d.update_yaxes(title_text="мм", row=2, col=1)
fig_2d.update_yaxes(title_text="мм", row=3, col=1)
fig_2d.update_yaxes(title_text="град", row=4, col=1)
fig_2d.update_layout(height=1000, title_text="Законы движения рабочих органов станка", showlegend=True)
fig_2d.write_html("kinematics_2d_graphs.html")

# ======================================================================
# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ======================================================================
results = {
    's': s_array,
    'theta': theta_actual,
    'Z': Z_actual,
    'R': R_actual,
    'phi': coords[:, 3],      # радианы
    # 'success': success,
    'z_offset': z_offset
}
scipy.io.savemat('kinematics_results.mat', results)
print("\nРезультаты сохранены в kinematics_results.mat")
print("Графики сохранены: kinematics_3d.html, kinematics_2d_graphs.html")

# Верификация: вычисляем ТСН по найденным координатам
R_tsn_reconstructed = np.zeros_like(points_wall)
for i in range(len(s_array)):
    state = MachineState(coords[i])
    R_tsn_reconstructed[i] = machine_ode.forward(state)['point']
    # print(np.linalg.norm(R_tsn_reconstructed[i]-points_wall[i])/np.linalg.norm(points_wall[i]))
# error = np.linalg.norm(points_wall - R_tsn_reconstructed, axis=1)
# Верификация: вычисляем ТСН по найденным координатам
target_points_original = np.array([tsn_func(s) for s in s_array])
error = np.linalg.norm(target_points_original - R_tsn_reconstructed, axis=1)
print(f"Средняя ошибка прямой задачи: {np.mean(error):.3e} мм, макс: {np.max(error):.3e} мм")
print(f"Средняя ошибка прямой задачи: {np.mean(error):.3e} мм, макс: {np.max(error):.3e} мм")

import pickle

# Сохраняем параметры станка
machine_params = {
    'ring_radius': 50.0,
    'd_offset': 100.0,
    'type': 'Machine3AxisExact_ODE'
}
with open('machine_params.pkl', 'wb') as f:
    pickle.dump(machine_params, f)
print("Параметры станка сохранены в machine_params.pkl")


# После уточнения, перед сохранением
s_vals_for_save = s_array   # это тот же s_vals, что использовался для сплайнов
tsn_pts_for_save = np.array([tsn_trajectory.R(s) for s in s_vals_for_save])   # пересчитайте, если не сохранили
# Но у вас уже был массив tsn_pts, созданный ранее, используйте его

results_full = {
    's': s_vals_for_save,
    'theta': theta_actual,
    'Z': Z_actual,
    'R': R_actual,
    'phi': coords[:, 3],
    'z_offset': z_offset,
    'tsn_pts': tsn_pts,          # вместо points_wall
    'mandrel_pts': mandrel_pts,  # вместо lu_points_global
    'ring_radius': 50.0,
    'd_offset': 100.0,
}
scipy.io.savemat('kinematics_results_full.mat', results_full)