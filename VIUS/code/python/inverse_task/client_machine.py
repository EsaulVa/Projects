import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
from scipy.optimize import root

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.intersection import RayTracer, PiecewisePolynomialIntersection
from constraints.corridor_max_calculator import CorridorMaxCalculator
from constraints.corridor_min_calculator import CorridorMinCalculator
from machine.kinematics_base import *
from machine.machine3axis_exact import *
from machine.deployer import TrajectoryDeployer


# ======================================================================
# 3. ИСХОДНЫЕ ДАННЫЕ И РАСЧЕТ КОРРИДОРА
# ======================================================================
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]; cyl_r_opravka = 251.705
E2_opravka = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366, -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]; cyl_r_safe = 352.387
E1_safety = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)
z_offset = (bound_safe[3] - bound_opravka[3]) / 2

data_l = scipy.io.loadmat('LU_data.mat'); r_etalon = data_l['r']
lu_trajectory = Trajectory.from_points(r_etalon, method='nurbs')

print("===== 1. Расчет коридоров =====")
tracer = RayTracer()
tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
result_max = CorridorMaxCalculator(lu_trajectory, E1_safety, tracer, safe_distance=15.0).calculate(num_points=200)
result_min = CorridorMinCalculator(lu_trajectory, E2_opravka, safe_margin=10.0).calculate(num_points=200)


# Фильтрация: берем только те точки стены, которые прошли обе проверки
valid_mask = result_max.valid_mask & result_min.valid_mask
print(f"Точек после двойной фильтрации: {np.sum(valid_mask)} из 200")

# Подготовка чистой траектории на стене для станка
s_valid = result_max.s_array[valid_mask]
points_wall = result_max.safety_points[valid_mask]
# tsn_trajectory = Trajectory.from_points(points_wall, method='nurbs')
# Берем точки линии укладки на оправке для подсказки кинематике
lu_points_valid = result_max.lu_points[valid_mask]

# Создаем объект Trajectory на стене
tsn_trajectory = Trajectory.from_points(points_wall, method='cubic')

# ЗАПУСК ДИСПЕТЧЕРА (передаем ему и траекторию, и точки на оправке для умного старта)
print("\n===== 2. Пространственная развертка =====")
# machine = Machine3AxisExact(ring_radius=50.0, d_offset=100.0)
# deployer = TrajectoryDeployer(machine, mandrel_radius=cyl_r_opravka) # 251.705 из ваших данных

# dummy_theta = np.linspace(0, 10.0, len(s_valid))
# # alpha = np.arctan2(lu_points_on_mandrel[:,1], lu_points_on_mandrel[:,0])
# # theta_array = np.unwrap(alpha)  # разворачиваем углы, чтобы не было скачков
# # deploy_result = deployer.deploy(tsn_trajectory, theta_array, lu_points_on_mandrel)
# # Создание станка
# machine = Machine3AxisExact(ring_radius=50.0, d_offset=100.0)

# # Создание диспетчера
# deployer = TrajectoryDeployer(machine)

# # Вызов deploy
# # Обратите внимание: теперь нужно передать lu_points_on_mandrel (уже есть)
# deploy_result = deployer.deploy(tsn_trajectory, dummy_theta, lu_points_on_mandrel=lu_points_valid)
# Вычисляем реальный угол поворота оправки из точек касания
dummy_theta = np.linspace(0, 10.0, len(s_valid))
alpha = np.arctan2(lu_points_valid[:, 1], lu_points_valid[:, 0])
theta_array = np.unwrap(alpha)
# theta_array=dummy_theta

# Создание станка и диспетчера
machine = Machine3AxisExact(ring_radius=50.0, d_offset=100.0)
deployer = TrajectoryDeployer(machine)

# Запуск развёртки с правильным theta_array
deploy_result = deployer.deploy(tsn_trajectory, theta_array, lu_points_on_mandrel=lu_points_valid)

# # ======================================================================
# # 4. ЗАПУСК КИНЕМАТИКИ СТАНКА
# # ======================================================================
# print("\n===== 2. Пространственная развертка (Обратная задача кинематики) =====")
# # Параметры станка (примерные, основанные на графике)
# machine = Machine3AxisExact(ring_radius=20.0, d_offset=700.0)
# deployer = TrajectoryDeployer(machine)

# # Углы оправки (приближенно: длина дуги / средний радиус)
# dummy_theta = np.linspace(0, np.sum(np.diff(lu_trajectory._s_func(s_valid))) / 250.0, len(s_valid))

# deploy_result = deployer.deploy(tsn_trajectory, dummy_theta)
dummy_theta=theta_array.copy()
success_rate = np.sum(deploy_result['success'])
print(f"Успешно решено: {success_rate} из {len(dummy_theta)} точек ({100*success_rate/len(dummy_theta):.0f}%)")
if success_rate < len(dummy_theta):
    print("Примечание: Метод Ньютона может падать на сложных переходах без подбора реальных начальных приближений.")

# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ (3D + 2D) – с коррекцией систем координат
# ======================================================================
print("\n===== Построение графиков =====")

# Диагностика систем координат
print(f"lu_points_valid Z range: [{np.min(lu_points_valid[:,2]):.2f}, {np.max(lu_points_valid[:,2]):.2f}]")
print(f"points_wall Z range: [{np.min(points_wall[:,2]):.2f}, {np.max(points_wall[:,2]):.2f}]")
print(f"z_offset = {z_offset:.2f}")

# Извлекаем данные из результата развёртки
s_array = deploy_result['s_array']
coords = deploy_result['coords']
theta_actual = coords[:, 0]
Z_dep = coords[:, 1] + z_offset
R_dep = coords[:, 2]
Phi_dep_deg = np.degrees(coords[:, 3])

# --- 3D график ---
fig3d = go.Figure()

# 1. Оправка (E2) – локальная система, сдвигаем на z_offset
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

# 2. Стена безопасности (E1) – уже в глобальной системе, сдвиг не нужен
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

# 3. Траектория ТСН на стене (красная) – глобальная
fig3d.add_trace(go.Scatter3d(x=points_wall[:,0], y=points_wall[:,1], z=points_wall[:,2],
                             mode='lines', line=dict(color='red', width=3),
                             name='ТСН (на стене)'))

# 4. Линия укладки на оправке (зелёная) – предполагаем, что в локальной системе, поэтому сдвигаем
fig3d.add_trace(go.Scatter3d(x=lu_points_valid[:,0], y=lu_points_valid[:,1], z=lu_points_valid[:,2] + z_offset,
                             mode='lines', line=dict(color='green', width=3),
                             name='Линия укладки (оправка)'))

# 5. Траектория центра кольца (оранжевая) – используем реальные theta из решения
X_center = R_dep * np.cos(theta_actual)
Y_center = R_dep * np.sin(theta_actual)
fig3d.add_trace(go.Scatter3d(x=X_center, y=Y_center, z=Z_dep,
                             mode='lines', line=dict(color='orange', width=3, dash='dot'),
                             name='Центр кольца (рабочие органы)'))

fig3d.update_layout(title='Развертка кинематики: Стена -> Станок',
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                    margin=dict(l=0, r=0, b=0, t=30))
fig3d.write_html("kinematics_3d.html")

# --- 2D графики ---
fig_2d = make_subplots(rows=3, cols=1,
                       subplot_titles=('Координата Z(s)', 'Радиальное смещение R(s)', 'Угол по кольцу φ(s)'))

fig_2d.add_trace(go.Scatter(x=s_array, y=Z_dep, mode='lines+markers',
                            name='Z(s)', line=dict(color='blue', width=2)), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=R_dep, mode='lines+markers',
                            name='R(s)', line=dict(color='green', width=2)), row=2, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=Phi_dep_deg, mode='lines+markers',
                            name='φ(s)', line=dict(color='purple', width=2)), row=3, col=1)

fig_2d.update_xaxes(title_text="Натуральный параметр s, мм", row=3, col=1)
for i in range(1, 4):
    fig_2d.update_yaxes(title_text="мм / градусы", row=i, col=1)
fig_2d.update_layout(height=800, title_text="Законы движения рабочих органов станка", showlegend=True)
fig_2d.write_html("kinematics_2d_graphs.html")

print("Графики сохранены.")
# # ======================================================================
# # 5. ВИЗУАЛИЗАЦИЯ (3D + 2D Графики) – окончательная версия
# # ======================================================================
# print("\n===== Построение графиков =====")

# # Извлекаем данные из результата развёртки
# s_array = deploy_result['s_array']                      # натуральный параметр вдоль ТСН
# coords = deploy_result['coords']                        # (N,4): [theta, Z, R, phi]
# theta_sol = coords[:, 0]                                # реальный угол поворота оправки из решения
# Z_dep = coords[:, 1] + z_offset                         # координата каретки со смещением
# R_dep = coords[:, 2]                                    # радиальное смещение центра кольца
# Phi_dep_deg = np.degrees(coords[:, 3])                  # угол на кольце в градусах

# # Траектория ТСН на стене безопасности (из данных, которые мы использовали для развёртки)
# # Это те же точки, что и tsn_global в старом коде, но они хранятся в points_wall
# tsn_points = points_wall   # определена ранее

# # 3D график
# fig3d = go.Figure()

# # Поверхность оправки (E2)
# u_opr = np.linspace(0, 768.54, 40)
# v_opr = np.linspace(0, 2*np.pi, 30)
# Uo, Vo = np.meshgrid(u_opr, v_opr)
# Xo, Yo = np.zeros_like(Uo), np.zeros_like(Uo)
# Zo = Uo.copy()
# for i in range(Uo.shape[0]):
#     for j in range(Uo.shape[1]):
#         p = E2_opravka.position(Uo[i,j], Vo[i,j])
#         Xo[i,j], Yo[i,j] = p[0], p[1]
# fig3d.add_trace(go.Surface(x=Xo, y=Yo, z=Zo + z_offset, opacity=0.4,
#                            colorscale='Blues', showscale=False, name='Оправка'))

# # Поверхность безопасности (E1)
# u_safe = np.linspace(0, 955.956, 60)
# v_safe = np.linspace(0, 2*np.pi, 30)
# Us, Vs = np.meshgrid(u_safe, v_safe)
# Xs, Ys = np.zeros_like(Us), np.zeros_like(Us)
# Zs = Us.copy()
# for i in range(Us.shape[0]):
#     for j in range(Us.shape[1]):
#         p = E1_safety.position(Us[i,j], Vs[i,j])
#         Xs[i,j], Ys[i,j] = p[0], p[1]
# fig3d.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds',
#                            showscale=False, name='Стена безопасности'))

# # Заданная траектория ТСН (красная) – точки на стене
# fig3d.add_trace(go.Scatter3d(x=tsn_points[:,0], y=tsn_points[:,1], z=tsn_points[:,2],
#                              mode='lines', line=dict(color='red', width=3),
#                              name='Цель (ТСН на стене)'))
# # 4. Линия укладки на оправке (зелёная) – добавляем по вашему запросу
# fig3d.add_trace(go.Scatter3d(x=lu_points_valid[:,0], y=lu_points_valid[:,1], z=lu_points_valid[:,2],
#                              mode='lines', line=dict(color='green', width=3),
#                              name='Линия укладки (оправка)'))
# # Траектория центра кольца (оранжевая) – используем theta из решения для правильного вращения
# X_center = R_dep * np.cos(theta_sol)
# Y_center = R_dep * np.sin(theta_sol)
# fig3d.add_trace(go.Scatter3d(x=X_center, y=Y_center, z=Z_dep,
#                              mode='lines', line=dict(color='orange', width=3, dash='dot'),
#                              name='Центр кольца (рабочие органы)'))

# fig3d.update_layout(title='Развертка кинематики: Стена -> Станок',
#                     scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
#                     margin=dict(l=0, r=0, b=0, t=30))
# fig3d.write_html("kinematics_3d.html")

# # 2D графики зависимостей от натурального параметра s
# fig_2d = make_subplots(rows=3, cols=1,
#                        subplot_titles=('Координата Z(s)', 'Радиальное смещение R(s)', 'Угол по кольцу φ(s)'))

# fig_2d.add_trace(go.Scatter(x=s_array, y=Z_dep, mode='lines+markers',
#                             name='Z(s)', line=dict(color='blue', width=2)), row=1, col=1)
# fig_2d.add_trace(go.Scatter(x=s_array, y=R_dep, mode='lines+markers',
#                             name='R(s)', line=dict(color='green', width=2)), row=2, col=1)
# fig_2d.add_trace(go.Scatter(x=s_array, y=Phi_dep_deg, mode='lines+markers',
#                             name='φ(s)', line=dict(color='purple', width=2)), row=3, col=1)

# fig_2d.update_xaxes(title_text="Натуральный параметр s, мм", row=3, col=1)
# for i in range(1, 4):
#     fig_2d.update_yaxes(title_text="мм / градусы", row=i, col=1)
# fig_2d.update_layout(height=800, title_text="Законы движения рабочих органов станка", showlegend=True)
# fig_2d.write_html("kinematics_2d_graphs.html")

# print("Графики сохранены.")