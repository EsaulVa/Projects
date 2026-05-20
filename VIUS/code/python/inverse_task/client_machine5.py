# client_machine_5axis.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
from scipy.interpolate import CubicSpline

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.intersection import RayTracer, PiecewisePolynomialIntersection
from constraints.corridor_max_calculator import CorridorMaxCalculator
from constraints.corridor_min_calculator import CorridorMinCalculator
from machine.machine5axis import Machine5AxisExact_ODE, MachineState

# ======================================================================
# 1. ИСХОДНЫЕ ДАННЫЕ И РАСЧЕТ КОРРИДОРА (как в трёхосевом)
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
Num_points = 200
tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
result_max = CorridorMaxCalculator(lu_trajectory, E1_safety, tracer, safe_distance=15.0).calculate(num_points=Num_points)
result_min = CorridorMinCalculator(lu_trajectory, E2_opravka, safe_margin=10.0).calculate(num_points=Num_points)

valid_mask = result_max.valid_mask & result_min.valid_mask
print(f"Точек после двойной фильтрации: {np.sum(valid_mask)} из {Num_points}")

points_wall = result_max.safety_points[valid_mask]          # глобальные ТСН
lu_points_local = result_max.lu_points[valid_mask]          # локальная линия укладки
tsn_trajectory = Trajectory.from_points(points_wall, method='cubic')

# Глобальные точки касания на оправке (для визуализации и сплайнов)
lu_points_global = lu_points_local.copy()
lu_points_global[:, 2] += z_offset

# ======================================================================
# 2. ПОДГОТОВКА ТРАЕКТОРИЙ КАК ФУНКЦИЙ ОТ S
# ======================================================================
# Параметризация s (длина дуги траектории ТСН)
s_vals = np.linspace(0, tsn_trajectory.total_length, len(lu_points_global))
tsn_pts = np.array([tsn_trajectory.R(s) for s in s_vals])
mandrel_pts = lu_points_global

# Сплайны для точки схода нити (ТСН) и точки касания на оправке
tsn_spline = CubicSpline(s_vals, tsn_pts, axis=0, bc_type='natural')
mandrel_spline = CubicSpline(s_vals, mandrel_pts, axis=0, bc_type='natural')
tsn_func = lambda s: tsn_spline(s)
mandrel_func = lambda s: mandrel_spline(s)
d_tsn_func = lambda s: tsn_spline(s, nu=1)
d_mandrel_func = lambda s: mandrel_spline(s, nu=1)

# ======================================================================
# 3. ВЫЧИСЛЕНИЕ КАСАТЕЛЬНОЙ tau(s) И НОРМАЛИ m(s) НА ЛИНИИ УКЛАДКИ
# ======================================================================
# Используем исходную линию укладки lu_trajectory (в локальной системе)
# Строим сплайн для неё, чтобы получить производные.
# Поскольку lu_trajectory уже знает длину дуги, можно получить значения в точках s_vals.
# Но lu_trajectory.s_total не совпадает с tsn_trajectory.total_length.
# Поэтому лучше перепараметризовать линию укладки по своей длине дуги,
# а затем интерполировать на сетку s_vals (длину дуги ТСН).
# Упростим: вычислим tau и m в точках s_vals, используя исходные точки r_etalon.
# У нас есть массив r_etalon и его параметр u (хордовая длина). Создадим сплайн.

# Построим сплайн для линии укладки в локальной системе
lu_pts_local = r_etalon   # исходные точки (локальные)
# Вычисляем накопленную длину дуги для линии укладки
diffs = np.diff(lu_pts_local, axis=0)
dists = np.sqrt(np.sum(diffs**2, axis=1))
u_lu = np.zeros(len(lu_pts_local))
u_lu[1:] = np.cumsum(dists)
total_lu = u_lu[-1]
# Интерполяция локальной линии укладки на сетку s_vals (но s_vals – длина дуги ТСН)
# Это приближение, но допустим, что длины дуг близки. Более строго нужно маппинг.
# Для демонстрации используем интерполяцию по u_lu (своей длине дуги) и затем значения на s_vals.
s_normalized = s_vals / s_vals[-1] * total_lu  # пропорциональное масштабирование
lu_spline_local = CubicSpline(u_lu, lu_pts_local, axis=0, bc_type='natural')
lu_points_on_s = np.array([lu_spline_local(u) for u in s_normalized])  # локальные точки
# Касательные (производная по натуральному параметру линии укладки)
dlu_du = lu_spline_local.derivative(1)(s_normalized)
# Нормализуем касательные
tau_local = dlu_du / (np.linalg.norm(dlu_du, axis=1, keepdims=True) + 1e-12)
# Переводим касательные в глобальную систему (Z сдвигаем, но касательные не меняются)
tau_global = tau_local.copy()   # X,Y,Z остаются (но Z не сдвигаем, т.к. касательная вдоль Z не меняется)

# Вычисление нормали m(s) к поверхности оправки в точках линии укладки
# Для поверхности вращения E2_opravka, нормаль можно вычислить через градиент
def get_surface_normal(u, v):
    # u – координата вдоль оси, v – азимутальный угол
    # Параметризация поверхности: r(u,v) = (R(u)*cos(v), R(u)*sin(v), u)
    # Частные производные:
    # ru = (R'(u)*cos(v), R'(u)*sin(v), 1)
    # rv = (-R(u)*sin(v), R(u)*cos(v), 0)
    # Нормаль = ru × rv
    # Для численного вычисления можно использовать центральные разности
    eps = 1e-6
    p = E2_opravka.position(u, v)
    pu_plus = E2_opravka.position(u+eps, v)
    pu_minus = E2_opravka.position(u-eps, v)
    ru = (pu_plus - pu_minus) / (2*eps)
    pv_plus = E2_opravka.position(u, v+eps)
    pv_minus = E2_opravka.position(u, v-eps)
    rv = (pv_plus - pv_minus) / (2*eps)
    normal = np.cross(ru, rv)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return normal

# Для каждой точки линии укладки в локальной системе (lu_points_on_s) найдём (u,v) параметры
# Обратная задача: точка на поверхности вращения задана декартовыми координатами (x,y,z).
# Для цилиндрической части легко, но для произвольной поверхности нужен итеративный поиск.
# Упростим: используем цилиндрические координаты: u = z, v = arctan2(y,x).
# Для поверхности вращения это даст ближайшую точку на поверхности (проекция по радиусу).
# Но для точной нормали нужно именно по поверхности линии укладки, которая лежит на оправке.
# В данном случае lu_points_local – это точки лежащие на оправке, потому что result_max.lu_points
# получены трассировкой лучей до оправки. Поэтому можно использовать их напрямую.
# Для каждой точки lu_points_local (локальные) вычислим нормаль через поверхность.
# Приведём точки в локальную систему (z уже без смещения).
normals_local = []
for i, pt in enumerate(lu_points_local):
    # Цилиндрические координаты
    rho = np.sqrt(pt[0]**2 + pt[1]**2)
    v = np.arctan2(pt[1], pt[0])
    u = pt[2]
    # Вычисляем нормаль поверхности в точке (u,v)
    n = get_surface_normal(u, v)
    normals_local.append(n)
normals_local = np.array(normals_local)
# Переводим нормали в глобальную систему (Z сдвигаем, но нормали не меняются, только точка приложения)
normals_global = normals_local

# Создаём сплайны для tau(s) и m(s)
tau_spline = CubicSpline(s_vals, tau_global, axis=0, bc_type='natural')
m_spline = CubicSpline(s_vals, normals_global, axis=0, bc_type='natural')
tau_func = lambda s: tau_spline(s)
m_func = lambda s: m_spline(s)
d_tau_func = lambda s: tau_spline(s, nu=1)
d_m_func = lambda s: m_spline(s, nu=1)

# ======================================================================
# 2. ПОДГОТОВКА РЕАЛЬНОГО theta_array (азимут точек касания в глобальной системе)
# ======================================================================
lu_points_global = lu_points_local.copy()
lu_points_global[:, 2] += z_offset
alpha = np.arctan2(lu_points_global[:, 1], lu_points_global[:, 0])
theta_array = np.unwrap(alpha)

# ======================================================================
# 4. ПРОСТРАНСТВЕННАЯ РАЗВЕРТКА (5-КООРДИНАТНЫЙ СТАНОК)
# ======================================================================
print("\n===== 2. Пространственная развертка (5-осевой станок) =====")

# Параметры пятикоординатного станка (нужно подобрать для конкретной модели)
# В качестве примера используем значения из диссертации или предположительные
# ... (начало как ранее, до создания machine_ode)
# Параметры пятикоординатного станка (подставьте реальные значения из чертежей)
params_5axis = {
    'a2': 110.0,
    'a12': 0.0,
    'S4': 150.0,
    'S6': 120.0,
    'S7': 0.0,
    'S8': 80.0,
    'S9': 80.0,
    'S10': 60.0,
    'S11': 60.0,
}
# machine_ode = Machine5AxisExact_ODE(params_5axis)

# Начальное приближение: используем координаты из трёхосевого решения, если есть
# Или задаём разумные значения
initial_guess = MachineState(np.array([
    mandrel_pts[0,2] - z_offset,   # x1: продольное (каретка)
    0.0,                           # x2: поперечное
    0.0,                           # x3: угол головки
    0.0,                           # x4: угол ролика
    np.arctan2(mandrel_pts[0,1], mandrel_pts[0,0])   # x5: угол оправки
]))
machine_ode = Machine5AxisExact_ODE(params_5axis)
# print("a2=", machine_ode.a2)
# state_test = MachineState([0,0,0,0,0])
# print("forward point:", machine_ode.forward(state_test)['point'])
print("Тест forward при нулевых координатах:")
test_state = MachineState([0,0,0,0,0])
print(machine_ode.forward(test_state)['point'])

print("Тест изменения a2:")
params_5axis['a2'] = 300
machine_ode2 = Machine5AxisExact_ODE(params_5axis)
print(machine_ode2.forward(test_state)['point'])

from machine.machine3axis_exact import Machine3AxisExact
machine3 = Machine3AxisExact(50.0, 100.0)
# Начальные условия для первой точки
target0 = {'point': tsn_func(s_vals[0]), 'r_mandrel': mandrel_func(s_vals[0])}
guess = MachineState([theta_array[0], mandrel_pts[0,2], np.linalg.norm(mandrel_pts[0,:2]), 0.0])
# target0 = {'point': tsn_func(s_vals[0]), 'r_mandrel': mandrel_func(s_vals[0])}
# guess = MachineState([0.0, mandrel_pts[0,2] - z_offset, 250.0, 0.0])  # локальные
q3 = machine3.inverse(target0, guess)
print("Точное решение трёхосевой модели:", q3.coords)
# q3.coords: [theta, Z_carriage_local, R, phi]

# 2. Формируем начальное приближение для 5-осевого
#    x1 = Z_carriage (из 3-осевого), x2 = R, x3=0, x4=0, x5 = theta
initial_guess_5 = MachineState(np.array([
    abs(q3.coords[1])+100,      # x1: продольное перемещение каретки (Z_carriage)
    abs(q3.coords[2])+100,      # x2: поперечное смещение (R)
    0.0,               # x3: поворот головки (пока 0)
    0.0,               # x4: поворот ролика
    q3.coords[0]       # x5: угол оправки (theta)
]))
# Начальные условия для первой точки
# target0 = {
#     'point': tsn_func(s_vals[0]),
#     'tau': tau_func(s_vals[0]),
#     'm': m_func(s_vals[0])
# }
# target0 = {'point': np.array([0, 0, tsn_pts[0,2]]), 'tau': tau_func(s_vals[0]), 'm': m_func(s_vals[0])}
# Начальное приближение: 5 обобщённых координат
# x1 (продольное) – Z_carriage (глобальная Z точки касания, но нужно учитывать смещение)
# x2 (поперечное) – 0
# x3 (угол оправки) – азимут точки касания (theta_array[0])
# x4, x5 – 0 (начальное положение головки)
# initial_guess = MachineState(np.array([
#     mandrel_pts[0, 2] - z_offset,  # x1: локальная Z каретки (приближённо)
#     0.0,                           # x2: поперечное смещение
#     np.arctan2(mandrel_pts[0,1], mandrel_pts[0,0]),  # x3: азимут
#     0.0,                           # x4
#     0.0                            # x5
# ]))
F0 = machine_ode.residuals(target0, initial_guess)
print(f"Initial residuals: {F0}")
q0 = machine_ode.inverse_first_point(target0, initial_guess)

# Интегрирование
deploy_result = machine_ode.integrate(
    (s_vals[0], s_vals[-1]), q0.coords,
    tsn_func, tau_func, m_func,
    d_tsn_func, d_tau_func, d_m_func,
    s_eval=s_vals
)

coords = deploy_result['coords']   # (N,5)
s_array = deploy_result['s_array']

# Извлечение обобщённых координат
x1_actual = coords[:, 0]           # продольное перемещение каретки (локальная Z)
x2_actual = coords[:, 1]           # поперечное смещение
x3_actual = coords[:, 2]           # угол поворота оправки
x4_actual = np.degrees(coords[:, 3])  # угол головки (градусы)
x5_actual = np.degrees(coords[:, 4])  # угол ролика (градусы)

# Глобальная Z каретки для визуализации
Z_carriage_global = x1_actual + z_offset

# ======================================================================
# 5. ВЕРИФИКАЦИЯ
# ======================================================================
# Вычисляем восстановленную ТСН по найденным координатам
R_tsn_reconstructed = np.zeros_like(points_wall[:len(s_array)])
for i in range(len(s_array)):
    state = MachineState(coords[i])
    R_tsn_reconstructed[i] = machine_ode.forward(state)['point']
target_points = np.array([tsn_func(s) for s in s_array])
error = np.linalg.norm(target_points - R_tsn_reconstructed, axis=1)
print(f"Средняя ошибка прямой задачи: {np.mean(error):.3e} мм, макс: {np.max(error):.3e} мм")

# ======================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ======================================================================
print("\n===== Построение графиков =====")

# 2D графики для 5 координат
fig_2d = make_subplots(rows=5, cols=1,
                       subplot_titles=('x1 (продольная каретка), мм',
                                       'x2 (поперечное смещение), мм',
                                       'x3 (угол оправки), рад',
                                       'x4 (угол головки), град',
                                       'x5 (угол ролика), град'))
fig_2d.add_trace(go.Scatter(x=s_array, y=Z_carriage_global, mode='lines', name='x1'), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=x2_actual, mode='lines', name='x2'), row=2, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=x3_actual, mode='lines', name='x3'), row=3, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=x4_actual, mode='lines', name='x4'), row=4, col=1)
fig_2d.add_trace(go.Scatter(x=s_array, y=x5_actual, mode='lines', name='x5'), row=5, col=1)
fig_2d.update_layout(height=1200, title_text="Законы движения пятикоординатного станка")
fig_2d.write_html("kinematics_5axis_2d.html")

# 3D график: оправка, стена, ТСН, линия укладки, центр ролика (выходное звено)
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
# Траектория ТСН (красная)
fig3d.add_trace(go.Scatter3d(x=points_wall[:,0], y=points_wall[:,1], z=points_wall[:,2],
                             mode='lines', line=dict(color='red', width=3),
                             name='ТСН (на стене)'))
# Линия укладки (зелёная)
lu_global = lu_points_local.copy()
lu_global[:,2] += z_offset
fig3d.add_trace(go.Scatter3d(x=lu_global[:,0], y=lu_global[:,1], z=lu_global[:,2],
                             mode='lines', line=dict(color='green', width=3),
                             name='Линия укладки (оправка)'))
# Траектория выходного звена (центра ролика) – оранжевая
# Для пятикоординатного станка координаты выходного звена можно получить через forward
center_pts = np.array([machine_ode.forward(MachineState(coords[i]))['point'] for i in range(len(s_array))])
fig3d.add_trace(go.Scatter3d(x=center_pts[:,0], y=center_pts[:,1], z=center_pts[:,2],
                             mode='lines', line=dict(color='orange', width=3, dash='dot'),
                             name='Центр ролика'))
fig3d.update_layout(title='Развертка кинематики (5-осевой станок)',
                    scene=dict(aspectmode='data'))
fig3d.write_html("kinematics_5axis_3d.html")

# ======================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ======================================================================
results = {
    's': s_array,
    'x1': x1_actual,
    'x2': x2_actual,
    'x3': x3_actual,
    'x4': x4_actual,
    'x5': x5_actual,
    'z_offset': z_offset,
    'tsn_pts': tsn_pts,
    'mandrel_pts': mandrel_pts,
}
scipy.io.savemat('kinematics_5axis_results.mat', results)
print("\nРезультаты сохранены в kinematics_5axis_results.mat")
print("Графики сохранены: kinematics_5axis_2d.html, kinematics_5axis_3d.html")