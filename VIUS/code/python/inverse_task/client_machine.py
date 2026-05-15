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

# ======================================================================
# 1. КИНЕМАТИКА СТАНКА (Реальная математика 3-х осевого)
# ======================================================================
class MachineState:
    def __init__(self, coords): self.coords = np.asarray(coords, dtype=float)

import numpy as np
from scipy.optimize import least_squares

class Machine3AxisExact:
    def __init__(self, ring_radius, d_offset):
        self.r_ring = ring_radius
        self.d_off = d_offset

    def forward(self, state):
        theta, Z, R, phi = state.coords
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        r_tsn = self.r_ring - R 
        X = r_tsn * cp * ct - self.r_ring * sp * st
        Y = r_tsn * cp * st + self.r_ring * sp * ct
        Z = self.r_ring * sp + Z + self.d_off
        point_3d = np.array([X, Y, Z])
        tau = np.array([-self.r_ring * sp * ct - r_tsn * st, -self.r_ring * sp * st + r_tsn * ct, self.r_ring * cp])
        n = np.array([cp * ct, cp * st, sp])
        return {'point': point_3d, 'tau': tau, 'n': n}

    def residuals(self, target_data, state):
        res = self.forward(state)
        F = np.zeros(4)
        F[0] = res['point'][0] - target_data['point'][0]
        F[1] = res['point'][1] - target_data['point'][1]
        F[2] = res['point'][2] - target_data['point'][2]
        F[3] = np.dot(target_data['tau'], res['n']) - 1.0 
        return F

    def inverse(self, target_data, initial_guess):
        from scipy.optimize import least_squares
        # Используем least_squares вместо root. Он ищет минимум суммы квадратов невязок.
        # Это спасает от сходимости в "плохую" ветвь на сложной геометрии.
        res = least_squares(
            lambda x: self.residuals(target_data, MachineState(x)), 
            initial_guess.coords, 
            method='lm' # Метод Левенберга-Марквардта (самый надежный для сложных нелинейных систем)
        )
        if res.success and np.max(np.abs(res.fun)) < 1e-3: # Проверяем, что мы реально попали в цель
            return MachineState(res.x)
        return None # Если метрика большая, значит не сошлись

# ======================================================================
# 2. ДИСПЕТЧЕР РАЗВЕРТКИ
# ======================================================================
class TrajectoryDeployer:
    def __init__(self, machine, mandrel_radius=251.705): 
        self.machine = machine
        self.mandrel_r = mandrel_radius # Радиус цилиндрической части оправки

    def deploy(self, tsn_trajectory, theta_array, lu_points_on_mandrel):
        N = len(theta_array)
        history_coords = np.zeros((N, 4))
        success_flags = np.ones(N, dtype=bool)
        
        # 1. ПРАВИЛЬНЫЙ СТАРТ для нулевого шага
        # Углы phi и начальные Z, R нужно как-то задать. 
        # Для прототипа зададим их нулевыми или близкими к нулю, 
        # чтобы ролик смотрел параллельно оси оправки.
        state_0 = self.machine.inverse(
            {'point': tsn_trajectory.R(0.0), 'tau': tsn_trajectory.R_deriv(0.0)}, 
            MachineState([theta_array[0], lu_points_on_mandrel[0, 2], self.mandrel_r, 0.0])
        )
        if state_0 is not None: history_coords[0] = state_0.coords
        else: success_flags[0] = False

               # 2. МАРШЕВЫЙ ПРОХОД С УМНЫМ НАЧАЛОМ
        for i in range(1, N):
            s_val = tsn_trajectory.total_length * (i / (N - 1))
            target_data = {'point': tsn_trajectory.R(s_val), 'tau': tsn_trajectory.R_deriv(s_val)}
            
            # УМНЫЙ START: Берем координаты станка с предыдущего удачного шага...
            Z_prev = history_coords[i-1, 1]
            R_prev = history_coords[i-1, 2]
            phi_prev = history_coords[i-1, 3]
            
            # ...НО! Корректируем Z по текущей координате линии укладки на оправке.
            Z_current_lu = lu_points_on_mandrel[i, 2] # Текущая точка ЛУ
            
            # Если шаг по s большой, линейно экстраполируем разницу
            if i > 1:
                Z_prev_lu = lu_points_on_mandrel[i-1, 2]
                # Прибавляем к текущей точке ЛУ разницу в движении каретки
                Z_current_lu = Z_current_lu + (Z_prev - Z_prev_lu) * 0.5 
                
            # Собираем начальное приближение для текущего шага
            guess = MachineState(np.array([theta_array[i], Z_current_lu, R_prev, phi_prev]))
            
            state_i = self.machine.inverse(target_data, guess)
            
            if state_i is not None:
                history_coords[i] = state_i.coords
            else:
                success_flags[i] = False
                history_coords[i] = history_coords[i-1] # Замораживаем
        return {'success': success_flags, 'coords': history_coords, 'theta': theta_array}
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
lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')

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
machine = Machine3AxisExact(ring_radius=50.0, d_offset=100.0)
deployer = TrajectoryDeployer(machine, mandrel_radius=cyl_r_opravka) # 251.705 из ваших данных

dummy_theta = np.linspace(0, 10.0, len(s_valid))
deploy_result = deployer.deploy(tsn_trajectory, dummy_theta, lu_points_on_mandrel=lu_points_valid)

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
success_rate = np.sum(deploy_result['success'])
print(f"Успешно решено: {success_rate} из {len(dummy_theta)} точек ({100*success_rate/len(dummy_theta):.0f}%)")
if success_rate < len(dummy_theta):
    print("Примечание: Метод Ньютона может падать на сложных переходах без подбора реальных начальных приближений.")

# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ (3D + 2D Графики)
# ======================================================================
print("\n===== Построение графиков =====")
s_plot = deploy_result['theta'] # Используем угол как базовую ось для графиков
Z_dep = deploy_result['coords'][:, 1] + z_offset
R_dep = deploy_result['coords'][:, 2]
Phi_dep = deploy_result['coords'][:, 3]

# --- БЛОК 3D ГРАФИКА ---
fig3d = go.Figure()
# (Сюда вставьте ваш старый блок отрисовки поверхностей E2 и E1 из предыдущего клиента)
u_opr = np.linspace(0, 768.54, 40); v_opr = np.linspace(0, 2*np.pi, 30)
Uo, Vo = np.meshgrid(u_opr, v_opr); Zo = Uo.copy(); Xo, Yo = np.zeros_like(Uo), np.zeros_like(Uo)
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = E2_opravka.position(Uo[i,j], Vo[i,j]); Xo[i,j], Yo[i,j] = p[0], p[1]
fig3d.add_trace(go.Surface(x=Xo, y=Yo, z=Zo + z_offset, opacity=0.4, colorscale='Blues', showscale=False))

u_safe = np.linspace(0, 955.956, 60); v_safe = np.linspace(0, 2*np.pi, 30)
Us, Vs = np.meshgrid(u_safe, v_safe); Zs = Us.copy(); Xs, Ys = np.zeros_like(Us), np.zeros_like(Us)
for i in range(Us.shape[0]):
    for j in range(Us.shape[1]):
        p = E1_safety.position(Us[i,j], Vs[i,j]); Xs[i,j], Ys[i,j] = p[0], p[1]
fig3d.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds', showscale=False))

# Целевая траектория на стене (Красная)
tsn_global = points_wall.copy() # Они уже в глобальной системе (E1)
fig3d.add_trace(go.Scatter3d(x=tsn_global[:,0], y=tsn_global[:,1], z=tsn_global[:,2],
                          mode='lines', line=dict(color='red', width=3), name='Цель (ТСН на стене)'))

# Траектория рабочего органа станка (Оранжевая)
X_m = R_dep * np.cos(dummy_theta)
Y_m = R_dep * np.sin(dummy_theta)
fig3d.add_trace(go.Scatter3d(x=X_m, y=Y_m, z=Z_dep, mode='lines', 
                          line=dict(color='orange', width=3, dash='dot'), name='Органы станка (Z, R)'))

fig3d.update_layout(title='Развертка кинематики: Стена -> Станок', 
                   scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                   margin=dict(l=0, r=0, b=0, t=30))
fig3d.write_html("kinematics_3d.html")

# --- БЛОК 2D ГРАФИКОВ ЗАВИСИМОСТЕЙ (Собственно, то, что вы просили) ---
fig_2d = make_subplots(rows=3, cols=1, subplot_titles=('Координата Z(s)', 'Координата R(s)', 'Угол по кольцу Phi(s)'))

fig_2d['trace'][0].add_trace(go.Scatter(x=s_plot, y=Z_dep, mode='lines+markers', name='Z(s)', line=dict(color='blue', width=2)))
fig_2d['trace'][1].add_trace(go.Scatter(x=s_plot, y=R_dep, mode='lines+markers', name='R(s)', line=dict(color='green', width=2)))
fig_2d['trace'][2].add_trace(go.Scatter(x=s_plot, y=np.degrees(Phi_dep), mode='lines+markers', name='Phi(s)', line=dict(color='purple', width=2)))

for i in range(3):
    fig_2d[f'xaxis{i+1}'].set_title('Натуральный параметр s, мм', fontsize=10)
    fig_2d[f'yaxis{i+1}'].set_title('мм / градусы', fontsize=10)
    fig_2d.update_layout(height=800, title_text="Законы движения рабочих органов станка", showlegend=True)
fig_2d.write_html("kinematics_2d_graphs.html")

print("Графики сохранены.")