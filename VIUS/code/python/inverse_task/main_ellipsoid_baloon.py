"""
Обратная задача: траектория R(z) на внешнем эллипсоиде E1,
линия укладки восстанавливается на составном баллоне E2
(цилиндр + сферические днища).

Используется меридиан на E1 (u=0) для простоты и наглядности.
Начальная точка на E2 выбрана из условия касания.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.trajectory import Trajectory
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.scipy_solver import SciPySolver
from inverse_winding.inverse_winding_builder import InvWindingLineBuilder, InverseWindingLineBuilder
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.const_dev_law import ConstantDeviation

# ----------------------------------------------------------------------
# 1. Поверхности
# ----------------------------------------------------------------------
# Внешний эллипсоид E1
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)

# Внутренняя составная поверхность — баллон
R_cyl = 0.8          # радиус цилиндра и сфер
L_cyl = 4.0          # длина цилиндрической части
z_cyl_min = -L_cyl / 2
z_cyl_max =  L_cyl / 2

cyl_seg = CylinderSegment(R_cyl, z_cyl_min, z_cyl_max)
lower_sphere = SphereSegment(R_cyl, z_cyl_min, is_upper=False)
upper_sphere = SphereSegment(R_cyl, z_cyl_max, is_upper=True)
E2 = CompositeSurface([lower_sphere, cyl_seg, upper_sphere])

# # ----------------------------------------------------------------------
# # 2. Траектория точки схода на E1 (меридиан u=0)
# # ----------------------------------------------------------------------
# # Выбираем v_start так, чтобы x0 = R_cyl (для касания на цилиндре)
# v_start = np.arccos(R_cyl / a1)   # положительный угол
# v_end = np.pi/2 - 0.1            # почти до верхнего полюса
# num_pts = 500
# v_vals = np.linspace(-v_end, v_end, num_pts)   # симметрично для охвата всего баллона
# meridian_pts = np.array([np.array(E1.position(0.0, v)) for v in v_vals])

# traj = Trajectory.from_points(meridian_pts, method='cubic', bc_type='natural')
# print(f"Длина траектории: {traj.total_length:.3f}")

# # ----------------------------------------------------------------------
# # 3. Начальная точка на E2 (условие касания)
# # ----------------------------------------------------------------------
# R0 = traj.R(0.0)
# # Поскольку v_start = arccos(R_cyl/a1), имеем x0 = a1*cos(v_start) = R_cyl, y0 = 0, z0 = c1*sin(v_start)
# # На цилиндре тому же z0 соответствует точка касания с u=0, v0 = z0
# v0 = R0[2]   # высота
# print(f"Начальная точка: u=0, v={v0:.4f}")
# ----------------------------------------------------------------------
# 2. Прямая задача: построение линии укладки на E1
# ----------------------------------------------------------------------
print("===== Прямая задача: построение линии укладки на E1 =====")

# Закон отклонения (постоянный малый угол)
deviation_law = ConstantDeviation(tan_theta=0.1)

# Решатель ОДУ
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)

# Строитель прямой задачи
forward_builder = ForwardWindingBuilder(
    surface=E1,
    deviation_law=deviation_law,
    solver=solver_forward,
    normalize_tangent=True,
    eps=1e-12
)

# Начальные условия на E1
u0, v0 = 0.0, 0.0          # экватор
alpha = np.pi / 6          # угол намотки 30°
s_end = 30.0                # длина линии
count_points=100
s_eval = np.linspace(0, s_end, count_points)

# Запуск прямой задачи
s_vals, line_E1 = forward_builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
    eval_points=s_eval
)

if not forward_builder.last_run_successful:
    raise RuntimeError("Прямая задача завершилась с ошибкой")

uv_E1 = forward_builder.get_uv_states()
print(f"Прямая задача: построено {len(s_vals)} точек на E1")

# ----------------------------------------------------------------------
# 3. Создание траектории точки схода из линии укладки на E1
# ----------------------------------------------------------------------
print("\n===== Построение траектории точки схода по точкам линии =====")

# Метод кубического сплайна по 3D-точкам
traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")


# ----------------------------------------------------------------------
# 4. Обратная задача
# ----------------------------------------------------------------------
rhs_calc = RightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=1.0,
    max_ds_dz=0.5
)

solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
inv_raw = InvWindingLineBuilder(E2, traj, rhs_calc, solver)
inverse_builder = InverseWindingLineBuilder(inv_raw)

z_eval = np.linspace(0, traj.total_length, 400)
z_vals, line_E2 = inverse_builder.build(
    initial_point=(0.0, v0),
    eval_points=z_eval
)

if not inverse_builder.last_run_successful:
    print("Обратная задача не удалась.")
    diag = inverse_builder.get_diagnostics()
    print(diag.get('message', 'нет сообщения'))
    exit(1)

print(f"Построено {len(z_vals)} точек.")
z_res, deltas = inverse_builder.get_residuals()
print(f"Максимальная невязка δ: {np.max(np.abs(deltas)):.4e}")

# ----------------------------------------------------------------------
# 5. График невязки
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(z_res, deltas, 'b-', linewidth=1.5)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('z (длина дуги)')
plt.ylabel('δ')
plt.title('Невязка δ вдоль траектории')
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# 6. 3D-визуализация
# ----------------------------------------------------------------------
# Сетки для поверхностей
u_grid = np.linspace(0, 2*np.pi, 80)
v_grid_E1 = np.linspace(-np.pi/2, np.pi/2, 50)
v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 80)

def surface_grid(surf, u_arr, v_arr):
    U, V = np.meshgrid(u_arr, v_arr)
    X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surf.position(U[i,j], V[i,j])
            X[i,j], Y[i,j], Z[i,j] = np.array(p)
    return X, Y, Z

X1, Y1, Z1 = surface_grid(E1, u_grid, v_grid_E1)
X2, Y2, Z2 = surface_grid(E2, u_grid, v_grid_E2)

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='Баллон E2'))

# Траектория на E1
fig.add_trace(go.Scatter3d(x=meridian_pts[:,0], y=meridian_pts[:,1], z=meridian_pts[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='R(z) на E1'))

# Восстановленная линия на E2
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                           mode='lines', line=dict(color='red', width=4), name='Линия укладки на E2'))

# Соединительные отрезки
step = 40
for i in range(0, len(z_vals), step):
    Rz = meridian_pts[i]
    Pz = line_E2[i]
    fig.add_trace(go.Scatter3d(x=[Rz[0], Pz[0]], y=[Rz[1], Pz[1]], z=[Rz[2], Pz[2]],
                               mode='lines', line=dict(color='green', width=2, dash='solid'), showlegend=False))

fig.update_layout(
    title='Обратная задача: эллипсоид → баллон',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=1000, height=800
)
fig.show()