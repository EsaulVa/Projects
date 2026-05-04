"""
Комбинированная задача: эллипсоид (E1) → баллон (E2).
Прямая задача на E1 генерирует линию укладки, которая становится траекторией R(z).
Обратная задача восстанавливает линию укладки на составном баллоне E2,
стартуя с той же параметрической точки (u0, v0).
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.scipy_solver import SciPySolver
from inverse_winding.inverse_winding_builder import InvWindingLineBuilder, InverseWindingLineBuilder

# ----------------------------------------------------------------------
# 1. Параметры поверхностей
# ----------------------------------------------------------------------
# Внешний эллипсоид E1
a1, b1, c1 = 2.0, 2.5, 5.0
E2 = EllipsoidWithDerivatives(a1, b1, c1)

# Внутренний баллон E2 (цилиндр + полусферы)
R_cyl = 4
L_cyl = 16.0
z_cyl_min = -L_cyl / 2
z_cyl_max =  L_cyl / 2

cyl_seg = CylinderSegment(R_cyl, z_cyl_min, z_cyl_max)
lower_sphere = SphereSegment(R_cyl, z_cyl_min, is_upper=False)
upper_sphere = SphereSegment(R_cyl, z_cyl_max, is_upper=True)
E1 = CompositeSurface([lower_sphere, cyl_seg, upper_sphere])

# ----------------------------------------------------------------------
# 2. Прямая задача на E1 (точно как в main_combined.py)
# ----------------------------------------------------------------------
print("===== Прямая задача: построение линии укладки на E1 =====")

deviation_law = ConstantDeviation(tan_theta=0.0)
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)

forward_builder = ForwardWindingBuilder(
    surface=E1,
    deviation_law=deviation_law,
    solver=solver_forward,
    normalize_tangent=True,
    eps=1e-12
)
# Начальная точка: на нижнем днище, немного отступив от полюса.
# Возьмём u0 = 0 (нулевая долгота), v0 = v_min + 0.1, где v_min – нижняя граница баллона.
v_start = E1.v_min +0.1  # отступим от самого низа, чтобы избежать полюса
u0 = 0.0
v0 = v_start
# u0, v0 = 0,-np.pi/3          # экватор
alpha = np.pi / 3          # 30°
s_end = 2
count_points = 20
s_eval = np.linspace(0, s_end, count_points)

s_vals, line_E1 = forward_builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
    eval_points=s_eval
)

if not forward_builder.last_run_successful:
    raise RuntimeError("Прямая задача завершилась с ошибкой")

print(f"Прямая задача: построено {len(s_vals)} точек на E1")

# ----------------------------------------------------------------------
# 3. Траектория из линии на E1
# ----------------------------------------------------------------------
print("\n===== Построение траектории точки схода =====")
traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 4. Обратная задача на E2 (старт из той же (u0, v0))
# ----------------------------------------------------------------------
print("\n===== Обратная задача: восстановление линии укладки на E2 =====")

rhs_calc = RightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=0.0,
    max_ds_dz=2.0,
    delta_clip=0.999,
    eps=1e-12
)

solver_inverse = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
builder_inv_raw = InvWindingLineBuilder(E2, traj, rhs_calc, solver_inverse)
inverse_builder = InverseWindingLineBuilder(builder_inv_raw)

z_eval = np.linspace(0, traj.total_length, count_points)
z_vals, line_E2 = inverse_builder.build(
    initial_point=(u0, v0),
    eval_points=z_eval
)

diag = inverse_builder.get_diagnostics()
if not diag['success']:
    print(f"Обратная задача не удалась: {diag['message']}")
    exit(1)

print(f"Обратная задача: построено {diag['num_points']} точек")
z_res, deltas = inverse_builder.get_residuals()
print(f"Максимальная невязка δ: {np.max(np.abs(deltas)):.4e}")

# График невязки
plt.figure(figsize=(10, 4))
plt.plot(z_res, deltas, 'b-', linewidth=1.5)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('z')
plt.ylabel('δ')
plt.title('Невязка δ(z)')
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# 5. Визуализация
# ----------------------------------------------------------------------
print("\n===== Построение 3D-графика =====")

def surface_grid(surf, u_arr, v_arr):
    U, V = np.meshgrid(u_arr, v_arr)
    X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surf.position(U[i,j], V[i,j])
            X[i,j], Y[i,j], Z[i,j] = np.array(p)
    return X, Y, Z

u_grid = np.linspace(0, 2*np.pi, 80)
v_grid_E1 = np.linspace(-np.pi/2, np.pi/2, 50)
v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 80)

X1, Y1, Z1 = surface_grid(E1, u_grid, v_grid_E1)
X2, Y2, Z2 = surface_grid(E2, u_grid, v_grid_E2)

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='Баллон E2'))

fig.add_trace(go.Scatter3d(x=line_E1[:,0], y=line_E1[:,1], z=line_E1[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='Траектория R(z)'))
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                           mode='lines', line=dict(color='red', width=4), name='Линия укладки на E2'))

step = 5
for i in range(0, len(z_vals), step):
    fig.add_trace(go.Scatter3d(x=[line_E1[i,0], line_E2[i,0]],
                               y=[line_E1[i,1], line_E2[i,1]],
                               z=[line_E1[i,2], line_E2[i,2]],
                               mode='lines', line=dict(color='green', width=2), showlegend=False))

fig.update_layout(title='Эллипсоид → баллон (прямая + обратная задача)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)

# fig.show()
# Сохраняем интерактивный график в HTML-файл
fig.write_html('winding_plot.html')
print("График сохранён в winding_plot.html")