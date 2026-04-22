"""
Контрольный пример восстановления линии укладки на эллипсоидальной оправке.

Траектория R(z) — меридиан на внешнем эллипсоиде E1 (поверхность безопасности).
Оправка — внутренний соосный эллипсоид E2.
Результат визуализируется в интерактивном 3D‑графике (Plotly).
"""
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# Импорт компонентов нашего фреймворка
from geometry.ellipsoid import EllipsoidWithDerivatives
from core.trajectory import Trajectory
# from trajectory import register_builders  # регистрирует кубический сплайн
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.scipy_solver import SciPySolver
from inverse_winding.inverse_winding_builder import WindingLineBuilder

"""
Контрольный пример: меридиан на эллипсоидах.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fsolve
import jax.numpy as jnp



# ----------------------------------------------------------------------
# 1. Эллипсоиды
# ----------------------------------------------------------------------
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)

scale = 0.6
a2, b2, c2 = a1*scale, b1*scale, c1*scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)

# ----------------------------------------------------------------------
# 2. Меридиан на E1 (u = 0)
# ----------------------------------------------------------------------
u_fixed = 0.0
v_start = -np.pi/2 + 0.2
v_end   =  np.pi/2 - 0.2
num_points = 500

v_vals = np.linspace(v_start, v_end, num_points)
geo_points = np.array([np.array(E1.position(u_fixed, v)) for v in v_vals])

print(f"Сгенерировано {len(geo_points)} точек меридиана на E1.")
traj = Trajectory.from_points(geo_points, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 3. Поиск начальной точки на E2 (линия тени для z=0)
# ----------------------------------------------------------------------
R0 = np.array(traj.R(0.0))

# ----------------------------------------------------------------------
# 3. Поиск начальной точки на E2 (линия тени для z=0)
# ----------------------------------------------------------------------
R0 = np.array(traj.R(0.0))

def shadow_equation(v):
    # Используем чистый NumPy, чтобы избежать проблем с JAX
    u = u_fixed
    a, b, c = E2.a, E2.b, E2.c
    cos_u, sin_u = np.cos(u), np.sin(u)
    cos_v, sin_v = np.cos(v), np.sin(v)

    # Точка на E2
    rx = a * cos_u * cos_v
    ry = b * sin_u * cos_v
    rz = c * sin_v
    r = np.array([rx, ry, rz])

    # Нормаль (ненормированная)
    nx = cos_u * cos_v / a
    ny = sin_u * cos_v / b
    nz = sin_v / c
    n_unnorm = np.array([nx, ny, nz])
    n = n_unnorm / np.linalg.norm(n_unnorm)

    return float(np.dot(R0 - r, n))

v0_guess = v_start
v0 = fsolve(shadow_equation, v0_guess)[0]
print(f"Начальная широта на E2: {v0:.4f}")

# Начальное приближение: можно взять ту же широту, что и у R0,
# но с небольшим смещением внутрь.
# Для R0 = (x0, 0, z0) приблизительно v0_guess = arctan(z0 / (scale * sqrt(...))),
# но для простоты используем v_start.
v0_guess = v_start
v0 = fsolve(shadow_equation, v0_guess)[0]
print(f"Начальная широта на E2: {v0:.4f}")

# ----------------------------------------------------------------------
# 4. Настройка и расчёт
# ----------------------------------------------------------------------
rhs_calc = RightHandSideCalculator(E2, traj, k=1.5, max_ds_dz=0.2)
solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
builder = WindingLineBuilder(E2, traj, rhs_calc, solver)

z_eval = np.linspace(0, traj.total_length, 400)
z_vals, line_3d = builder.compute(u_fixed, v0, z_eval=z_eval)
uv_states = builder.get_uv_states()
print(f"Рассчитано {len(z_vals)} точек.")

# Проверка: все u должны быть близки к u_fixed
u_vals = uv_states[:, 0]
max_dev = np.max(np.abs(u_vals - u_fixed))
print(f"Максимальное отклонение u от {u_fixed}: {max_dev:.2e}")

# ----------------------------------------------------------------------
# 5. Визуализация
# ----------------------------------------------------------------------
def surface_grid(surface, U, V):
    pts = np.zeros((U.shape[0], U.shape[1], 3))
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surface.position(U[i,j], V[i,j])
            pts[i,j] = np.array(p)
    return pts[...,0], pts[...,1], pts[...,2]

u_grid = np.linspace(0, 2*np.pi, 60)
v_grid = np.linspace(-np.pi/2, np.pi/2, 40)
U, V = np.meshgrid(u_grid, v_grid)

X1, Y1, Z1 = surface_grid(E1, U, V)
X2, Y2, Z2 = surface_grid(E2, U, V)

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='E2'))

# Траектория R(z)
fig.add_trace(go.Scatter3d(x=geo_points[:,0], y=geo_points[:,1], z=geo_points[:,2],
                           mode='lines', line=dict(color='red', width=4), name='R(z) на E1 (меридиан)'))

# Линия укладки
fig.add_trace(go.Scatter3d(x=line_3d[:,0], y=line_3d[:,1], z=line_3d[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='Линия укладки на E2'))

# Соединительные отрезки
step = 20
for i in range(0, len(z_vals), step):
    Rz = traj.R(z_vals[i])
    Pz = line_3d[i]
    fig.add_trace(go.Scatter3d(x=[Rz[0], Pz[0]], y=[Rz[1], Pz[1]], z=[Rz[2], Pz[2]],
                               mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False))

fig.update_layout(title='Восстановление линии укладки (меридиан)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)
fig.show()