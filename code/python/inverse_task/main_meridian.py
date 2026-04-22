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
from winding.rhs_calculator import RightHandSideCalculator
from winding.scipy_solver import SciPySolver
from winding.winding_builder import WindingLineBuilder

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

# ----------------------------------------------------------------------
# 3. Поиск начальной точки на E2 (линия тени для z=0)
# ----------------------------------------------------------------------
R0 = np.array(traj.R(0.0))

def shadow_equation(v):
    v = float(v[0])                         # гарантируем, что v – число
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

    dot_val = np.dot(R0 - r, n)
    return dot_val.item()                # извлекаем Python float

v0_guess = v_start
v0 = fsolve(shadow_equation, v0_guess)[0]
print(f"Начальная широта на E2: {v0:.4f}")
# ------------------ ДИАГНОСТИКА НАЧАЛЬНОЙ ТОЧКИ ------------------
r0 = np.array(E2.position(u_fixed, v0))
n0 = np.array(E2.normal(u_fixed, v0))
R0 = np.array(traj.R(0.0))
R0_deriv = np.array(traj.R_deriv(0.0))

print("=== Диагностика начальной точки ===")
print(f"u_fixed = {u_fixed}")
print(f"v0 = {v0:.6f}")
print(f"R(0)    = {R0}")
print(f"r(0)    = {r0}")
print(f"n(0)    = {n0}")
print(f"δ = <R-r, n> = {np.dot(R0 - r0, n0):.6f}")
print(f"<R'(0), n>    = {np.dot(R0_deriv, n0):.6f}")

# Создаём временный RHS и считаем первый шаг
rhs_temp = RightHandSideCalculator(E2, traj, k=1.5, max_ds_dz=0.2)
du_dz, dv_dz = rhs_temp(0.0, np.array([u_fixed, v0]))
print(f"Первый шаг: du/dz = {du_dz:.6f}, dv/dz = {dv_dz:.6f}")
print("=" * 40)
print(f"R_deriv(0) = {R0_deriv}")
print(f"y-компонента R_deriv(0): {R0_deriv[1]:.6e}")
# Проверка отклонения y в траектории
z_test = np.linspace(0, traj.total_length, 1000)
y_vals = [traj.R(z)[1] for z in z_test]
print(f"Max |y| в R(z): {np.max(np.abs(y_vals)):.6e}")

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
print("\nПервые 15 точек (u, v):")
for i in range(min(15, len(z_vals))):
    u_i, v_i = uv_states[i]
    print(f"z={z_vals[i]:5.3f}: u={u_i:.8f}, v={v_i:.8f}")
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
                               mode='lines', line=dict(color='green', width=2, dash='solid'), showlegend=False))

fig.update_layout(title='Восстановление линии укладки (меридиан)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)
# fig.show()
# Сохраняем интерактивный график в HTML-файл
fig.write_html('winding_plot.html')
print("График сохранён в winding_plot.html")
y_vals_line = line_3d[:, 1]
print(f"Max |y| в линии укладки: {np.max(np.abs(y_vals_line)):.6e}")
print(f"Диапазон Y линии укладки: [{np.min(line_3d[:,1]):.6f}, {np.max(line_3d[:,1]):.6f}]")
print("Первые 5 точек линии укладки (x,y,z):")
for i in range(5):
    print(f"  {line_3d[i]}")

def is_outside(x, y, z, a, b, c):
    return (x**2 / a**2 + y**2 / b**2 + z**2 / c**2) >= 1.0

# Проверим каждый 20-й отрезок
for i in range(0, len(z_vals), 20):
    Rz = traj.R(z_vals[i])
    Pz = line_3d[i]
    # Берем 5 пробных точек вдоль отрезка
    for t in np.linspace(0, 1, 5, endpoint=False):
        pt = Rz * t + Pz * (1 - t)
        if not is_outside(pt[0], pt[1], pt[2], a2, b2, c2):
            print(f"ОТРЕЗОК {i} ПРОНЗАЕТ ОПРАВКУ при t={t:.2f}")
            break
    else:
        continue
    break
else:
    print("Все проверенные отрезки снаружи оправки.")
print("\n=== Проверка отрезка 20 ===")
problematic_idx = 20
Rz = traj.R(z_vals[problematic_idx])
Pz = line_3d[problematic_idx]

def F2(x, y, z):
    return x**2/a2**2 + y**2/b2**2 + z**2/c2**2 - 1.0

print(f"R(z) на E1: {F2(*Rz):.6e} (должно быть около {a1**2/a2**2 - 1:.3f})")
print(f"P(z) на E2: {F2(*Pz):.6e} (должно быть ~0)")

for t in np.linspace(0, 1, 11):
    pt = Rz * t + Pz * (1 - t)
    val = F2(*pt)
    print(f"t={t:.1f}: F2={val:.6e} -> {'внутри' if val < -1e-9 else 'снаружи'}")
# Можно автоматически открыть файл в браузере
import webbrowser
webbrowser.open('winding_plot.html')