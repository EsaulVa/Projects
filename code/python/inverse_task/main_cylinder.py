"""
Сценарий: цилиндр внутри эллипсоида (соосные).
Траектория точки схода — меридиан на внешнем эллипсоиде.
Оправка — цилиндр.
"""

import numpy as np
import plotly.graph_objects as go

from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.cylinder import CylinderWithDerivatives
from core.trajectory import Trajectory
from winding.rhs_calculator import RightHandSideCalculator
from winding.scipy_solver import SciPySolver
from winding.winding_builder import WindingLineBuilder

# ----------------------------------------------------------------------
# 1. Параметры поверхностей
# ----------------------------------------------------------------------
# Внешний эллипсоид (поверхность безопасности)
a1, b1, c1 = 2.0, 2.0, 4.0   # ось Z — большая
E1 = EllipsoidWithDerivatives(a1, b1, c1)

# Внутренний цилиндр (оправка) — соосный, радиус меньше a1
radius = 0.8
E2 = CylinderWithDerivatives(radius)

# ----------------------------------------------------------------------
# 2. Меридиан на эллипсоиде (u=0)
# ----------------------------------------------------------------------
u_fixed = 0.0
# Чтобы лучи касались цилиндра, берём v_start из условия x0 = radius
v_max = np.arccos(radius / a1)
v_start = -v_max
v_end   =  v_max
num_points = 2500

num_points = 5000
v_vals = np.linspace(v_start, v_end, num_points)
geo_points = np.array([np.array(E1.position(u_fixed, v)) for v in v_vals])

# Производные на концах для clamped сплайна
def R_deriv_analytical(v):
    # Производная меридиана по v
    a, b, c = a1, b1, c1
    return np.array([-a * np.sin(u_fixed) * np.cos(v),  # = 0 при u=0
                     b * np.cos(u_fixed) * np.cos(v),   # = b * cos(v)
                     -c * np.sin(v)])                   # = -c * sin(v)

# Начальная и конечная производные
dR_start = R_deriv_analytical(v_start)
dR_end   = R_deriv_analytical(v_end)

traj = Trajectory.from_points(
    geo_points,
    method='cubic',
    bc_type='clamped',
    bc_start={'m': dR_start},   # нужно передать производные в формате, ожидаемом CubicSplineCurve
    bc_end={'m': dR_end}
)
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 3. Начальная точка на цилиндре (аналитически)
# ----------------------------------------------------------------------
# R(0) соответствует v = v_start
R0 = traj.R(0.0)
# Для цилиндра точка касания должна иметь u=0 и z = z0 (та же высота, что у R0)
v0 = R0[2]
print(f"Начальная высота на цилиндре: {v0:.4f}")

# ----------------------------------------------------------------------
# 4. Настройка RHS с защитой от деления на ноль
# ----------------------------------------------------------------------
class SafeRightHandSideCalculator(RightHandSideCalculator):
    def __call__(self, z, state):
        u, v = state
        # Если вторая квадратичная форма близка к нулю, ограничиваем ds/dz
        L, M, N = self.surface.second_fundamental_form(u, v)
        # Для цилиндра L = -R, M = 0, N = 0
        # На меридиане u' = 0, поэтому II = 0
        # В этом случае ds/dz должно определяться только геометрией, а не кривизной
        # Используем запасной вариант: ds/dz = ||R'(z)|| / ||R - r|| (приближение)
        try:
            return super().__call__(z, state)
        except ZeroDivisionError:
            R = self.trajectory.R(z)
            r = np.array(self.surface.position(u, v))
            diff = R - r
            diff_norm = np.linalg.norm(diff)
            R_deriv = self.trajectory.R_deriv(z)
            # Упрощённое ds/dz (без учёта кривизны)
            ds_dz = np.linalg.norm(R_deriv) / diff_norm
            # Ограничиваем
            if self.max_ds_dz and abs(ds_dz) > self.max_ds_dz:
                ds_dz = np.sign(ds_dz) * self.max_ds_dz
            # du/dz = 0 (остаёмся на меридиане), dv/dz = ds_dz / ||rv||
            rv = self.surface.derivatives(u, v)['rv']
            dv_dz = ds_dz / np.linalg.norm(rv)
            return 0.0, dv_dz

rhs_calc = SafeRightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=2.0,           # коррекция не нужна при точных данных
    max_ds_dz=0.5    # ограничение шага
)

# ----------------------------------------------------------------------
# 5. Интегратор и строитель
# ----------------------------------------------------------------------
solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
builder = WindingLineBuilder(E2, traj, rhs_calc, solver)

z_eval = np.linspace(0, traj.total_length, 400)
z_vals, line_3d = builder.compute(u_fixed, v0, z_eval=z_eval)
uv_states = builder.get_uv_states()
print(f"Рассчитано {len(z_vals)} точек.")

# ----------------------------------------------------------------------
# 6. Диагностика
# ----------------------------------------------------------------------
print("\nПервые 5 точек (u, v):")
for i in range(5):
    u_i, v_i = uv_states[i]
    print(f"z={z_vals[i]:.3f}: u={u_i:.6f}, v={v_i:.6f}")

u_vals = uv_states[:, 0]
max_dev = np.max(np.abs(u_vals - u_fixed))
print(f"Максимальное отклонение u от {u_fixed}: {max_dev:.2e}")

# Проверка невязки
print("\nПроверка невязки δ вдоль решения:")
for i in range(0, len(z_vals), 50):
    z = z_vals[i]
    Rz = traj.R(z)
    u, v = uv_states[i]
    r = np.array(E2.position(u, v))
    n = np.array(E2.normal(u, v))
    delta = np.dot(Rz - r, n)
    print(f"z={z:.3f}: δ={delta:.6e}")

# ----------------------------------------------------------------------
# 7. Визуализация
# ----------------------------------------------------------------------
def surface_grid(surf, U, V):
    pts = np.zeros((U.shape[0], U.shape[1], 3))
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            pts[i,j] = np.array(surf.position(U[i,j], V[i,j]))
    return pts[...,0], pts[...,1], pts[...,2]

# Эллипсоид
u_grid = np.linspace(0, 2*np.pi, 60)
v_grid = np.linspace(-np.pi/2, np.pi/2, 40)
U, V = np.meshgrid(u_grid, v_grid)
X1, Y1, Z1 = surface_grid(E1, U, V)

# Цилиндр (конечная высота)
v_cyl = np.linspace(v0 - 2.0, v0 + 2.0, 40)
Uc, Vc = np.meshgrid(u_grid, v_cyl)
X2, Y2, Z2 = surface_grid(E2, Uc, Vc)

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='E2'))
fig.add_trace(go.Scatter3d(x=geo_points[:,0], y=geo_points[:,1], z=geo_points[:,2],
                           mode='lines', line=dict(color='red', width=4), name='R(z)'))
fig.add_trace(go.Scatter3d(x=line_3d[:,0], y=line_3d[:,1], z=line_3d[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='Линия укладки'))

# Соединительные отрезки (каждый 30-й)
step = 30
for i in range(0, len(z_vals), step):
    Rz = traj.R(z_vals[i])
    Pz = line_3d[i]
    fig.add_trace(go.Scatter3d(x=[Rz[0], Pz[0]], y=[Rz[1], Pz[1]], z=[Rz[2], Pz[2]],
                               mode='lines', line=dict(color='green', width=2, dash='solid'), showlegend=False))

fig.update_layout(title='Восстановление линии укладки (цилиндр в эллипсоиде)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)
fig.write_html('winding_cylinder_fixed.html')
print("График сохранён в winding_cylinder_fixed.html")