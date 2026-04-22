"""
Сценарий: цилиндр внутри эллипсоида с аналитической траекторией меридиана.
Траектория R(z) задана точно через эллиптические интегралы.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.special import ellipe, ellipkinc
from scipy.optimize import root_scalar
from scipy.interpolate import CubicSpline

from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.cylinder import CylinderWithDerivatives
from winding.rhs_calculator import RightHandSideCalculator
from winding.scipy_solver import SciPySolver
from winding.winding_builder import WindingLineBuilder

# ----------------------------------------------------------------------
# 1. Аналитическая траектория меридиана эллипсоида
# ----------------------------------------------------------------------
class AnalyticMeridianTrajectory:
    """
    Точная параметризация меридиана эллипсоида вращения (a=b) длиной дуги.
    Параметры:
        a, c - полуоси эллипсоида
        u_fixed - долгота (для меридиана 0)
        v_range - (v_start, v_end) - диапазон широт
        num_samples - число точек для табуляции обратной функции
    """
    def __init__(self, a, c, u_fixed=0.0, v_range=(-np.pi/2, np.pi/2), num_samples=1000):
        self.a = a
        self.c = c
        self.u_fixed = u_fixed
        self.v_start, self.v_end = v_range
        # Табулируем длину дуги s(v) и обратную v(s)
        v_vals = np.linspace(self.v_start, self.v_end, num_samples)
        s_vals = np.zeros_like(v_vals)
        # Параметр эллиптического интеграла
        # Для меридиана x = a cos(v), z = c sin(v)
        # ds = sqrt(a^2 sin^2 v + c^2 cos^2 v) dv
        # Пусть k^2 = 1 - (c/a)^2 если a >= c, иначе другая формула
        if a >= c:
            k = np.sqrt(1.0 - (c/a)**2)
            # s(v) = a * E(v, k) от 0 до v (если v_start=0)
            # Используем неполный эллиптический интеграл второго рода
            def s_func(v):
                # ellipkinc(phi, m) где m = k^2, phi в радианах
                return a * ellipkinc(v, k**2)
        else:
            # c > a, переформулируем через параметр t = ... но проще: используем численное интегрирование
            # Мы можем просто численно проинтегрировать ds/dv с высокой точностью
            from scipy.integrate import quad
            def integrand(v):
                return np.sqrt(a**2 * np.sin(v)**2 + c**2 * np.cos(v)**2)
            s_vals[0] = 0.0
            for i in range(1, len(v_vals)):
                s_vals[i] = s_vals[i-1] + quad(integrand, v_vals[i-1], v_vals[i])[0]
            self.s_func = lambda v: np.interp(v, v_vals, s_vals)
            self.v_of_s = CubicSpline(s_vals, v_vals, extrapolate=True)
            self.total_length = s_vals[-1]
            self._v_vals = v_vals
            self._s_vals = s_vals
            return

        # Для a >= c
        s_vals = a * ellipkinc(v_vals, k**2)
        # Сдвигаем, чтобы s(v_start) = 0
        s_vals -= s_vals[0]
        self.total_length = s_vals[-1]
        self._v_vals = v_vals
        self._s_vals = s_vals
        # Обратная интерполяция
        self.v_of_s = CubicSpline(s_vals, v_vals, extrapolate=True)

    def R(self, z):
        """Точка на меридиане по длине дуги z."""
        v = float(self.v_of_s(z))
        # Координаты на эллипсоиде
        x = self.a * np.cos(self.u_fixed) * np.cos(v)
        y = self.a * np.sin(self.u_fixed) * np.cos(v)  # = 0 при u_fixed=0
        z_coord = self.c * np.sin(v)
        return np.array([x, y, z_coord])

    def R_deriv(self, z):
        """Касательный вектор (единичный)."""
        v = float(self.v_of_s(z))
        # Производные по v
        dx_dv = -self.a * np.cos(self.u_fixed) * np.sin(v)
        dy_dv = -self.a * np.sin(self.u_fixed) * np.sin(v)
        dz_dv = self.c * np.cos(v)
        dR_dv = np.array([dx_dv, dy_dv, dz_dv])
        # ds/dv
        ds_dv = np.sqrt(self.a**2 * np.sin(v)**2 + self.c**2 * np.cos(v)**2)
        # Единичный касательный
        return dR_dv / ds_dv

    @property
    def domain(self):
        return (0.0, self.total_length)

# ----------------------------------------------------------------------
# 2. Параметры поверхностей
# ----------------------------------------------------------------------
a1, b1, c1 = 2.0, 2.0, 4.0   # эллипсоид с большой осью Z
E1 = EllipsoidWithDerivatives(a1, b1, c1)
radius = 0.8
E2 = CylinderWithDerivatives(radius)

# ----------------------------------------------------------------------
# 3. Аналитическая траектория меридиана
# ----------------------------------------------------------------------
u_fixed = 0.0
v_max = np.arccos(radius / a1)
v_start = -v_max
v_end   =  v_max

traj = AnalyticMeridianTrajectory(a1, c1, u_fixed, (v_start, v_end))
print(f"Аналитическая траектория меридиана, длина: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 4. Начальная точка на цилиндре (аналитически)
# ----------------------------------------------------------------------
R0 = traj.R(0.0)
v0 = R0[2]   # z-координата
print(f"Начальная высота на цилиндре: {v0:.4f}")

# ----------------------------------------------------------------------
# 5. Настройка RHS с аналитической траекторией
# ----------------------------------------------------------------------
# Поскольку траектория теперь точная, k можно оставить 0
rhs_calc = RightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=0.0,           # коррекция не нужна
    max_ds_dz=0.5
)

# ----------------------------------------------------------------------
# 6. Интегратор и строитель
# ----------------------------------------------------------------------
solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
builder = WindingLineBuilder(E2, traj, rhs_calc, solver)

z_eval = np.linspace(0, traj.total_length, 400)
z_vals, line_3d = builder.compute(u_fixed, v0, z_eval=z_eval)
uv_states = builder.get_uv_states()
print(f"Рассчитано {len(z_vals)} точек.")

# ----------------------------------------------------------------------
# 7. Диагностика
# ----------------------------------------------------------------------
print("\nПервые 5 точек (u, v):")
for i in range(5):
    u_i, v_i = uv_states[i]
    print(f"z={z_vals[i]:.3f}: u={u_i:.6f}, v={v_i:.6f}")

u_vals = uv_states[:, 0]
max_dev = np.max(np.abs(u_vals - u_fixed))
print(f"Максимальное отклонение u от {u_fixed}: {max_dev:.2e}")

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
# 8. Визуализация
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

# Цилиндр (вокруг всей высоты траектории)
z_min = min(line_3d[:,2]) - 0.5
z_max = max(line_3d[:,2]) + 0.5
v_cyl = np.linspace(z_min, z_max, 40)
Uc, Vc = np.meshgrid(u_grid, v_cyl)
X2, Y2, Z2 = surface_grid(E2, Uc, Vc)

# Точки траектории для визуализации
vis_z = np.linspace(0, traj.total_length, 500)
vis_points = np.array([traj.R(z) for z in vis_z])

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='E2'))
fig.add_trace(go.Scatter3d(x=vis_points[:,0], y=vis_points[:,1], z=vis_points[:,2],
                           mode='lines', line=dict(color='red', width=4), name='R(z)'))
fig.add_trace(go.Scatter3d(x=line_3d[:,0], y=line_3d[:,1], z=line_3d[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='Линия укладки'))

# Соединительные отрезки (каждый 30-й)
step = 30
for i in range(0, len(z_vals), step):
    Rz = traj.R(z_vals[i])
    Pz = line_3d[i]
    fig.add_trace(go.Scatter3d(x=[Rz[0], Pz[0]], y=[Rz[1], Pz[1]], z=[Rz[2], Pz[2]],
                               mode='lines', line=dict(color='gray', width=1, dash='dot'), showlegend=False))

fig.update_layout(title='Восстановление линии укладки (аналитическая траектория)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)
fig.write_html('winding_analytic.html')
print("График сохранён в winding_analytic.html")