import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method import newton_corrector
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor, RayTracer
from helpers.intersection import CylinderIntersection, SphereIntersection
from geometry.tsurfaces import FixedPointTrajectory

# ----------------------------------------------------------------------
# 1. Внутренний баллон (оправка) и внешний (поверхность безопасности)
# ----------------------------------------------------------------------
R_int, L_int = 2.0, 6.0
z_min_int, z_max_int = -L_int/2, L_int/2
cyl_int = CylinderSegment(R_int, z_min_int, z_max_int)
E_int = CompositeSurface([SphereSegment(R_int, z_min_int, is_upper=False),
                         cyl_int,
                         SphereSegment(R_int, z_max_int, is_upper=True)])

R_ext, L_ext = 4.0, 12.0
z_min_ext, z_max_ext = -L_ext/2, L_ext/2
cyl_ext = CylinderSegment(R_ext, z_min_ext, z_max_ext)
E_ext = CompositeSurface([SphereSegment(R_ext, z_min_ext, is_upper=False),
                         cyl_ext,
                         SphereSegment(R_ext, z_max_ext, is_upper=True)])

# ----------------------------------------------------------------------
# 2. Прямая задача на внутреннем баллоне — траектория R(z)
# ----------------------------------------------------------------------
dev_law = ConstantDeviation(tan_theta=0.1)
solver_fwd = SciPySolver(method='BDF', rtol=1e-8, atol=1e-10)
fwd_builder = ForwardWindingBuilder(
    surface=E_int, deviation_law=dev_law,
    solver=solver_fwd, normalize_tangent=True, eps=1e-12
)

u0_int = 0.0
v0_int = E_int.v_min + 0.2   # нижнее днище
alpha = np.pi / 6
s_end = 25.0
s_eval = np.linspace(0, s_end, 200)

print("Прямая задача на внутреннем баллоне...")
s_vals, line_int = fwd_builder.build(
    initial_point=(u0_int, v0_int),
    initial_tangent=(alpha,),
    eval_points=s_eval
)
if not fwd_builder.last_run_successful:
    raise RuntimeError("Прямая задача не завершена")

traj = Trajectory.from_points(line_int, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 3. Начальная точка на ВНЕШНЕМ баллоне – ТОЛЬКО НА ДНИЩЕ
# ----------------------------------------------------------------------
R0 = traj.R(0.0)

# Пробуем найти проекцию на нижнюю полусферу внешнего баллона
u_guess = u0_int
v_guess = E_ext.v_min + 0.5   # нижняя полусфера (далеко от цилиндра)

if hasattr(E_ext, 'project_point'):
    u0_ext, v0_ext, Phi0, conv = E_ext.project_point(R0, u_guess, v_guess,
                                                     eps_Phi=1e-12, max_iter=20)
else:
    dummy = FixedPointTrajectory(R0)
    u0_ext, v0_ext, Phi0, _, conv = newton_corrector(
        E_ext, dummy, u_guess, v_guess, 0.0, eps_Phi=1e-12, max_iter=20)
print(f"Начальная точка на внешнем баллоне: u={u0_ext:.4f}, v={v0_ext:.4f}, Φ={Phi0:.2e}")

# Доводка через траекторию (необязательно, но для уверенности)
if abs(Phi0) > 1e-8:
    u0_ext, v0_ext, Phi0, _, conv = newton_corrector(
        E_ext, traj, u0_ext, v0_ext, 0.0, eps_Phi=1e-12, max_iter=20)

# ----------------------------------------------------------------------
# 4. ПРОВЕРКА ДОСТИЖИМОСТИ
# ----------------------------------------------------------------------
if abs(Phi0) > 1e-6:
    print("=" * 60)
    print("ВНИМАНИЕ: Не удалось найти точку касания на внешнем баллоне.")
    print(f"Текущая невязка Φ = {Phi0:.2e}")
    print("Возможные причины:")
    print("  - начальная точка траектории слишком близка к цилиндрической части,")
    print("    где луч изнутри не может коснуться внешней поверхности;")
    print("  - необходимо выбрать начальную точку на сферическом днище внешнего баллона")
    print("    (например, v_guess = E_ext.v_min + 0.5) и проверить сходимость.")
    print("=" * 60)
    import sys
    sys.exit(1)

# ----------------------------------------------------------------------
# 5. Настройка предикторов
# ----------------------------------------------------------------------
solver_dae = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)

ray_tracer = RayTracer()
ray_tracer.register(CylinderSegment, CylinderIntersection())
ray_tracer.register(SphereSegment, SphereIntersection())
optical_predictor = OpticalPredictor(ray_tracer)

# ----------------------------------------------------------------------
# 6. Гибридная обратная задача (на внешнем баллоне)
# ----------------------------------------------------------------------
result = inverse_winding_hybrid(
    E_ext, traj, u0_ext, v0_ext,
    count_points=300,
    eps_Phi=1e-10, max_newton=7, max_bisect=4, jump_threshold=3.0,
    predictor_dae=dae_predictor,
    predictor_optical=optical_predictor,
    eps_kappa=1e-4,
    u_margin=0.05,
    force_optical_after_fail=True
)

z_vals = result['z_eval']
line_ext = result['points_3d']
Phi_hist = result['Phi']
print(f"Максимальная невязка |Φ| = {np.max(np.abs(Phi_hist)):.2e}")

# ----------------------------------------------------------------------
# 7. Визуализация
# ----------------------------------------------------------------------
def surface_grid(surf, u_arr, v_arr):
    U, V = np.meshgrid(u_arr, v_arr)
    X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surf.position(U[i,j], V[i,j])
            X[i,j], Y[i,j], Z[i,j] = np.array(p)
    return X, Y, Z

u_grid = np.linspace(0, 2*np.pi, 80)
v_grid_int = np.linspace(E_int.v_min, E_int.v_max, 60)
v_grid_ext = np.linspace(E_ext.v_min, E_ext.v_max, 80)

X_int, Y_int, Z_int = surface_grid(E_int, u_grid, v_grid_int)
X_ext, Y_ext, Z_ext = surface_grid(E_ext, u_grid, v_grid_ext)

fig = go.Figure()
fig.add_trace(go.Surface(x=X_int, y=Y_int, z=Z_int, opacity=0.4, colorscale='Reds', showscale=False,
                         name='Внутренний баллон (траектория)'))
fig.add_trace(go.Surface(x=X_ext, y=Y_ext, z=Z_ext, opacity=0.2, colorscale='Blues', showscale=False,
                         name='Внешний баллон (укладка)'))

# Траектория R(z)
dist = np.zeros(len(line_int))
dist[1:] = np.linalg.norm(np.diff(line_int, axis=0), axis=1).cumsum()
cs = CubicSpline(dist, line_int, axis=0)
dense_dist = np.linspace(dist[0], dist[-1], len(line_int)*5)
smooth_R = cs(dense_dist)
fig.add_trace(go.Scatter3d(x=smooth_R[:,0], y=smooth_R[:,1], z=smooth_R[:,2],
                           mode='lines', line=dict(color='red', width=4), name='Траектория R(z)'))

# Линия укладки на внешнем баллоне
dist2 = np.zeros(len(line_ext))
dist2[1:] = np.linalg.norm(np.diff(line_ext, axis=0), axis=1).cumsum()
cs2 = CubicSpline(dist2, line_ext, axis=0)
dense_dist2 = np.linspace(dist2[0], dist2[-1], len(line_ext)*5)
smooth_E2 = cs2(dense_dist2)
fig.add_trace(go.Scatter3d(x=smooth_E2[:,0], y=smooth_E2[:,1], z=smooth_E2[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='Линия укладки на внешнем баллоне'))

# Соединительные отрезки (каждый 20-й)
step = 20
for i in range(0, len(z_vals), step):
    R_pt = traj.R(z_vals[i])
    r_pt = line_ext[i]
    fig.add_trace(go.Scatter3d(x=[R_pt[0], r_pt[0]], y=[R_pt[1], r_pt[1]], z=[R_pt[2], r_pt[2]],
                               mode='lines', line=dict(color='green', width=1, dash='dot'),
                               showlegend=False))

# Начальные и конечные точки
fig.add_trace(go.Scatter3d(x=[line_int[0,0]], y=[line_int[0,1]], z=[line_int[0,2]],
                           mode='markers', marker=dict(color='black', size=6, symbol='circle'),
                           name='Старт R(z)'))
fig.add_trace(go.Scatter3d(x=[line_int[-1,0]], y=[line_int[-1,1]], z=[line_int[-1,2]],
                           mode='markers', marker=dict(color='black', size=6, symbol='x'),
                           name='Конец R(z)'))
fig.add_trace(go.Scatter3d(x=[line_ext[0,0]], y=[line_ext[0,1]], z=[line_ext[0,2]],
                           mode='markers', marker=dict(color='green', size=8, symbol='diamond'),
                           name='Старт укладки'))
fig.add_trace(go.Scatter3d(x=[line_ext[-1,0]], y=[line_ext[-1,1]], z=[line_ext[-1,2]],
                           mode='markers', marker=dict(color='orange', size=8, symbol='cross'),
                           name='Финиш укладки'))

fig.update_layout(title='Обратная задача: внутр. траектория → внешняя укладка (гибрид, днище)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1200, height=800)
fig.write_html('inner_traj_outer_winding_hybrid_fixed.html')
print("График сохранён в inner_traj_outer_winding_hybrid_fixed.html")