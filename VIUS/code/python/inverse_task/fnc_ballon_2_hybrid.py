import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
import scipy.io as sio

from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method import newton_corrector
from geometry.tsurfaces import FixedPointTrajectory

# Гибридный решатель и компоненты предикторов
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor
from helpers.optical_predictor import RayTracer
from helpers.intersection import CylinderIntersection, SphereIntersection

# ----------------------------------------------------------------------
# 1. Внешний баллон E1 (поверхность безопасности)
# ----------------------------------------------------------------------
R1, L1 = 6.0, 7.0
z1_min, z1_max = -L1/2, L1/2
cyl1 = CylinderSegment(R1, z1_min, z1_max)
E1 = CompositeSurface([SphereSegment(R1, z1_min, is_upper=False),
                       cyl1,
                       SphereSegment(R1, z1_max, is_upper=True)])

# ----------------------------------------------------------------------
# 2. Прямая задача на E1 – траектория точки схода
# ----------------------------------------------------------------------
dev_law = ConstantDeviation(tan_theta=0)
fwd_builder = ForwardWindingBuilder(
    surface=E1, deviation_law=dev_law,
    solver=SciPySolver(method='BDF', rtol=1e-8, atol=1e-10),
    normalize_tangent=True, eps=1e-12
)

u0_ext = -6
v0_ext = 0.01
alpha = 70*np.pi / 180
s_end = 50.0
s_eval = np.linspace(0, s_end, 200)

print("Прямая задача на внешнем баллоне…")
s_vals, line_E1 = fwd_builder.build(
    initial_point=(u0_ext, v0_ext),
    initial_tangent=(alpha,),
    eval_points=s_eval
)
if not fwd_builder.last_run_successful:
    raise RuntimeError("Прямая задача не удалась")

traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 3. Внутренний баллон E2 (оправка)
# ----------------------------------------------------------------------
R2, L2 = 4.0, 8.0
z2_min, z2_max = -L2/2, L2/2
cyl2 = CylinderSegment(R2, z2_min, z2_max)
E2 = CompositeSurface([SphereSegment(R2, z2_min, is_upper=False),
                       cyl2,
                       SphereSegment(R2, z2_max, is_upper=True)])

# ----------------------------------------------------------------------
# 4. Начальная точка на оправке (аналитическое приближение + корректор)
# ----------------------------------------------------------------------
R0 = traj.R(0.0)
R_ext, R_int = R1, R2

if abs(v0_ext) <= L2/2:
    theta = np.arccos(R_int / R_ext)
    u_guess = theta if R0[0] >= 0 else -theta
    v_guess = v0_ext
else:
    u_guess = u0_ext
    v_guess = np.clip(R0[2], E2.v_min, E2.v_max)

print(f"Начальное приближение: u={u_guess:.4f}, v={v_guess:.4f}")

if hasattr(E2, 'project_point'):
    u0_int, v0_int, Phi0, conv = E2.project_point(R0, u_guess, v_guess,
                                                   eps_Phi=1e-12, max_iter=20)
else:
    dummy = FixedPointTrajectory(R0)
    u0_int, v0_int, Phi0, n, conv = newton_corrector(E2, dummy, u_guess, v_guess, 0.0,
                                                     eps_Phi=1e-12, max_iter=20)
print(f"После коррекции: u={u0_int:.4f}, v={v0_int:.4f}, Φ={Phi0:.2e}, сходимость={conv}")

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
# 6. Гибридная обратная задача
# ----------------------------------------------------------------------
result = inverse_winding_hybrid(
    E2, traj, u0_int, v0_int,
    count_points=300,
    eps_Phi=1e-10, max_newton=7, max_bisect=4, jump_threshold=3.0,
    predictor_dae=dae_predictor,
    predictor_optical=optical_predictor,
    eps_kappa=1e-4,
    u_margin=0.05,
    force_optical_after_fail=True
)

z_vals = result['z_eval']
line_E2 = result['points_3d']
Phi_hist = result['Phi']
print(f"Максимальная невязка |Φ| = {np.max(np.abs(Phi_hist)):.2e}")

# ----------------------------------------------------------------------
# 7. Визуализация (как в исходном скрипте)
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
v_grid_E1 = np.linspace(E1.v_min, E1.v_max, 80)
v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 60)

X1, Y1, Z1 = surface_grid(E1, u_grid, v_grid_E1)
X2, Y2, Z2 = surface_grid(E2, u_grid, v_grid_E2)

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='Внешний баллон'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds', showscale=False, name='Внутренний баллон (оправка)'))

# Сглаженная траектория R(z)
dist = np.zeros(len(line_E1))
dist[1:] = np.linalg.norm(np.diff(line_E1, axis=0), axis=1).cumsum()
cs = CubicSpline(dist, line_E1, axis=0)
dense_dist = np.linspace(dist[0], dist[-1], len(line_E1)*5)
smooth_R = cs(dense_dist)
fig.add_trace(go.Scatter3d(x=smooth_R[:,0], y=smooth_R[:,1], z=smooth_R[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='Траектория R(z)'))

# Сглаженная линия укладки на E2
dist2 = np.zeros(len(line_E2))
dist2[1:] = np.linalg.norm(np.diff(line_E2, axis=0), axis=1).cumsum()
cs2 = CubicSpline(dist2, line_E2, axis=0)
dense_dist2 = np.linspace(dist2[0], dist2[-1], len(line_E2)*5)
smooth_E2 = cs2(dense_dist2)
fig.add_trace(go.Scatter3d(x=smooth_E2[:,0], y=smooth_E2[:,1], z=smooth_E2[:,2],
                           mode='lines', line=dict(color='red', width=4), name='Линия укладки на E2'))

# Эталонная линия (если есть)
r_etalon = None
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(x=r_etalon[:,0], y=r_etalon[:,1], z=r_etalon[:,2],
                               mode='lines', line=dict(color='orange', width=3, dash='dot'),
                               name='Эталонная ЛУ'))

# Соединительные отрезки
step = 20
for i in range(0, len(z_vals), step):
    R_pt = traj.R(z_vals[i])
    r_pt = line_E2[i]
    fig.add_trace(go.Scatter3d(x=[R_pt[0], r_pt[0]], y=[R_pt[1], r_pt[1]], z=[R_pt[2], r_pt[2]],
                               mode='lines', line=dict(color='green', width=1, dash='dot'),
                               showlegend=False))

# Начальные и конечные точки
fig.add_trace(go.Scatter3d(x=[line_E1[0,0]], y=[line_E1[0,1]], z=[line_E1[0,2]],
                           mode='markers', marker=dict(color='black', size=6, symbol='circle'),
                           name='Старт R(z)'))
fig.add_trace(go.Scatter3d(x=[line_E1[-1,0]], y=[line_E1[-1,1]], z=[line_E1[-1,2]],
                           mode='markers', marker=dict(color='black', size=6, symbol='x'),
                           name='Конец R(z)'))
fig.add_trace(go.Scatter3d(x=[line_E2[0,0]], y=[line_E2[0,1]], z=[line_E2[0,2]],
                           mode='markers', marker=dict(color='green', size=8, symbol='diamond'),
                           name='Старт укладки'))
fig.add_trace(go.Scatter3d(x=[line_E2[-1,0]], y=[line_E2[-1,1]], z=[line_E2[-1,2]],
                           mode='markers', marker=dict(color='orange', size=8, symbol='cross'),
                           name='Финиш укладки'))

fig.update_layout(title='Обратная задача: внешний баллон → внутренняя оправка (гибридный решатель)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1200, height=800)
fig.write_html('outer_traj_inner_winding_hybrid.html')
print("График сохранён в outer_traj_inner_winding_hybrid.html")