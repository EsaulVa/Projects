# restore_winding_from_tsn_fixed.py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from solvers.scipy_solver import SciPySolver
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor, RayTracer
from helpers.intersection import RevolutionIntersection,RobustRevolutionIntersection

# ============================================================================
# 1. Загрузка данных
# ============================================================================
data = scipy.io.loadmat('refined_kinematics.mat')
s_array = data['s'].flatten()
coords = data['coords']
z_offset = float(data['z_offset'].flatten()[0]) if 'z_offset' in data else 0.0
tsn_pts = data.get('tsn_pts', None)
original_lu = data.get('mandrel_pts', None)

if tsn_pts is None:
    from machine.machine3axis_exact import Machine3AxisExact_ODE, MachineState
    machine = Machine3AxisExact_ODE(ring_radius=50.0, d_offset=100.0)
    tsn_pts = np.zeros((len(s_array), 3))
    for i, q in enumerate(coords):
        state = MachineState(q)
        tsn_pts[i] = machine.forward(state)['point']

# ============================================================================
# 2. Поверхность оправки
# ============================================================================
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705
# Создаём безопасную версию поверхности
class SafeSurface(PiecewisePolynomialRevolution):
    def _get_segment(self, u):
        u = np.clip(u, self.a, self.d)
        if self.a <= u <= self.b:
            return 1
        elif self.b < u < self.c:
            return 2
        elif self.c <= u <= self.d:
            return 3
        else:
            return 2

# Затем используем SafeSurface вместо исходного класса
E2_opravka = SafeSurface(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)
# E2_opravka = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

# Добавляем атрибуты, которые использует inverse_winding_hybrid для проверки границ
E2_opravka.v_min = -np.pi
E2_opravka.v_max = np.pi
E2_opravka.u_min = bound_opravka[0]
E2_opravka.u_max = bound_opravka[-1]

# Приводим ТСН к локальной системе
tsn_pts_local = tsn_pts.copy()
tsn_pts_local[:, 2] -= z_offset
traj = Trajectory.from_points(tsn_pts_local, method='cubic')

# Начальные параметры
if original_lu is not None:
    first_lu_local = original_lu[0].copy()
    first_lu_local[2] -= z_offset
    u0 = np.clip(first_lu_local[2], bound_opravka[0], bound_opravka[-1])
    v0 = np.arctan2(first_lu_local[1], first_lu_local[0])
else:
    u0 = np.clip(tsn_pts_local[0, 2], bound_opravka[0], bound_opravka[-1])
    v0 = np.arctan2(tsn_pts_local[0, 1], tsn_pts_local[0, 0])

# ============================================================================
# 3. Обратная задача намотки (гибридный метод)
# ============================================================================
# Настройка предикторов
solver_dae = SciPySolver(method='RK45', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)

ray_tracer = RayTracer()
# ray_tracer.register(PiecewisePolynomialRevolution, RevolutionIntersection())
ray_tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
optical_predictor = OpticalPredictor(ray_tracer)
# result = inverse_winding_hybrid(
#     surface=E2_opravka,
#     traj=traj,
#     u0=u0,
#     v0=v0,
#     count_points=len(s_array),
#     eps_Phi=1e-12,
#     max_newton=8,
#     max_bisect=5,
#     jump_threshold=3.0,
#     eps_kappa=1e-4,
#     u_margin=0.01,
#     force_optical_after_fail=True
# )
result = inverse_winding_hybrid(
    E2_opravka, traj, u0, v0,
    count_points=len(s_array),
    eps_Phi=1e-10, max_newton=2, max_bisect=10, jump_threshold=3.0,
    predictor_dae=dae_predictor,
    predictor_optical=optical_predictor,
    eps_kappa=1e-2,
    u_margin=0.05,
    force_optical_after_fail=True
)

recovered_lu_local = result['points_3d']
recovered_lu_global = recovered_lu_local.copy()
recovered_lu_global[:, 2] += z_offset

# ============================================================================
# 4. Визуализация
# ============================================================================
fig = go.Figure()
# Оправка
u_opr = np.linspace(0, 768.54, 40)
v_opr = np.linspace(0, 2*np.pi, 30)
Uo, Vo = np.meshgrid(u_opr, v_opr)
Xo, Yo = np.zeros_like(Uo), np.zeros_like(Uo)
Zo = Uo.copy()
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = E2_opravka.position(Uo[i,j], Vo[i,j])
        Xo[i,j], Yo[i,j] = p[0], p[1]
fig.add_trace(go.Surface(x=Xo, y=Yo, z=Zo + z_offset, opacity=0.4, colorscale='Blues', showscale=False, name='Оправка'))
# ТСН
fig.add_trace(go.Scatter3d(x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
                           mode='lines', line=dict(color='red', width=3), name='ТСН (скорректированная)'))
# Исходная линия укладки
if original_lu is not None:
    fig.add_trace(go.Scatter3d(x=original_lu[:,0], y=original_lu[:,1], z=original_lu[:,2],
                               mode='lines', line=dict(color='green', width=3), name='Линия укладки (исходная)'))
# Восстановленная линия
fig.add_trace(go.Scatter3d(x=recovered_lu_global[:,0], y=recovered_lu_global[:,1], z=recovered_lu_global[:,2],
                           mode='lines', line=dict(color='orange', width=3, dash='dot'), name='Линия укладки (восстановленная)'))
fig.update_layout(title='Восстановление линии укладки (гибридный метод)', scene=dict(aspectmode='data'))
fig.write_html("restored_winding.html")
print("3D сцена сохранена в restored_winding.html")

# Ошибка
if original_lu is not None:
    error = np.linalg.norm(original_lu - recovered_lu_global, axis=1)
    print(f"Средняя ошибка восстановления: {np.mean(error):.3e} мм, макс: {np.max(error):.3e} мм")
    plt.figure()
    plt.plot(s_array, error, 'b-')
    plt.xlabel('s, мм')
    plt.ylabel('Ошибка восстановления, мм')
    plt.title('Отклонение восстановленной линии укладки от исходной')
    plt.grid(True)
    plt.savefig('restoration_error.png')
    plt.show()

scipy.io.savemat('recovered_lu.mat', {'s': s_array, 'lu_points': recovered_lu_global})
print("Восстановленная линия укладки сохранена в recovered_lu.mat")