import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from geometry.ellipsoid import EllipsoidWithDerivatives
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver

# ----------------------------------------------------------------------
# 1. Параметры эллипсоидов и прямая задача
# ----------------------------------------------------------------------
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)
scale = 0.8
a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)

print("===== Прямая задача =====")
deviation_law = ConstantDeviation(tan_theta=0.1)
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
forward_builder = ForwardWindingBuilder(
    surface=E1, deviation_law=deviation_law,
    solver=solver_forward, normalize_tangent=True, eps=1e-12
)
u0, v0 = 0.2, 0.02
alpha = np.pi / 6
s_end = 30.0
count_points = 100
s_eval = np.linspace(0, s_end, count_points)
s_vals, line_E1 = forward_builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
    eval_points=s_eval
)
if not forward_builder.last_run_successful:
    raise RuntimeError("Прямая задача завершилась с ошибкой")
print(f"Прямая задача: построено {len(s_vals)} точек")

# Траектория точки схода
# traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
traj = Trajectory.from_points(line_E1, method='nurbs',degree=5)
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 2. Вспомогательные функции
# ----------------------------------------------------------------------
def compute_tangent_components(surface, u, v, tau_3d):
    """Контравариантные компоненты (u', v') 3D-касательной tau_3d."""
    geom = surface.derivatives(u, v)
    ru, rv = geom['ru'], geom['rv']
    E, F, G = surface.first_fundamental_form(u, v)
    det = E*G - F*F
    if abs(det) < 1e-12:
        raise ValueError("Singular metric")
    b1 = np.dot(tau_3d, ru)
    b2 = np.dot(tau_3d, rv)
    u_prime = (G * b1 - F * b2) / det
    v_prime = (-F * b1 + E * b2) / det
    return u_prime, v_prime

def normal_curvature(surface, u, v, u_prime, v_prime):
    """κ_n = II(τ, τ)."""
    L, M, N = surface.second_fundamental_form(u, v)
    return L * u_prime**2 + 2 * M * u_prime * v_prime + N * v_prime**2

def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
    """
    Компоненты ∂Φ/∂u, ∂Φ/∂v.
    Формула: ∂Φ/∂u^α = -λ * b_α^β * τ_β.
    """
    E, F, G = surface.first_fundamental_form(u, v)
    L, M, N = surface.second_fundamental_form(u, v)
    det = E*G - F*F
    if abs(det) < 1e-12:
        raise ValueError("Singular metric in grad_Phi")
    guu = G / det
    guv = -F / det
    gvv = E / det

    # ковариантные компоненты τ
    tau_u = E * u_prime + F * v_prime
    tau_v = F * u_prime + G * v_prime

    # смешанные компоненты b_α^β = g^{βγ} b_{αγ}
    b_u_u = guu * L + guv * M
    b_u_v = guv * L + gvv * M
    b_v_u = guu * M + guv * N
    b_v_v = guv * M + gvv * N

    dPhidu = -lam * (b_u_u * tau_u + b_u_v * tau_v)
    dPhidv = -lam * (b_v_u * tau_u + b_v_v * tau_v)
    return dPhidu, dPhidv

# ----------------------------------------------------------------------
# 3. Алгоритм обратного интегрирования (FNC-корректор)
# ----------------------------------------------------------------------
print("\n===== Обратная задача (FNC-корректор) =====")

z_eval = np.linspace(0, traj.total_length, count_points)
N = len(z_eval)
u_hist = np.zeros(N)
v_hist = np.zeros(N)
Phi_hist = np.zeros(N)

u_cur, v_cur = u0, v0
u_hist[0], v_hist[0] = u0, v0
eps_Phi = 1e-8
eps_grad = 1e-12

for i in range(N - 1):
    z_k = z_eval[i]
    z_next = z_eval[i + 1]
    dz = z_next - z_k

    r_k = E2.position(u_cur, v_cur)
    m_k = E2.normal(u_cur, v_cur)
    R_k = traj.R(z_k)
    delta_k = R_k - r_k                     # Вектор от оправки к R
    lam_k = np.linalg.norm(delta_k)
    
    if lam_k < 1e-12:
        u_hist[i+1], v_hist[i+1], Phi_hist[i+1] = u_cur, v_cur, 0.0
        continue

    # ИСПРАВЛЕНИЕ 2: Правильный знак касательной (от R к оправке)
    # tau_3d = -delta_k / lam_k 
    # u_prime, v_prime = compute_tangent_components(E2, u_cur, v_cur, tau_3d)
        # Сырой вектор направления нити (в 3D)
    tau_3d_raw = -delta_k / lam_k 
    
    # !!! ФИКС: Ортогональная проекция на касательную плоскость оправки !!!
    # Формула: tau_proj = tau_raw - (tau_raw . m) * m
    tau_3d_surf = tau_3d_raw - np.dot(tau_3d_raw, m_k) * m_k
    norm_surf = np.linalg.norm(tau_3d_surf)
    
    if norm_surf < 1e-12:
        # Нить уперлась в оправку перпендикулярно (направление не определено)
        # В реальном станке это обрыв нити. Пропускаем шаг или бросаем исключение.
        u_hist[i+1], v_hist[i+1], Phi_hist[i+1] = u_cur, v_cur, 0.0
        continue

    # Нормализованная касательная, лежащая НА ПОВЕРХНОСТИ
    tau_3d = tau_3d_surf / norm_surf
    
    # Теперь считаем компоненты для ПРАВИЛЬНОЙ касательной
    u_prime, v_prime = compute_tangent_components(E2, u_cur, v_cur, tau_3d)

    kappa_n = normal_curvature(E2, u_cur, v_cur, u_prime, v_prime)
    if abs(kappa_n) < 1e-12:
        print(f"Предупреждение: κ_n ≈ 0 на шаге {i}, z={z_k:.3f}")
        # В реальном CAM здесь нужно уменьшать dz или вызывать оптический корректор
        u_hist[i+1], v_hist[i+1], Phi_hist[i+1] = u_cur, v_cur, 0.0
        continue

    # Предиктор Савина
    R_prime_k = traj.R_deriv(z_k)
    v_speed = np.dot(R_prime_k, m_k) / (lam_k * kappa_n)
    
    u_pred = u_cur + v_speed * dz * u_prime
    v_pred = v_cur + v_speed * dz * v_prime

    r_pred = E2.position(u_pred, v_pred)
    m_pred = E2.normal(u_pred, v_pred)

    # Невязка (3.16)
    R_next = traj.R(z_next)
    Phi = np.dot(R_next - r_pred, m_pred)
    
    # --- БЛОК ДИАГНОСТИКИ FNC (опционально, но полезно) ---
    # В реальной задаче сюда нужно подставить репер Ферми-Уокера
    # Пока оставляем заглушку, что фрейм совпадает с глобальным (только для логики)
    # xi = np.dot(R_next - r_pred, E1_frame) # и т.д.
    # -------------------------------------------------------

    if abs(Phi) > eps_Phi:
        # ИСПРАВЛЕНИЕ 1: Пересчет геометрии В ТОЧКЕ ПРЕДИКТОРА
        delta_pred = R_next - r_pred
        lam_pred = np.linalg.norm(delta_pred)
        if lam_pred < 1e-12:
             u_hist[i+1], v_hist[i+1], Phi_hist[i+1] = u_pred, v_pred, Phi
             continue
             
        tau_pred = -delta_pred / lam_pred # Правильный знак!
        u_prime_pred, v_prime_pred = compute_tangent_components(E2, u_pred, v_pred, tau_pred)

        try:
            # Передаем ПРЕДИКТОРНЫЕ значения!
            dPhidu, dPhidv = compute_grad_Phi(
                E2, u_pred, v_pred, u_prime_pred, v_prime_pred, lam_pred
            )
        except ValueError:
            print(f"Ошибка вырождения метрики на шаге {i}")
            break

        E, F, G = E2.first_fundamental_form(u_pred, v_pred)
        det = E*G - F*F
        guu = G / det
        guv = -F / det
        gvv = E / det
        
        # Квадрат нормы градиента
        N_grad = guu * dPhidu**2 + 2 * guv * dPhidu * dPhidv + gvv * dPhidv**2

        if N_grad < eps_grad:
            # Вырождение (каустика или цилиндр). 
            # Лучшее решение здесь - вызов оптического корректора (Алгоритм 2).
            # Пока просто принимаем предиктор как есть (будет дрейфовать).
            u_cur, v_cur = u_pred, v_pred
        else:
            # Шаг Ньютона
            du_corr = -Phi / N_grad * (guu * dPhidu + guv * dPhidv)
            dv_corr = -Phi / N_grad * (guv * dPhidu + gvv * dPhidv)
            u_cur = u_pred + du_corr
            v_cur = v_pred + dv_corr
            
            # Перезаписываем Phi для лога после успешной коррекции
            r_corr = E2.position(u_cur, v_cur)
            m_corr = E2.normal(u_cur, v_cur)
            Phi = np.dot(R_next - r_corr, m_corr) 
    else:
        u_cur, v_cur = u_pred, v_pred

    u_hist[i+1] = u_cur
    v_hist[i+1] = v_cur
    Phi_hist[i+1] = Phi

# 3D-точки восстановленной линии
points_E2 = np.array([E2.position(u_hist[i], v_hist[i]) for i in range(N)])
line_E2 = points_E2

# ----------------------------------------------------------------------
# 4. Диагностика
# ----------------------------------------------------------------------
print("Максимальная невязка |Φ|:", np.max(np.abs(Phi_hist)))
plt.figure(figsize=(10, 4))
plt.plot(z_eval, Phi_hist, 'b-', linewidth=1.5)
plt.axhline(y=0, color='gray', linestyle='--')
plt.xlabel('z')
plt.ylabel('Φ')
plt.title('Невязка условия тени (FNC-корректор)')
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# 5. Визуализация
# ----------------------------------------------------------------------
u_grid = np.linspace(0, 2*np.pi, 80)
v_grid = np.linspace(-np.pi/2, np.pi/2, 50)
U, V = np.meshgrid(u_grid, v_grid)

X1 = np.zeros_like(U)
Y1 = np.zeros_like(U)
Z1 = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = E1.position(U[i,j], V[i,j])
        X1[i,j], Y1[i,j], Z1[i,j] = p

X2 = np.zeros_like(U)
Y2 = np.zeros_like(U)
Z2 = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = E2.position(U[i,j], V[i,j])
        X2[i,j], Y2[i,j], Z2[i,j] = p

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2,
                         colorscale='Blues', showscale=False,
                         name='E1 (внешний)'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3,
                         colorscale='Reds', showscale=False,
                         name='E2 (внутренний)'))
fig.add_trace(go.Scatter3d(x=line_E1[:,0], y=line_E1[:,1], z=line_E1[:,2],
                           mode='lines', line=dict(color='blue', width=5),
                           name='Исходная линия (траектория)'))
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                           mode='lines', line=dict(color='red', width=5),
                           name='Восстановленная линия'))

step = max(1, len(z_eval) // 20)
for i in range(0, len(z_eval), step):
    fig.add_trace(go.Scatter3d(x=[line_E1[i,0], line_E2[i,0]],
                               y=[line_E1[i,1], line_E2[i,1]],
                               z=[line_E1[i,2], line_E2[i,2]],
                               mode='lines', line=dict(color='green', width=2, dash='solid'),
                               showlegend=False))
fig.add_trace(go.Scatter3d(x=[line_E1[0,0], line_E2[0,0]],
                           y=[line_E1[0,1], line_E2[0,1]],
                           z=[line_E1[0,2], line_E2[0,2]],
                           mode='markers',
                           marker=dict(color='black', size=5),
                           name='Начальные точки'))
fig.update_layout(
    title='Прямая + обратная задача (FNC-корректор)',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
               aspectmode='data'),
    width=1000, height=800, hovermode='closest'
)
fig.write_html('winding_plot.html')
print("График сохранён в winding_plot.html")