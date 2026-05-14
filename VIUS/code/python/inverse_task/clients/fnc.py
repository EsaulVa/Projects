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
deviation_law = ConstantDeviation(tan_theta=0)
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
forward_builder = ForwardWindingBuilder(
    surface=E1, deviation_law=deviation_law,
    solver=solver_forward, normalize_tangent=True, eps=1e-12
)
u0, v0 = 0.1, 0.0
alpha = np.pi / 6
s_end = 30.0
count_points = 50
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
traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")
# ----------------------------------------------------------------------
# 2. Вспомогательные функции (исправленные)
# ----------------------------------------------------------------------

def compute_tangent_components(surface, u, v, tau_3d):
    """
    Контравариантные компоненты (u', v') проекции 3D-вектора tau_3d
    на касательную плоскость к surface в точке (u, v).

    Решает систему:
        tau_3d ≈ u' * r_u + v' * r_v   (проекция)
    через первую фундаментальную форму.
    """
    geom = surface.derivatives(u, v)
    ru, rv = geom['ru'], geom['rv']
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика поверхности")
    b1 = np.dot(tau_3d, ru)
    b2 = np.dot(tau_3d, rv)
    u_prime = (G * b1 - F * b2) / det
    v_prime = (-F * b1 + E * b2) / det
    return u_prime, v_prime


def normal_curvature(surface, u, v, u_prime, v_prime):
    """
    Нормальная кривизна κ_n = II(τ,τ) / I(τ,τ).
    u_prime, v_prime — контравариантные компоненты направления.
    """
    L, M, N_ff = surface.second_fundamental_form(u, v)
    E, F, G = surface.first_fundamental_form(u, v)
    II_val = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_ff * v_prime**2
    I_val = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
    if abs(I_val) < 1e-15:
        return 0.0
    return II_val / I_val


def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
    """
    Ковариантные компоненты ∂Φ/∂u^α на поверхности.

    Φ = ⟨R − r, m⟩,  τ = (R − r)/λ

    ∂Φ/∂u^α = ⟨R − r, ∂_α m⟩ = −λ b_α^β τ_β

    где b_α^β = g^{βγ} b_{αγ} — смешанный тензор Вейнгартена,
        τ_β = g_{βδ} τ^δ — ковариантные компоненты проекции τ.
    """
    E, F, G = surface.first_fundamental_form(u, v)
    L, M, N_ff = surface.second_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика в grad_Phi")

    # Обратная метрика
    guu = G / det
    guv = -F / det
    gvv = E / det

    # Ковариантные компоненты τ_α = g_{αβ} τ^β
    tau_u = E * u_prime + F * v_prime
    tau_v = F * u_prime + G * v_prime

    # Смешанный тензор Вейнгартена b_α^β = g^{βγ} b_{αγ}
    # α = u: b_{u,γ} = (L, M),  b_u^u = g^{uu}L + g^{uv}M, b_u^v = g^{uv}L + g^{vv}M
    # α = v: b_{v,γ} = (M, N),  b_v^u = g^{uu}M + g^{uv}N, b_v^v = g^{uv}M + g^{vv}N
    b_u_u = guu * L + guv * M
    b_u_v = guv * L + gvv * M
    b_v_u = guu * M + guv * N_ff
    b_v_v = guv * M + gvv * N_ff

    # ∂Φ/∂u = −λ (b_u^β τ_β) = −λ (b_u^u τ_u + b_u^v τ_v)
    dPhidu = -lam * (b_u_u * tau_u + b_u_v * tau_v)
    dPhidv = -lam * (b_v_u * tau_u + b_v_v * tau_v)
    return dPhidu, dPhidv


def inverse_metric(surface, u, v):
    """Компоненты обратной метрики g^{αβ}."""
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    return G / det, -F / det, E / det   # guu, guv, gvv


def recompute_thread_geometry(surface, traj, u, v, z):
    """
    Пересчёт геометрии нити в точке (u, v) при параметре z.

    Возвращает:
        tau_3d  — единичный вектор направления нити (R − r)/λ
        lam     — длина свободного участка
        u_p, v_p — контравариантные компоненты проекции τ на T_r S
    """
    r = surface.position(u, v)
    R = traj.R(z)
    delta = R - r
    lam = np.linalg.norm(delta)
    if lam < 1e-13:
        raise ValueError("Нулевая длина нити")
    tau_3d = delta / lam
    u_p, v_p = compute_tangent_components(surface, u, v, tau_3d)
    return tau_3d, lam, u_p, v_p


# ----------------------------------------------------------------------
# 3. Алгоритм обратного интегрирования (исправленный FNC-корректор)
# ----------------------------------------------------------------------
print("\n===== Обратная задача (FNC-корректор, исправленный) =====")

z_eval = np.linspace(0, traj.total_length, count_points)
N = len(z_eval)

u_hist = np.zeros(N)
v_hist = np.zeros(N)
Phi_hist = np.zeros(N)
kappa_n_hist = np.zeros(N)
newton_iters_hist = np.zeros(N, dtype=int)
lam_hist = np.zeros(N)

# Параметры
eps_Phi = 1e-8          # порог невязки
eps_grad = 1e-14        # порог вырождения градиента
max_newton = 20          # максимум итераций Ньютона на шаг

# Начальные условия
u_cur, v_cur = u0, v0
u_hist[0], v_hist[0] = u0, v0

# Инициализация: du/dz, dv/dz для предиктора
# На первом шаге используем формулу Савина, далее — экстраполяцию
du_dz_prev = None
dv_dz_prev = None

for i in range(N - 1):
    z_k = z_eval[i]
    z_next = z_eval[i + 1]
    dz = z_next - z_k

    # ------------------------------------------------------------------
    # A. Геометрия нити в текущей точке
    # ------------------------------------------------------------------
    try:
        tau_3d_k, lam_k, up_k, vp_k = recompute_thread_geometry(
            E2, traj, u_cur, v_cur, z_k
        )
    except ValueError as e:
        print(f"Шаг {i}: ошибка геометрии — {e}")
        u_hist[i + 1:] = u_cur
        v_hist[i + 1:] = v_cur
        break

    lam_hist[i] = lam_k
    m_k = E2.normal(u_cur, v_cur)

    kn_k = normal_curvature(E2, u_cur, v_cur, up_k, vp_k)
    kappa_n_hist[i] = kn_k

    # ------------------------------------------------------------------
    # B. Предиктор
    # ------------------------------------------------------------------
    # Скорость Савина: v = ⟨R'(z), m⟩ / (λ κ_n)
    R_prime_k = traj.R_deriv(z_k)
    savin_numerator = np.dot(R_prime_k, m_k)

    if abs(kn_k) < 1e-12:
        # Вырождение κ_n → 0: используем экстраполяцию или малый шаг
        print(f"  Шаг {i}: κ_n ≈ 0 ({kn_k:.2e}), экстраполяция")
        if du_dz_prev is not None:
            u_pred = u_cur + du_dz_prev * dz
            v_pred = v_cur + dv_dz_prev * dz
        else:
            u_pred = u_cur
            v_pred = v_cur
    else:
        v_savin = savin_numerator / (lam_k * kn_k)

        if du_dz_prev is not None and i >= 2:
            # Предиктор второго порядка: экстраполяция + Савин (среднее)
            # Используем экстраполяцию для направления,
            # Савина — для масштаба скорости
            du_dz_extrap = du_dz_prev
            dv_dz_extrap = dv_dz_prev

            # Нормируем экстраполяцию и масштабируем скоростью Савина
            E_f, F_f, G_f = E2.first_fundamental_form(u_cur, v_cur)
            speed_extrap = np.sqrt(
                E_f * du_dz_extrap**2
                + 2 * F_f * du_dz_extrap * dv_dz_extrap
                + G_f * dv_dz_extrap**2
            )
            if speed_extrap > 1e-14:
                # Направление из экстраполяции, скорость из Савина
                scale = abs(v_savin) / speed_extrap
                du_dz_pred = du_dz_extrap * scale * np.sign(v_savin)
                dv_dz_pred = dv_dz_extrap * scale * np.sign(v_savin)
            else:
                du_dz_pred = v_savin * up_k
                dv_dz_pred = v_savin * vp_k
        else:
            # Первые шаги: направление из проекции нити
            # (совпадает с направлением укладки при малых углах)
            du_dz_pred = v_savin * up_k
            dv_dz_pred = v_savin * vp_k

        u_pred = u_cur + du_dz_pred * dz
        v_pred = v_cur + dv_dz_pred * dz

    # ------------------------------------------------------------------
    # C. Итеративный корректор Ньютона
    # ------------------------------------------------------------------
    u_c, v_c = u_pred, v_pred
    converged = False

    for nit in range(max_newton):
        # Пересчёт геометрии в точке корректора
        r_c = E2.position(u_c, v_c)
        m_c = E2.normal(u_c, v_c)
        R_next = traj.R(z_next)
        delta_c = R_next - r_c
        lam_c = np.linalg.norm(delta_c)

        # Невязка
        Phi_c = np.dot(delta_c, m_c)

        if abs(Phi_c) < eps_Phi:
            converged = True
            Phi_hist[i + 1] = Phi_c
            newton_iters_hist[i + 1] = nit
            break

        if lam_c < 1e-13:
            print(f"  Шаг {i}, Ньютон iter {nit}: λ → 0, останов")
            break

        # Направление нити в точке корректора
        tau_c = delta_c / lam_c
        try:
            up_c, vp_c = compute_tangent_components(E2, u_c, v_c, tau_c)
        except ValueError:
            print(f"  Шаг {i}, Ньютон iter {nit}: вырождение метрики")
            break

        # Градиент Φ
        try:
            dPdu, dPdv = compute_grad_Phi(E2, u_c, v_c, up_c, vp_c, lam_c)
        except ValueError:
            print(f"  Шаг {i}, Ньютон iter {nit}: ошибка градиента")
            break

        # Обратная метрика
        guu, guv, gvv = inverse_metric(E2, u_c, v_c)

        # |∇_S Φ|² = g^{αβ} ∂_α Φ ∂_β Φ
        N_grad = guu * dPdu**2 + 2.0 * guv * dPdu * dPdv + gvv * dPdv**2

        if N_grad < eps_grad:
            print(f"  Шаг {i}, Ньютон iter {nit}: |∇Φ|² = {N_grad:.2e}, вырождение")
            break

        # Шаг Ньютона: δu^α = −Φ / |∇Φ|² · g^{αβ} ∂_β Φ
        du_corr = -Phi_c / N_grad * (guu * dPdu + guv * dPdv)
        dv_corr = -Phi_c / N_grad * (guv * dPdu + gvv * dPdv)

        u_c += du_corr
        v_c += dv_corr

    if not converged:
        # Записать последнюю невязку
        r_c = E2.position(u_c, v_c)
        m_c = E2.normal(u_c, v_c)
        Phi_hist[i + 1] = np.dot(traj.R(z_next) - r_c, m_c)
        newton_iters_hist[i + 1] = max_newton

    u_cur, v_cur = u_c, v_c
    u_hist[i + 1] = u_cur
    v_hist[i + 1] = v_cur

    # ------------------------------------------------------------------
    # D. Обновление du/dz для экстраполяции на следующем шаге
    # ------------------------------------------------------------------
    if dz > 1e-15:
        du_dz_prev = (u_hist[i + 1] - u_hist[i]) / dz
        dv_dz_prev = (v_hist[i + 1] - v_hist[i]) / dz

# Последний шаг: заполнить lam и kappa_n
try:
    _, lam_last, up_last, vp_last = recompute_thread_geometry(
        E2, traj, u_hist[-1], v_hist[-1], z_eval[-1]
    )
    lam_hist[-1] = lam_last
    kappa_n_hist[-1] = normal_curvature(E2, u_hist[-1], v_hist[-1], up_last, vp_last)
except ValueError:
    pass

# 3D-точки восстановленной линии
points_E2 = np.array([E2.position(u_hist[i], v_hist[i]) for i in range(N)])
line_E2 = points_E2

# ----------------------------------------------------------------------
# 4. Расширенная диагностика
# ----------------------------------------------------------------------
print(f"\nМаксимальная невязка |Φ|: {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя невязка |Φ|:      {np.mean(np.abs(Phi_hist)):.2e}")
print(f"Среднее число итераций Ньютона: {np.mean(newton_iters_hist[1:]):.2f}")
print(f"Максимум итераций Ньютона:      {np.max(newton_iters_hist[1:])}")

fig_diag, axes = plt.subplots(2, 2, figsize=(14, 10))

# Невязка Φ
axes[0, 0].plot(z_eval, Phi_hist, 'b-', linewidth=1.2)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 0].axhline(y=eps_Phi, color='red', linestyle=':', linewidth=0.8, label=f'ε = {eps_Phi}')
axes[0, 0].axhline(y=-eps_Phi, color='red', linestyle=':', linewidth=0.8)
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('Φ')
axes[0, 0].set_title('Невязка условия тени')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Число итераций Ньютона
axes[0, 1].bar(z_eval[1:], newton_iters_hist[1:], width=0.8 * (z_eval[1] - z_eval[0]),
               color='steelblue', alpha=0.7)
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('Итерации')
axes[0, 1].set_title('Итерации Ньютона на шаг')
axes[0, 1].grid(True, axis='y')

# Нормальная кривизна
axes[1, 0].plot(z_eval, kappa_n_hist, 'g-', linewidth=1.2)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('κ_n')
axes[1, 0].set_title('Нормальная кривизна оправки')
axes[1, 0].grid(True)

# Длина нити
axes[1, 1].plot(z_eval, lam_hist, 'm-', linewidth=1.2)
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('λ')
axes[1, 1].set_title('Длина свободного участка нити')
axes[1, 1].grid(True)

fig_diag.suptitle('Диагностика FNC-корректора', fontsize=14)
plt.tight_layout()
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
