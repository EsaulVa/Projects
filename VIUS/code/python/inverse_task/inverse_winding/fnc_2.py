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
u0, v0 = 0.7, 0.0
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
# Вспомогательные функции (дополнение)
# ----------------------------------------------------------------------

def project_to_tangent_plane(surface, u, v, vec_3d):
    """
    Проекция 3D-вектора на касательную плоскость к surface в (u, v).
    Возвращает спроецированный 3D-вектор.
    """
    m = surface.normal(u, v)
    return vec_3d - np.dot(vec_3d, m) * m


def compute_dr_dz(surface, traj, u, v, z):
    """
    Аналитическое вычисление du/dz, dv/dz из условия тени.
    
    Используем:
      dΦ/dz = ∂Φ/∂z + (∂Φ/∂u)(du/dz) + (∂Φ/∂v)(dv/dz) = 0
    
    Плюс условие: направление dr/dz определяется проекцией R'(z)
    на касательную плоскость с учётом связи Φ = 0.
    
    Метод: dr/dz = R'_∥ − (dλ/dz)·τ_∥ − λ·(dτ/dz)_∥
    Упрощение (Савин): при условии Φ = 0 проекция R' на m 
    полностью уходит в dλ/dz, и:
    
      (dr/dz)_∥ ≈ R'_∥ + коррекция от кривизны
    
    Здесь реализуем точную формулу через дифференцирование Φ.
    """
    r = surface.position(u, v)
    R = traj.R(z)
    m = surface.normal(u, v)
    delta = R - r
    lam = np.linalg.norm(delta)
    
    if lam < 1e-13:
        return 0.0, 0.0
    
    tau = delta / lam
    R_prime = traj.R_deriv(z)
    
    # ∂Φ/∂z = ⟨R'(z), m⟩
    dPhi_dz = np.dot(R_prime, m)
    
    # Контравариантные компоненты τ на поверхности
    up, vp = compute_tangent_components(surface, u, v, tau)
    
    # Ковариантные компоненты градиента Φ
    dPhi_du, dPhi_dv = compute_grad_Phi(surface, u, v, up, vp, lam)
    
    # Проекция R' на касательную плоскость → контравариантные компоненты
    R_prime_par = project_to_tangent_plane(surface, u, v, R_prime)
    Rp_u, Rp_v = compute_tangent_components(surface, u, v, R_prime_par)
    
    # Стратегия: dr/dz = R'_∥ + коррекция, чтобы dΦ/dz = 0.
    #
    # Если бы du/dz = Rp_u, dv/dz = Rp_v (просто проекция R'),
    # то dΦ/dz + dPhi_du·Rp_u + dPhi_dv·Rp_v ≠ 0 в общем случае.
    #
    # Невязка:
    residual = dPhi_dz + dPhi_du * Rp_u + dPhi_dv * Rp_v
    
    # Корректируем вдоль ∇_S Φ:
    # du/dz = Rp_u + μ · (g^{uα} ∂_α Φ)
    # dv/dz = Rp_v + μ · (g^{vα} ∂_α Φ)
    # где μ выбирается из dΦ/dz = 0:
    #   residual + μ · |∇Φ|² = 0  →  μ = −residual / |∇Φ|²
    
    guu, guv, gvv = inverse_metric(surface, u, v)
    grad_u = guu * dPhi_du + guv * dPhi_dv  # (∇Φ)^u
    grad_v = guv * dPhi_du + gvv * dPhi_dv  # (∇Φ)^v
    
    norm_grad_sq = guu * dPhi_du**2 + 2 * guv * dPhi_du * dPhi_dv + gvv * dPhi_dv**2
    
    if norm_grad_sq < 1e-14:
        # Вырождение: ∇Φ ≈ 0, используем просто проекцию R'
        return Rp_u, Rp_v
    
    mu = -residual / norm_grad_sq
    
    du_dz = Rp_u + mu * grad_u
    dv_dz = Rp_v + mu * grad_v
    
    return du_dz, dv_dz

def adaptive_step_with_continuity_control(
    surface, traj, z_eval, u0, v0,
    eps_Phi=1e-10, max_newton=7, max_bisect=4, jump_ratio_threshold=3.0
):
    """
    Обратное интегрирование с:
    1. Аналитическим предиктором (compute_dr_dz)
    2. Корректором Ньютона
    3. Контролем непрерывности: если скачок > порога,
       делим шаг пополам (бисекция)
    """
    N = len(z_eval)
    
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N)
    flags = np.zeros(N, dtype=int)  # 0 = OK, 1 = bisected, 2 = warning
    
    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0
    
    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        
        # Геометрия в текущей точке
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k
            )
        except ValueError as e:
            print(f"Шаг {i}: {e}")
            u_hist[i+1:] = u_cur; v_hist[i+1:] = v_cur
            break
        
        lam_hist[i] = lam_k
        kappa_n_hist[i] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)
        
        # Аналитический предиктор
        try:
            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
        except ValueError:
            du_dz_k, dv_dz_k = 0.0, 0.0
        
        # Ожидаемое метрическое расстояние за полный шаг
        E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
        speed_sq = (E_f * du_dz_k**2 
                    + 2*F_f * du_dz_k * dv_dz_k 
                    + G_f * dv_dz_k**2)
        
        # -------------------------------------------------------
        # Бисекция: делаем шаг, проверяем скачок, если нужно — 
        # делим на подшаги
        # -------------------------------------------------------
        n_sub = 1  # число подшагов
        best_u, best_v = u_cur, v_cur
        best_Phi = 1.0
        best_nit = 0
        
        for bisect_iter in range(max_bisect + 1):
            sub_z = np.linspace(z_k, z_next, n_sub + 1)
            u_s, v_s = u_cur, v_cur
            total_nit = 0
            success = True
            
            for j in range(n_sub):
                z_a = sub_z[j]
                z_b = sub_z[j + 1]
                dz_sub = z_b - z_a
                
                # Предиктор в подточке
                try:
                    du_s, dv_s = compute_dr_dz(
                        surface, traj, u_s, v_s, z_a
                    )
                except ValueError:
                    du_s, dv_s = du_dz_k, dv_dz_k
                
                u_p = u_s + du_s * dz_sub
                v_p = v_s + dv_s * dz_sub
                
                # Корректор Ньютона
                u_c, v_c = u_p, v_p
                conv = False
                nit_sub = 0
                
                for nit in range(max_newton):
                    r_c = surface.position(u_c, v_c)
                    m_c = surface.normal(u_c, v_c)
                    R_b = traj.R(z_b)
                    delta_c = R_b - r_c
                    lam_c = np.linalg.norm(delta_c)
                    Phi_c = np.dot(delta_c, m_c)
                    
                    if abs(Phi_c) < eps_Phi:
                        conv = True
                        nit_sub = nit
                        break
                    
                    if lam_c < 1e-13:
                        break
                    
                    tau_c = delta_c / lam_c
                    try:
                        up_c, vp_c = compute_tangent_components(
                            surface, u_c, v_c, tau_c
                        )
                        dPdu, dPdv = compute_grad_Phi(
                            surface, u_c, v_c, up_c, vp_c, lam_c
                        )
                    except ValueError:
                        break
                    
                    guu, guv, gvv = inverse_metric(surface, u_c, v_c)
                    Ng = (guu * dPdu**2 
                          + 2*guv * dPdu * dPdv 
                          + gvv * dPdv**2)
                    
                    if Ng < 1e-14:
                        break
                    
                    u_c -= Phi_c / Ng * (guu * dPdu + guv * dPdv)
                    v_c -= Phi_c / Ng * (guv * dPdu + gvv * dPdv)
                    nit_sub = nit
                
                total_nit += nit_sub
                
                # Проверка скачка в подшаге
                du_j = u_c - u_s
                dv_j = v_c - v_s
                Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                ds_actual = np.sqrt(
                    Ej * du_j**2 + 2*Fj * du_j * dv_j + Gj * dv_j**2
                )
                ds_expect = np.sqrt(
                    Ej * du_s**2 + 2*Fj * du_s * dv_s + Gj * dv_s**2
                ) * dz_sub
                
                ratio = ds_actual / (ds_expect + 1e-15) if ds_expect > 1e-15 else 1.0
                
                if ratio > jump_ratio_threshold and bisect_iter < max_bisect:
                    success = False
                    break
                
                u_s, v_s = u_c, v_c
            
            if success:
                best_u, best_v = u_s, v_s
                # Финальная невязка
                r_f = surface.position(best_u, best_v)
                m_f = surface.normal(best_u, best_v)
                best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                best_nit = total_nit
                if bisect_iter > 0:
                    flags[i + 1] = 1  # бисекция была
                break
            else:
                n_sub *= 2  # удвоить число подшагов
        
        u_cur, v_cur = best_u, best_v
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = best_Phi
        newton_iters_hist[i + 1] = best_nit
    
    # Финальная точка
    try:
        _, lam_last, up_last, vp_last = recompute_thread_geometry(
            surface, traj, u_hist[-1], v_hist[-1], z_eval[-1]
        )
        lam_hist[-1] = lam_last
        kappa_n_hist[-1] = normal_curvature(
            surface, u_hist[-1], v_hist[-1], up_last, vp_last
        )
    except ValueError:
        pass
    
    return {
        'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags
    }


# ----------------------------------------------------------------------
# Запуск
# ----------------------------------------------------------------------
print("\n===== Обратная задача (v3: бисекция + контроль непрерывности) =====")

result = adaptive_step_with_continuity_control(
    E2, traj, z_eval, u0, v0,
    eps_Phi=1e-10,
    max_newton=7,
    max_bisect=4,
    jump_ratio_threshold=3.0
)

u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
lam_hist = result['lam']
flags = result['flags']

n_bisected = np.sum(flags == 1)
print(f"\nШагов с бисекцией: {n_bisected} из {len(z_eval)-1}")
print(f"Максимальная невязка |Φ|: {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя невязка |Φ|:      {np.mean(np.abs(Phi_hist)):.2e}")

# 3D-точки
points_E2 = np.array([E2.position(u_hist[i], v_hist[i]) for i in range(len(z_eval))])
line_E2 = points_E2

# ----------------------------------------------------------------------
# Диагностика
# ----------------------------------------------------------------------
fig_diag, axes = plt.subplots(2, 3, figsize=(18, 10))

# Невязка
axes[0, 0].semilogy(z_eval[1:], np.abs(Phi_hist[1:]) + 1e-16, 'b-', lw=1.2)
axes[0, 0].axhline(y=1e-10, color='red', ls=':', lw=0.8, label='ε')
axes[0, 0].set_title('|Φ| (лог)'); axes[0, 0].legend(); axes[0, 0].grid(True)

# Итерации Ньютона
axes[0, 1].bar(z_eval[1:], newton_iters_hist[1:],
               width=0.8*(z_eval[1]-z_eval[0]), color='steelblue', alpha=0.7)
axes[0, 1].set_title('Итерации Ньютона'); axes[0, 1].grid(True, axis='y')

# Флаги бисекции
axes[0, 2].bar(z_eval[1:], flags[1:],
               width=0.8*(z_eval[1]-z_eval[0]), color='orange', alpha=0.7)
axes[0, 2].set_title('Бисекция (1 = да)'); axes[0, 2].grid(True, axis='y')

# κ_n
axes[1, 0].plot(z_eval, kappa_n_hist, 'g-', lw=1.2)
axes[1, 0].axhline(y=0, color='gray', ls='--', lw=0.5)
axes[1, 0].set_title('κ_n'); axes[1, 0].grid(True)

# λ
axes[1, 1].plot(z_eval, lam_hist, 'm-', lw=1.2)
axes[1, 1].set_title('λ (длина нити)'); axes[1, 1].grid(True)

# u(z), v(z)
axes[1, 2].plot(z_eval, u_hist, 'r-', lw=1.2, label='u(z)')
axes[1, 2].plot(z_eval, v_hist, 'b-', lw=1.2, label='v(z)')
axes[1, 2].set_title('u(z), v(z)'); axes[1, 2].legend(); axes[1, 2].grid(True)

fig_diag.suptitle('Диагностика v3: бисекция + контроль', fontsize=14)
plt.tight_layout()
plt.show()
