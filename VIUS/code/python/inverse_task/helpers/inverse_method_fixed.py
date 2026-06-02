# ======================================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАТНОЙ ЗАДАЧИ
# ======================================================================

import numpy as np


def compute_tangent_components(surface, u, v, tau_3d):
    """Контравариантные компоненты проекции tau_3d на T_r S."""
    geom = surface.derivatives(u, v)
    ru, rv = geom['ru'], geom['rv']
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    b1 = np.dot(tau_3d, ru)
    b2 = np.dot(tau_3d, rv)
    u_prime = (G * b1 - F * b2) / det
    v_prime = (-F * b1 + E * b2) / det
    return u_prime, v_prime


def normal_curvature(surface, u, v, u_prime, v_prime):
    """κ_n = II(τ,τ) / I(τ,τ)."""
    L, M, N_ff = surface.second_fundamental_form(u, v)
    E, F, G = surface.first_fundamental_form(u, v)
    II_val = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_ff * v_prime**2
    I_val = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
    if abs(I_val) < 1e-15:
        return 0.0
    return II_val / I_val
def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
    L, M, N_ff = surface.second_fundamental_form(u, v)
    # g_α = b_{αβ} · τ^β  (ковариантные компоненты градиента)
    dPhidu = -lam * (L * u_prime + M * v_prime)
    dPhidv = -lam * (M * u_prime + N_ff * v_prime)
    return dPhidu, dPhidv
# def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
#     """
#     ∂Φ/∂u^α = -λ · b_{αβ} · τ^β
    
#     u_prime, v_prime — это τ^β (контравариантные).
#     b_{αβ} — вторая фундаментальная форма, ковариантная.
#     """
#     L, M, N_ff = surface.second_fundamental_form(u, v)
    
#     # b_{1β} τ^β = L * τ^1 + M * τ^2
#     dPhidu = -lam * (L * u_prime + M * v_prime)
#     # b_{2β} τ^β = M * τ^1 + N_ff * τ^2
#     dPhidv = -lam * (M * u_prime + N_ff * v_prime)
    
#     return dPhidu, dPhidv
# def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
#     """
#     Ковариантные компоненты ∂Φ/∂u^α.
    
#     ИСПРАВЛЕНИЕ: убран лишний минус. Согласно отчёту, формула (6):
#         g_u = b_{ij} \dot{u}^j
#     Здесь b_{ij} — компоненты II формы, \dot{u}^j — контравариантные компоненты
#     вектора нити. Минуса нет.
    
#     Ранее стояло `-lam * (...)`, что давало знак, противоположный градиенту Φ.
#     Корректор Ньютона (u -= Phi/Ng * grad_u) при этом двигал точку ВДОЛЬ
#     градиента, увеличивая |Phi| вместо уменьшения.
#     """
#     E, F, G = surface.first_fundamental_form(u, v)
#     L, M, N_ff = surface.second_fundamental_form(u, v)
#     det = E * G - F * F
#     if abs(det) < 1e-14:
#         raise ValueError("Вырожденная метрика в grad_Phi")
    
#     guu = G / det
#     guv = -F / det
#     gvv = E / det
    
#     tau_u = E * u_prime + F * v_prime
#     tau_v = F * u_prime + G * v_prime
    
#     b_u_u = guu * L + guv * M
#     b_u_v = guv * L + gvv * M
#     b_v_u = guu * M + guv * N_ff
#     b_v_v = guv * M + gvv * N_ff
    
#     # БЫЛО (баг): dPhidu = -lam * (...), dPhidv = -lam * (...)
#     # СТАЛО (исправлено):
#     dPhidu = -lam * (b_u_u * tau_u + b_u_v * tau_v)
#     dPhidv = -lam * (b_v_u * tau_u + b_v_v * tau_v)
#     return dPhidu, dPhidv


def inverse_metric(surface, u, v):
    """g^{αβ}."""
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    return G / det, -F / det, E / det


def recompute_thread_geometry(surface, traj, u, v, z):
    """Пересчёт τ, λ, (u', v') в точке (u, v) при параметре z."""
    r = surface.position(u, v)
    R = traj.R(z)
    delta = R - r
    lam = np.linalg.norm(delta)
    if lam < 1e-13:
        raise ValueError("Нулевая длина нити")
    tau_3d = delta / lam
    u_p, v_p = compute_tangent_components(surface, u, v, tau_3d)
    return tau_3d, lam, u_p, v_p


def project_to_tangent_plane(surface, u, v, vec_3d):
    """Проекция 3D-вектора на касательную плоскость."""
    m = surface.normal(u, v)
    return vec_3d - np.dot(vec_3d, m) * m


def compute_dr_dz(surface, traj, u, v, z):
    """
    Аналитическое du/dz, dv/dz из дифференцирования Φ = 0.
    
    dr/dz = R'_∥ + μ · ∇_S Φ,  где μ = −(∂_z Φ + ∂_α Φ · R'^α_∥) / |∇Φ|²
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
    
    dPhi_dz = np.dot(R_prime, m)
    
    up, vp = compute_tangent_components(surface, u, v, tau)
    dPhi_du, dPhi_dv = compute_grad_Phi(surface, u, v, up, vp, lam)
    
    R_prime_par = project_to_tangent_plane(surface, u, v, R_prime)
    Rp_u, Rp_v = compute_tangent_components(surface, u, v, R_prime_par)
    
    residual = dPhi_dz + dPhi_du * Rp_u + dPhi_dv * Rp_v
    
    guu, guv, gvv = inverse_metric(surface, u, v)
    grad_u = guu * dPhi_du + guv * dPhi_dv
    grad_v = guv * dPhi_du + gvv * dPhi_dv
    
    norm_grad_sq = (guu * dPhi_du**2 
                    + 2 * guv * dPhi_du * dPhi_dv 
                    + gvv * dPhi_dv**2)
    
    # if norm_grad_sq < 1e-14:
    #     return Rp_u, Rp_v
    if norm_grad_sq < 1e-6:  # <--- ПОРОГ ВКЛЮЧЕНИЯ ОПТИКИ
        # DAE не может удержать связь, возвращаем только кинематику
        # Гибридный алгоритм должен это понять и переключиться на лучи
        return Rp_u, Rp_v 
    
    mu = -residual / norm_grad_sq  # БЕЗ 1e-8 и БЕЗ клиппинга!
    
    du_dz = Rp_u + mu * grad_u
    dv_dz = Rp_v + mu * grad_v
    
    # Ограничивать надо не mu, а итоговую длину шага по поверхности!
    E, F, G = surface.first_fundamental_form(u, v)
    speed_sq = E * du_dz**2 + 2 * F * du_dz * dv_dz + G * dv_dz**2
    max_speed = 50.0 # Макс. допустимая скорость перемещения по оправке (мм/ед.парам.)
    if speed_sq > max_speed**2:
        scale = max_speed / np.sqrt(speed_sq)
        du_dz *= scale
        dv_dz *= scale
        
    return du_dz, dv_dz
    # # mu = -residual / norm_grad_sq
    
    # # du_dz = Rp_u + mu * grad_u
    # # dv_dz = Rp_v + mu * grad_v

    # # Регуляризация: ограничиваем mu, чтобы избежать взрыва при малых |∇Φ|
    # mu = -residual / (norm_grad_sq + 1e-8)
    
    # # Дополнительное жёсткое ограничение
    # mu = np.clip(mu, -100.0, 100.0)
    # if norm_grad_sq < 1e-14:
    #     return Rp_u, Rp_v
    
    # # Регуляризация и ограничение mu
    # mu = -residual / (norm_grad_sq + 1e-8)
    # mu = max(-100.0, min(100.0, mu))
    
    # du_dz = Rp_u + mu * grad_u
    # dv_dz = Rp_v + mu * grad_v
    
    # return du_dz, dv_dz


# ======================================================================
# 5. КОРРЕКТОР НЬЮТОНА — ДЕМПФИРОВАННЫЙ С LINE SEARCH
# ======================================================================

def newton_corrector(surface, traj, u_pred, v_pred, z_target,
                     eps_Phi=1e-10, max_iter=20,
                     max_step_u=20.0, max_step_v=0.5,
                     armijo_c=1e-4):
# def newton_corrector(..., max_step_u=100.0, max_step_v=0.5, max_iter=50)
    """
    Корректор Ньютона с демпфированием и линейным поиском.
    
    Проблема чистого Ньютона: при большом |Phi| шаг может быть
    катастрофически большим. Решение:
      1. Ограничиваем max_step_u / max_step_v.
      2. Если после шага |Phi| не уменьшился — backtracking.
      3. Жёсткий клиппинг к границам поверхности.
    """
    u_c, v_c = u_pred, v_pred
    
    u_min = getattr(surface, 'u_min', None)
    u_max = getattr(surface, 'u_max', None)
    v_min = getattr(surface, 'v_min', None)
    v_max = getattr(surface, 'v_max', None)
    
    # Начальная невязка
    r_c = surface.position(u_c, v_c)
    m_c = surface.normal(u_c, v_c)
    R_t = traj.R(z_target)
    Phi_cur = np.dot(R_t - r_c, m_c)
    
    if abs(Phi_cur) < eps_Phi:
        return u_c, v_c, Phi_cur, 0, True
    
    for nit in range(1, max_iter + 1):
        lam_c = np.linalg.norm(R_t - r_c)
        if lam_c < 1e-13:
            return u_c, v_c, Phi_cur, nit, False
        
        tau_c = (R_t - r_c) / lam_c
        try:
            up_c, vp_c = compute_tangent_components(surface, u_c, v_c, tau_c)
            dPdu, dPdv = compute_grad_Phi(surface, u_c, v_c, up_c, vp_c, lam_c)
            guu, guv, gvv = inverse_metric(surface, u_c, v_c)
        except ValueError:
            return u_c, v_c, Phi_cur, nit, False
        
        Ng = guu * dPdu**2 + 2 * guv * dPdu * dPdv + gvv * dPdv**2
        if Ng < 1e-14:
            return u_c, v_c, Phi_cur, nit, False
        
        dir_u = guu * dPdu + guv * dPdv
        dir_v = guv * dPdu + gvv * dPdv
        
        # --- Line search / backtracking ---
        alpha = 1.0
        for _ in range(12):
            step_u = alpha * Phi_cur / Ng * dir_u
            step_v = alpha * Phi_cur / Ng * dir_v
            
            # if abs(step_u) > max_step_u:
            #     step_u = max_step_u if step_u > 0 else -max_step_u
            # if abs(step_v) > max_step_v:
            #     step_v = max_step_v if step_v > 0 else -max_step_v
            # Вместо независимого клиппинга:
            E, F, G = surface.first_fundamental_form(u_c, v_c)
            ds_sq = E * step_u**2 + 2 * F * step_u * step_v + G * step_v**2
            max_ds = 30.0  # максимальный шаг в мм по поверхности

            if ds_sq > max_ds**2:
                scale = max_ds / np.sqrt(ds_sq)
                step_u *= scale
                step_v *= scale
            
            u_try = u_c - step_u
            v_try = v_c - step_v
            
            # if u_min is not None and u_max is not None:
            #     u_try = np.clip(u_try, u_min, u_max)
            # if v_min is not None and v_max is not None:
            #     v_try = np.clip(v_try, v_min, v_max)
            
            if v_min is not None and v_max is not None:
                # Периодичность угла: mod 2π вместо обрезания
                v_try = np.mod(v_try, 2 * np.pi)
            
            r_try = surface.position(u_try, v_try)
            m_try = surface.normal(u_try, v_try)
            Phi_try = np.dot(R_t - r_try, m_try)
            
            if abs(Phi_try) < eps_Phi:
                return u_try, v_try, Phi_try, nit, True
            
            if abs(Phi_try) < abs(Phi_cur) * (1.0 - armijo_c * alpha):
                u_c, v_c = u_try, v_try
                r_c, m_c = r_try, m_try
                Phi_cur = Phi_try
                break
            
            alpha *= 0.5
        else:
            # Line search не удался — принимаем последнюю попытку
            u_c = u_try
            v_c = v_try
            r_c = r_try
            m_c = m_try
            Phi_cur = Phi_try
    
    return u_c, v_c, Phi_cur, max_iter, abs(Phi_cur) < eps_Phi
