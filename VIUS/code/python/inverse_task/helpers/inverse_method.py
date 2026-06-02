
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
    """Ковариантные компоненты ∂Φ/∂u^α."""
    E, F, G = surface.first_fundamental_form(u, v)
    L, M, N_ff = surface.second_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика в grad_Phi")
    
    guu = G / det
    guv = -F / det
    gvv = E / det
    
    tau_u = E * u_prime + F * v_prime
    tau_v = F * u_prime + G * v_prime
    
    b_u_u = guu * L + guv * M
    b_u_v = guv * L + gvv * M
    b_v_u = guu * M + guv * N_ff
    b_v_v = guv * M + gvv * N_ff
    
    dPhidu = -lam * (b_u_u * tau_u + b_u_v * tau_v)
    dPhidv = -lam * (b_v_u * tau_u + b_v_v * tau_v)
    return dPhidu, dPhidv


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
    
    if norm_grad_sq < 1e-14:
        return Rp_u, Rp_v
    
    mu = -residual / norm_grad_sq
    
    du_dz = Rp_u + mu * grad_u
    dv_dz = Rp_v + mu * grad_v
    
    return du_dz, dv_dz


# ======================================================================
# 5. КОРРЕКТОР НЬЮТОНА (одиночный шаг)
# ======================================================================

# ======================================================================
# 5. КОРРЕКТОР НЬЮТОНА — ДЕМПФИРОВАННЫЙ С LINE SEARCH
# ======================================================================

def newton_corrector(surface, traj, u_pred, v_pred, z_target,
                     eps_Phi=1e-10, max_iter=20,
                     max_step_u=5.0, max_step_v=0.05,
                     armijo_c=1e-4):
    """
    Корректор Ньютона с демпфированием и линейным поиском (Armijo-like).
    
    Проблема чистого Ньютона: при большом |Phi| шаг delta = -Phi*grad/|grad|^2
    может быть катастрофически большим, точка улетает за границы, клиппинг
    обрезает, и сходимости не происходит.
    
    Решение:
      1. Ограничиваем max_step_u / max_step_v (параметрический шаг).
      2. Если после шага |Phi| не уменьшился — уменьшаем шаг (backtracking).
      3. Жёсткий клиппинг к границам поверхности после каждого шага.
    
    Возвращает (u, v, Phi, n_iter, converged).
    """
    u_c, v_c = u_pred, v_pred
    
    # Универсальные границы
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
        
        # Направление Ньютона (контравариантный градиент)
        dir_u = guu * dPdu + guv * dPdv
        dir_v = guv * dPdu + gvv * dPdv
        
        # --- Line search / backtracking ---
        alpha = 1.0
        for _ in range(12):  # до 12 уменьшений шага (min alpha ~ 1/4096)
            step_u = alpha * Phi_cur / Ng * dir_u
            step_v = alpha * Phi_cur / Ng * dir_v
            
            # Демпфирование: ограничить абсолютный шаг
            if abs(step_u) > max_step_u:
                step_u = max_step_u if step_u > 0 else -max_step_u
            if abs(step_v) > max_step_v:
                step_v = max_step_v if step_v > 0 else -max_step_v
            
            u_try = u_c - step_u
            v_try = v_c - step_v
            
            # Клиппинг к границам
            if u_min is not None and u_max is not None:
                u_try = np.clip(u_try, u_min, u_max)
            if v_min is not None and v_max is not None:
                v_try = np.mod(v_try, 2 * np.pi)
            
            # Проверяем, уменьшилась ли невязка
            r_try = surface.position(u_try, v_try)
            m_try = surface.normal(u_try, v_try)
            Phi_try = np.dot(R_t - r_try, m_try)
            
            if abs(Phi_try) < eps_Phi:
                return u_try, v_try, Phi_try, nit, True
            
            # Условие Armijo: |Phi| должно уменьшиться хотя бы немного
            if abs(Phi_try) < abs(Phi_cur) * (1.0 - armijo_c * alpha):
                u_c, v_c = u_try, v_try
                r_c, m_c = r_try, m_try
                Phi_cur = Phi_try
                break
            
            alpha *= 0.5
        else:
            # Line search не удался — принимаем последнюю попытку с минимальным alpha
            # (лучше, чем застревать на месте)
            u_c = u_try
            v_c = v_try
            r_c = r_try
            m_c = m_try
            Phi_cur = Phi_try
    
    return u_c, v_c, Phi_cur, max_iter, abs(Phi_cur) < eps_Phi

# ======================================================================
# 6. ГЛАВНЫЙ АЛГОРИТМ v3: предиктор + корректор + бисекция
# ======================================================================

def inverse_winding_v3(surface, traj, u0, v0, count_points=300,
                       eps_Phi=1e-10, max_newton=7,
                       max_bisect=4, jump_threshold=3.0):
    """
    Обратная задача намотки.
    
    Параметры:
        surface        — оправка (Ellipsoid)
        traj           — траектория раскладчика (Trajectory3D)
        u0, v0         — начальная точка на оправке
        count_points   — число точек дискретизации
        eps_Phi        — порог невязки
        max_newton     — макс. итераций Ньютона
        max_bisect     — макс. уровней бисекции
        jump_threshold — порог отношения ds_actual / ds_expected
    
    Возвращает словарь с результатами.
    """
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)
    
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N)
    flags = np.zeros(N, dtype=int)  # 0=OK, 1=bisected, 2=warning
    
    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0
    
    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        
        # --- Геометрия в текущей точке ---
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k
            )
        except ValueError as e:
            print(f"Шаг {i}: {e}")
            u_hist[i+1:] = u_cur
            v_hist[i+1:] = v_cur
            break
        
        lam_hist[i] = lam_k
        kappa_n_hist[i] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)
        
        # --- Аналитический предиктор ---
        try:
            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
        except ValueError:
            du_dz_k, dv_dz_k = 0.0, 0.0
        
        # Ожидаемая скорость (для контроля скачков)
        E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
        speed_expected = np.sqrt(
            max(E_f * du_dz_k**2 
                + 2 * F_f * du_dz_k * dv_dz_k 
                + G_f * dv_dz_k**2, 0.0)
        )
        
        # --- Бисекция при необходимости ---
        n_sub = 1
        best_u, best_v = u_cur, v_cur
        best_Phi = 1.0
        best_nit = 0
        bisected = False
        
        for bisect_level in range(max_bisect + 1):
            sub_z = np.linspace(z_k, z_next, n_sub + 1)
            u_s, v_s = u_cur, v_cur
            total_nit = 0
            jump_detected = False
            
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
                if hasattr(surface, 'u_min') and hasattr(surface, 'u_max'):
                    u_p = np.clip(u_p, surface.u_min, surface.u_max)
                if hasattr(surface, 'v_min') and hasattr(surface, 'v_max'):
                    v_p = np.clip(v_p, surface.v_min, surface.v_max)
                # u_p = np.clip(u_p, surface.u_min, surface.u_max)
                # Корректор
                u_c, v_c, Phi_c, nit, conv = newton_corrector(
                    surface, traj, u_p, v_p, z_b,
                    eps_Phi=eps_Phi, max_iter=max_newton
                )
                total_nit += nit
                
                # Проверка скачка
                du_j = u_c - u_s
                dv_j = v_c - v_s
                Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                ds_actual = np.sqrt(
                    max(Ej * du_j**2 + 2*Fj * du_j * dv_j + Gj * dv_j**2, 0.0)
                )
                ds_expect = speed_expected * dz_sub
                
                if ds_expect > 1e-12:
                    ratio = ds_actual / ds_expect
                else:
                    ratio = 1.0
                
                if (ratio > jump_threshold or ratio < 1.0/jump_threshold) \
                        and bisect_level < max_bisect:
                    jump_detected = True
                    break
                
                u_s, v_s = u_c, v_c
            
            if not jump_detected:
                best_u, best_v = u_s, v_s
                r_f = surface.position(best_u, best_v)
                m_f = surface.normal(best_u, best_v)
                best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                best_nit = total_nit
                if bisect_level > 0:
                    bisected = True
                break
            else:
                n_sub *= 2
        
        u_cur, v_cur = best_u, best_v
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = best_Phi
        newton_iters_hist[i + 1] = best_nit
        flags[i + 1] = 1 if bisected else 0
    
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
    
    # 3D-точки
    points_3d = np.array([
        surface.position(u_hist[k], v_hist[k]) for k in range(N)
    ])
    
    return {
        'z_eval': z_eval,
        'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }

from scipy.integrate import solve_ivp

def inverse_winding_v4(surface, traj, u0, v0, count_points=300,
                       eps_Phi=1e-10, max_newton=7,
                       max_bisect=4, jump_threshold=3.0,
                       rtol=1e-8, atol=1e-10, max_step=None):
    """
    Обратная задача намотки с предиктором на основе встроенного решателя ОДУ.
    
    На каждом интервале [z_k, z_{k+1}] вызывается solve_ivp с правой частью
    compute_dr_dz, затем применяется корректор Ньютона. Если solve_ivp
    не справляется или корректор не сходится, используется бисекция.
    """
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)
    
    u_hist = np.zeros(N); v_hist = np.zeros(N)
    Phi_hist = np.zeros(N); kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N); flags = np.zeros(N, dtype=int)
    
    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0
    
    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        
        # Текущая геометрия
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k)
        except ValueError as e:
            print(f"Шаг {i}: {e}")
            break
        
        lam_hist[i] = lam_k
        kappa_n_hist[i] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)
        
        # Предиктор: пытаемся решить ОДУ от z_k до z_next
        def rhs(z, uv):
            du, dv = compute_dr_dz(surface, traj, uv[0], uv[1], z)
            return [du, dv]
        
        pred_ok = False
        try:
            sol = solve_ivp(rhs, [z_k, z_next], [u_cur, v_cur],
                            method='Radau', rtol=rtol, atol=atol,
                            max_step=max_step)
            if sol.success:
                u_pred = sol.y[0, -1]
                v_pred = sol.y[1, -1]
                pred_ok = True
        except Exception:
            pass
        
        if pred_ok:
            # Проверяем, не вышли ли за границы
            u_pred = np.clip(u_pred, surface.u_min, surface.u_max)
            # (v можно не клиппить, это угол)
            # Корректор Ньютона
            u_c, v_c, Phi_c, nit, conv = newton_corrector(
                surface, traj, u_pred, v_pred, z_next,
                eps_Phi=eps_Phi, max_iter=max_newton)
            if conv and abs(Phi_c) < eps_Phi:
                # Успех – принимаем без бисекции
                u_cur, v_cur = u_c, v_c
                u_hist[i+1], v_hist[i+1] = u_cur, v_cur
                Phi_hist[i+1] = Phi_c
                newton_iters_hist[i+1] = nit
                # Оценка скачка для диагностики (бисекцию не включаем)
                du_j = u_cur - u_hist[i]; dv_j = v_cur - v_hist[i]
                Ej, Fj, Gj = surface.first_fundamental_form(u_hist[i], v_hist[i])
                ds_actual = np.sqrt(max(Ej*du_j**2 + 2*Fj*du_j*dv_j + Gj*dv_j**2, 0.0))
                # speed_expected вычислим для информации, но порог не проверяем
                try:
                    du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_hist[i], v_hist[i], z_k)
                    E_f, F_f, G_f = surface.first_fundamental_form(u_hist[i], v_hist[i])
                    speed_expect = np.sqrt(max(E_f*du_dz_k**2 + 2*F_f*du_dz_k*dv_dz_k + G_f*dv_dz_k**2, 0.0))
                    ds_expect = speed_expect * (z_next - z_k)
                    if ds_expect > 1e-12:
                        ratio = ds_actual / ds_expect
                        if ratio > jump_threshold or ratio < 1.0/jump_threshold:
                            # не фатально, просто отметим
                            flags[i+1] = 2  # warning
                except:
                    pass
                continue  # переходим к следующему шагу
        
        # Если solve_ivp не удался или корректор не сошёлся, 
        # применяем старый механизм с бисекцией
        try:
            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
        except ValueError:
            du_dz_k, dv_dz_k = 0.0, 0.0
        
        E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
        speed_expected = np.sqrt(max(E_f * du_dz_k**2 
                                     + 2*F_f*du_dz_k*dv_dz_k 
                                     + G_f*dv_dz_k**2, 0.0))
        
        # Бисекция как раньше
        n_sub = 1
        best_u, best_v = u_cur, v_cur
        best_Phi = 1.0
        best_nit = 0
        bisected = False
        
        for bisect_level in range(max_bisect + 1):
            sub_z = np.linspace(z_k, z_next, n_sub + 1)
            u_s, v_s = u_cur, v_cur
            total_nit = 0
            jump_detected = False
            
            for j in range(n_sub):
                z_a = sub_z[j]
                z_b = sub_z[j + 1]
                dz_sub = z_b - z_a
                
                try:
                    du_s, dv_s = compute_dr_dz(surface, traj, u_s, v_s, z_a)
                except ValueError:
                    du_s, dv_s = du_dz_k, dv_dz_k
                
                u_p = u_s + du_s * dz_sub
                v_p = v_s + dv_s * dz_sub
                # u_p = np.clip(u_p, surface.u_min, surface.u_max)
                if hasattr(surface, 'u_min') and hasattr(surface, 'u_max'):
                    u_p = np.clip(u_p, surface.u_min, surface.u_max)
                if hasattr(surface, 'v_min') and hasattr(surface, 'v_max'):
                    v_p = np.clip(v_p, surface.v_min, surface.v_max)
                
                u_c, v_c, Phi_c, nit, conv = newton_corrector(
                    surface, traj, u_p, v_p, z_b,
                    eps_Phi=eps_Phi, max_iter=max_newton)
                total_nit += nit
                
                du_j = u_c - u_s
                dv_j = v_c - v_s
                Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                ds_actual = np.sqrt(max(Ej*du_j**2 + 2*Fj*du_j*dv_j + Gj*dv_j**2, 0.0))
                ds_expect = speed_expected * dz_sub
                
                if ds_expect > 1e-12:
                    ratio = ds_actual / ds_expect
                else:
                    ratio = 1.0
                
                if (ratio > jump_threshold or ratio < 1.0/jump_threshold) \
                        and bisect_level < max_bisect:
                    jump_detected = True
                    break
                
                u_s, v_s = u_c, v_c
            
            if not jump_detected:
                best_u, best_v = u_s, v_s
                r_f = surface.position(best_u, best_v)
                m_f = surface.normal(best_u, best_v)
                best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                best_nit = total_nit
                if bisect_level > 0:
                    bisected = True
                break
            else:
                n_sub *= 2
        
        u_cur, v_cur = best_u, best_v
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = best_Phi
        newton_iters_hist[i + 1] = best_nit
        flags[i + 1] = 1 if bisected else 0
    
    # Финальная точка
    try:
        _, lam_last, up_last, vp_last = recompute_thread_geometry(
            surface, traj, u_hist[-1], v_hist[-1], z_eval[-1])
        lam_hist[-1] = lam_last
        kappa_n_hist[-1] = normal_curvature(
            surface, u_hist[-1], v_hist[-1], up_last, vp_last)
    except ValueError:
        pass
    
    points_3d = np.array([surface.position(u_hist[k], v_hist[k]) for k in range(N)])
    return {
        'z_eval': z_eval,
        'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }

