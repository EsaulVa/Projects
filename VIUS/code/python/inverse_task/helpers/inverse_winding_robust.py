"""
Робастная обратная задача намотки с ГИБРИДНЫМ ПРЕДИКТОРОМ.
Автоматически переключается между DAE и Оптическим предиктором
в зависимости от величины градиента связи.
"""
import numpy as np
from typing import Tuple, Optional, Any

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ГЕОМЕТРИЧЕСКИЕ ФУНКЦИИ
# ======================================================================

def project_to_tangent_plane(vec_3d, normal):
    """Проекция 3D-вектора на касательную плоскость."""
    return vec_3d - np.dot(vec_3d, normal) * normal

def normal_curvature(surface, u: float, v: float, u_prime: float, v_prime: float) -> float:
    """κ_n = II(τ,τ) / I(τ,τ)."""
    L, M, N = surface.second_fundamental_form(u, v)
    E, F, G = surface.first_fundamental_form(u, v)
    II_val = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N * v_prime**2
    I_val = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
    if abs(I_val) < 1e-15:
        return 0.0
    return II_val / I_val

def compute_tangent_components(surface, u: float, v: float, tau_3d) -> Tuple[float, float]:
    """Контравариантные компоненты проекции tau_3d на T_r S."""
    derivs = surface.derivatives(u, v)
    ru, rv = derivs['ru'], derivs['rv']
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    b1 = np.dot(tau_3d, ru)
    b2 = np.dot(tau_3d, rv)
    u_prime = (G * b1 - F * b2) / det
    v_prime = (-F * b1 + E * b2) / det
    return u_prime, v_prime

def inverse_metric(surface, u: float, v: float) -> Tuple[float, float, float]:
    """g^{αβ}."""
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    return G / det, -F / det, E / det

def compute_grad_Phi(surface, u: float, v: float, u_prime: float, v_prime: float,
                     lam: float) -> Tuple[float, float]:
    """
    Ковариантные компоненты ∂Φ/∂u^α.
    Формула: -λ * b_{αβ} τ^β (согласована с внешней нормалью).
    """
    E, F, G = surface.first_fundamental_form(u, v)
    L, M, N = surface.second_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика в grad_Phi")

    # Ковариантные компоненты τ
    tau_u = E * u_prime + F * v_prime
    tau_v = F * u_prime + G * v_prime

    # grad_α Φ = -λ * (L τ^u + M τ^v) для α=u и т.д.
    dPhidu = -lam * (L * u_prime + M * v_prime)
    dPhidv = -lam * (M * u_prime + N * v_prime)
    
    return dPhidu, dPhidv

def recompute_thread_geometry(surface, traj, u: float, v: float, z: float):
    """Пересчёт τ, λ, (u', v') в точке (u, v) при параметре z."""
    r = surface.position(u, v)
    R = traj.R(z)
    delta = R - r
    lam = float(np.linalg.norm(delta))
    if lam < 1e-13:
        raise ValueError("Нулевая длина нити")
    tau_3d = delta / lam
    u_p, v_p = compute_tangent_components(surface, u, v, tau_3d)
    return tau_3d, lam, u_p, v_p

# ======================================================================
# КОРРЕКТОР НЬЮТОНА
# ======================================================================

def newton_corrector_stable(
    surface, traj, u_pred: float, v_pred: float, z_target: float,
    eps_Phi: float = 1e-6,
    max_iter: int = 20,
    max_step_u: float = 50.0,
    max_step_v: float = 1.0,
    use_mod_v: bool = True,
    verbose: bool = False
) -> Tuple[float, float, float, int, bool]:
    """Стабильный корректор Ньютона с демпфированием и backtracking."""
    u_c, v_c = float(u_pred), float(v_pred)
    
    u_min = getattr(surface, 'u_min', -np.inf)
    u_max = getattr(surface, 'u_max', np.inf)
    
    r_c = surface.position(u_c, v_c)
    m_c = surface.normal(u_c, v_c)
    R_t = traj.R(z_target)
    Phi_cur = float(np.dot(R_t - r_c, m_c))

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

        alpha = 1.0
        success_step = False
        
        for _ in range(12):
            step_u = -alpha * Phi_cur / Ng * dir_u
            step_v = -alpha * Phi_cur / Ng * dir_v
            
            step_u = np.clip(step_u, -max_step_u, max_step_u)
            step_v = np.clip(step_v, -max_step_v, max_step_v)

            u_try = np.clip(u_c + step_u, u_min, u_max)
            v_try = np.mod(v_c + step_v, 2 * np.pi) if use_mod_v else np.clip(v_c + step_v, 0, 2*np.pi)

            try:
                r_try = surface.position(u_try, v_try)
                m_try = surface.normal(u_try, v_try)
                Phi_try = float(np.dot(R_t - r_try, m_try))
            except Exception:
                alpha *= 0.5
                continue

            if abs(Phi_try) < eps_Phi:
                return u_try, v_try, Phi_try, nit, True
                
            if abs(Phi_try) < abs(Phi_cur):
                u_c, v_c = u_try, v_try
                r_c, m_c = r_try, m_try
                Phi_cur = Phi_try
                success_step = True
                break
                
            alpha *= 0.5
            
        if not success_step:
            pass

    return u_c, v_c, Phi_cur, max_iter, abs(Phi_cur) < eps_Phi

# ======================================================================
# DAE ПРЕДИКТОР
# ======================================================================

def compute_dr_dz(surface, traj, u: float, v: float, z: float) -> Tuple[float, float, float]:
    """
    Аналитическое du/dz, dv/dz.
    Возвращает также norm_grad_sq для принятия решения о переключении.
    """
    r = surface.position(u, v)
    R = traj.R(z)
    m = surface.normal(u, v)
    delta = R - r
    lam = np.linalg.norm(delta)
    if lam < 1e-13:
        return 0.0, 0.0, 0.0

    tau = delta / lam
    R_prime = traj.R_deriv(z)
    dPhi_dz = float(np.dot(R_prime, m))
    
    R_prime_par = project_to_tangent_plane(R_prime, m)
    Rp_u, Rp_v = compute_tangent_components(surface, u, v, R_prime_par)
    
    up, vp = compute_tangent_components(surface, u, v, tau)
    dPhi_du, dPhi_dv = compute_grad_Phi(surface, u, v, up, vp, lam)
    
    residual = dPhi_dz + dPhi_du * Rp_u + dPhi_dv * Rp_v
    
    guu, guv, gvv = inverse_metric(surface, u, v)
    grad_u = guu * dPhi_du + guv * dPhi_dv
    grad_v = guv * dPhi_du + gvv * dPhi_dv
    
    norm_grad_sq = guu * dPhi_du**2 + 2 * guv * dPhi_du * dPhi_dv + gvv * dPhi_dv**2
    
    if norm_grad_sq < 1e-14:
        return Rp_u, Rp_v, norm_grad_sq
        
    mu = -residual / norm_grad_sq
    du_dz = Rp_u + mu * grad_u
    dv_dz = Rp_v + mu * grad_v
    
    return du_dz, dv_dz, norm_grad_sq

# ======================================================================
# ПОИСК НАЧАЛЬНОЙ ТОЧКИ
# ======================================================================

def find_valid_initial_point(surface, traj, z_start=0.0, num_attempts=20):
    """Поиск начальной точки перебором v."""
    best_u, best_v, best_Phi = None, None, np.inf
    found = False
    
    for v_try in np.linspace(0, 2*np.pi, num_attempts):
        u_try = traj.R(z_start)[2] 
        try:
            u_f, v_f, Phi_f, _, conv = newton_corrector_stable(
                surface, traj, u_try, v_try, z_start, eps_Phi=1e-6, max_iter=30
            )
            if conv and abs(Phi_f) < abs(best_Phi):
                best_u, best_v, best_Phi = u_f, v_f, Phi_f
                found = True
                if abs(Phi_f) < 1e-8: break
        except Exception:
            continue
            
    if not found:
        raise RuntimeError("Не удалось найти начальную точку")
    return best_u, best_v, best_Phi, found

# ======================================================================
# ОСНОВНОЙ ЦИКЛ С ГИБРИДНЫМ ПРЕДИКТОРОМ
# ======================================================================

def inverse_winding_robust(
    surface, traj, u0: float, v0: float,
    optical_predictor=None,      # <-- ОБЯЗАТЕЛЬНЫЙ ПАРАМЕТР ДЛЯ ГИБРИДА
    count_points: int = 200,
    eps_Phi: float = 1e-6,
    max_newton: int = 20,
    max_bisect: int = 8,         
    jump_threshold: float = 2.0, 
    grad_threshold: float = 1e-4,# Порог переключения на оптику
    verbose: bool = False
) -> dict:
    """
    Робастная обратная задача с гибридным предиктором.
    """
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)
    
    u_hist = np.zeros(N); v_hist = np.zeros(N)
    Phi_hist = np.zeros(N); kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    flags = np.zeros(N, dtype=int)
    predictor_type = [''] * N 
    
    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0
    
    surf_u_min = getattr(surface, 'u_min', -np.inf)
    surf_u_max = getattr(surface, 'u_max', np.inf)
    
    print(f"\n=== Обратная задача (Гибрид): {N} точек ===")
    print(f"Параметры: eps={eps_Phi}, bisect={max_bisect}, jump_thr={jump_threshold}")
    print(f"Оптический предиктор: {'ВКЛ' if optical_predictor else 'ВЫКЛ'}")

    for i in range(N - 1):
        z_k, z_next = z_eval[i], z_eval[i+1]
        dz = z_next - z_k
        
        # 1. Геометрия
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(surface, traj, u_cur, v_cur, z_k)
            kappa_n_hist[i] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)
        except ValueError:
            break

            # --- 2. ГИБРИДНЫЙ ПРЕДИКТОР ---
        u_pred, v_pred = None, None
        used_optical = False
        
        # Проверяем невязку на предыдущем шаге. Если она велика — доверяем только оптике.
        Phi_prev = Phi_hist[i] if i > 0 else 0.0
        
        # Сначала пробуем DAE
        try:
            du_dz, dv_dz, ng_sq = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
            
            # УСЛОВИЕ ПЕРЕКЛЮЧЕНИЯ:
            # 1. Градиент мал
            # 2. Невязка велика (Phi_prev > 1e-3) — значит, мы потерялись, нужен глобальный поиск
            # 3. Или просто каждый 10-й шаг для страховки
            if ng_sq < grad_threshold or abs(Phi_prev) > 1e-6 or (optical_predictor is not None and i % 10 == 0):
                if optical_predictor is not None:
                    try:
                        res = optical_predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj)
                        if res is not None:
                            u_pred, v_pred = res
                            used_optical = True
                            predictor_type[i] = 'OPT_GLOBAL' if abs(Phi_prev) > 1e-4 else 'OPT'
                    except Exception:
                        pass
                        
            if not used_optical:
                u_pred = u_cur + du_dz * dz
                v_pred = v_cur + dv_dz * dz
                predictor_type[i] = 'DAE'
                
        except Exception:
            if optical_predictor is not None:
                try:
                    res = optical_predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj)
                    if res is not None:
                        u_pred, v_pred = res
                        used_optical = True
                        predictor_type[i] = 'OPT_FALLBACK'
                except Exception:
                    pass

        if u_pred is not None:
            v_pred = np.mod(v_pred, 2 * np.pi)
            u_pred = np.clip(u_pred, surf_u_min, surf_u_max)   
        # # ================================================================
        # # 2. ГИБРИДНЫЙ ПРЕДИКТОР (ВОССТАНОВЛЕННАЯ ЛОГИКА)
        # # ================================================================
        # u_pred, v_pred = None, None
        # used_optical = False
        
        # # Сначала всегда пробуем DAE, чтобы оценить градиент
        # try:
        #     du_dz, dv_dz, ng_sq = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
            
        #     # РЕШЕНИЕ О ПЕРЕКЛЮЧЕНИИ:
        #     # Если градиент мал ИЛИ оптический предиктор включен (для страховки)
        #     if ng_sq < grad_threshold or (optical_predictor is not None):
        #         if optical_predictor is not None:
        #             try:
        #                 res = optical_predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj)
        #                 if res is not None:
        #                     u_pred, v_pred = res
        #                     used_optical = True
        #                     predictor_type[i] = 'OPT'
        #             except Exception:
        #                 pass
                    
        #     # Если оптика не сработала или не нужна — используем DAE
        #     if not used_optical:
        #         u_pred = u_cur + du_dz * dz
        #         v_pred = v_cur + dv_dz * dz
        #         predictor_type[i] = 'DAE'
                
        # except Exception:
        #     # Fallback на оптику при ошибке DAE
        #     if optical_predictor is not None:
        #         try:
        #             res = optical_predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj)
        #             if res is not None:
        #                 u_pred, v_pred = res
        #                 used_optical = True
        #                 predictor_type[i] = 'OPT_FALLBACK'
        #         except Exception:
        #             pass

        # if u_pred is not None:
        #     v_pred = np.mod(v_pred, 2 * np.pi)
        #     u_pred = np.clip(u_pred, surf_u_min, surf_u_max)

        # ================================================================
        # 3. КОРРЕКТОР
        # ================================================================
        success = False
        if u_pred is not None:
            try:
                u_c, v_c, Phi_c, nit, conv = newton_corrector_stable(
                    surface, traj, u_pred, v_pred, z_next,
                    eps_Phi=eps_Phi, max_iter=max_newton
                )
                newton_iters_hist[i+1] = nit
                
                if conv or abs(Phi_c) < 1e-3:
                    # Проверка скачка
                    du_j, dv_j = u_c - u_cur, v_c - v_cur
                    E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
                    ds_act = np.sqrt(max(E_f*du_j**2 + 2*F_f*du_j*dv_j + G_f*dv_j**2, 0))
                    
                    try:
                        spd = np.sqrt(max(E_f*du_dz**2 + 2*F_f*du_dz*dv_dz + G_f*dv_dz**2, 0))
                    except: 
                        spd = 1.0
                        
                    ds_exp = spd * dz if spd > 1e-12 else 1.0
                    ratio = ds_act / ds_exp if ds_exp > 1e-12 else 1.0
                    
                    if ratio < jump_threshold:
                        u_cur, v_cur = u_c, v_c
                        success = True
            except Exception:
                pass

        # ================================================================
        # 4. БИСЕКЦИЯ (Fallback)
        # ================================================================
        if not success:
            flags[i+1] = 1
            n_sub = 1
            best_u, best_v, best_Phi = u_cur, v_cur, 1e9
            
            for level in range(max_bisect + 1):
                sub_z = np.linspace(z_k, z_next, n_sub + 1)
                u_s, v_s = u_cur, v_cur
                failed = False
                
                for j in range(n_sub):
                    za, zb = sub_z[j], sub_z[j+1]
                    dz_sub = zb - za
                    
                    try:
                        du_s, dv_s, _ = compute_dr_dz(surface, traj, u_s, v_s, za)
                    except:
                        du_s, dv_s = 0.0, 0.0
                        
                    up = np.clip(u_s + du_s*dz_sub, surf_u_min, surf_u_max)
                    vp = np.mod(v_s + dv_s*dz_sub, 2*np.pi)
                    
                    uc, vc, Pc, _, _ = newton_corrector_stable(
                        surface, traj, up, vp, zb, eps_Phi=eps_Phi, max_iter=max_newton
                    )
                    
                    E_j, F_j, G_j = surface.first_fundamental_form(u_s, v_s)
                    dj = np.sqrt(max(E_j*(uc-u_s)**2 + 2*F_j*(uc-u_s)*(vc-v_s) + G_j*(vc-v_s)**2, 0))
                    exp_s = np.sqrt(max(E_j*du_s**2 + 2*F_j*du_s*dv_s + G_j*dv_s**2, 0)) * dz_sub
                    if exp_s < 1e-12: exp_s = 1.0
                    
                    if (dj / exp_s) > jump_threshold and level < max_bisect:
                        failed = True
                        break
                    u_s, v_s = uc, vc
                    
                if not failed:
                    best_u, best_v = u_s, v_s
                    try:
                        rf = surface.position(best_u, best_v)
                        mf = surface.normal(best_u, best_v)
                        best_Phi = float(np.dot(traj.R(z_next) - rf, mf))
                    except: pass
                    break
                n_sub *= 2
                
            u_cur, v_cur = best_u, best_v
            predictor_type[i] += '_BISECT'

        u_hist[i+1], v_hist[i+1] = u_cur, v_cur
        Phi_hist[i+1] = best_Phi if not success else Phi_c
        
        if verbose and i % 20 == 0:
            print(f"[{i:3d}] z={z_k:.1f} | {predictor_type[i]:12s} | Φ={Phi_hist[i+1]:.2e}")

    # Финализация
    points_3d = np.array([surface.position(u_hist[k], v_hist[k]) for k in range(N)])
    
    print(f"\n=== Результат ===")
    print(f"Успешно: {np.sum(flags==0)}/{N-1} | Бисекция: {np.sum(flags==1)}/{N-1}")
    valid = np.abs(Phi_hist) < 1e10
    if np.any(valid):
        print(f"Макс |Φ|: {np.max(np.abs(Phi_hist[valid])):.2e}")
        
    return {
        'z_eval': z_eval, 'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist, 'flags': flags,
        'points_3d': points_3d, 'predictor_type': predictor_type
    }