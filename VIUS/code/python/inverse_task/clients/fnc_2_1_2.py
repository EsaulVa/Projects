# ======================================================================
# Обратная задача намотки — v3
# Аналитический предиктор + корректор Ньютона + бисекция + диагностика
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================================================================
# 1. КЛАССЫ ГЕОМЕТРИИ
# ======================================================================

class Ellipsoid:
    """
    Эллипсоид с полуосями (a, b, c).
    Параметризация: u = θ ∈ [0, π], v = φ ∈ [0, 2π)
    
    r(θ, φ) = (a sinθ cosφ,  b sinθ sinφ,  c cosθ)
    """
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c
    
    def position(self, u, v):
        st, ct = np.sin(u), np.cos(u)
        sp, cp = np.sin(v), np.cos(v)
        return np.array([self.a * st * cp,
                         self.b * st * sp,
                         self.c * ct])
    
    def derivatives(self, u, v):
        st, ct = np.sin(u), np.cos(u)
        sp, cp = np.sin(v), np.cos(v)
        ru = np.array([self.a * ct * cp,
                       self.b * ct * sp,
                       -self.c * st])
        rv = np.array([-self.a * st * sp,
                        self.b * st * cp,
                        0.0])
        return {'ru': ru, 'rv': rv}
    
    def normal(self, u, v):
        d = self.derivatives(u, v)
        n = np.cross(d['ru'], d['rv'])
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            return np.array([0.0, 0.0, 1.0])
        return n / nn
    
    def first_fundamental_form(self, u, v):
        d = self.derivatives(u, v)
        ru, rv = d['ru'], d['rv']
        E = np.dot(ru, ru)
        F = np.dot(ru, rv)
        G = np.dot(rv, rv)
        return E, F, G
    
    def second_fundamental_form(self, u, v):
        st, ct = np.sin(u), np.cos(u)
        sp, cp = np.sin(v), np.cos(v)
        a, b, c = self.a, self.b, self.c
        
        ruu = np.array([-a * st * cp, -b * st * sp, -c * ct])
        ruv = np.array([-a * ct * sp,  b * ct * cp,  0.0])
        rvv = np.array([-a * st * cp, -b * st * sp,  0.0])
        
        m = self.normal(u, v)
        L = np.dot(ruu, m)
        M = np.dot(ruv, m)
        N = np.dot(rvv, m)
        return L, M, N


# ======================================================================
# 2. ГЕОДЕЗИЧЕСКАЯ НА ЭЛЛИПСОИДЕ (уравнения Клеро + числ. интегрирование)
# ======================================================================

def geodesic_on_ellipsoid(ell, u0, v0, alpha0, total_length, n_points=500):
    """
    Геодезическая на эллипсоиде методом Рунге-Кутты 4.
    
    Состояние: (u, v, ψ) где ψ — угол между касательной и ∂r/∂u.
    
    Уравнения геодезической через символы Кристоффеля.
    
    alpha0 — начальный угол намотки (угол с меридианом).
    
    Возвращает массивы (s, u, v) и 3D-точки.
    """
    def christoffel(ell, u, v):
        """Символы Кристоффеля Γ^k_{ij} через конечные разности."""
        h = 1e-7
        E, F, G = ell.first_fundamental_form(u, v)
        
        Eu, _, _ = ell.first_fundamental_form(u + h, v)
        Eu2, _, _ = ell.first_fundamental_form(u - h, v)
        dE_du = (Eu - Eu2) / (2 * h)
        
        _, _, Gu = ell.first_fundamental_form(u + h, v)
        _, _, Gu2 = ell.first_fundamental_form(u - h, v)
        dG_du = (Gu - Gu2) / (2 * h)
        
        Ev, _, _ = ell.first_fundamental_form(u, v + h)
        Ev2, _, _ = ell.first_fundamental_form(u, v - h)
        dE_dv = (Ev - Ev2) / (2 * h)
        
        _, _, Gv = ell.first_fundamental_form(u, v + h)
        _, _, Gv2 = ell.first_fundamental_form(u, v - h)
        dG_dv = (Gv - Gv2) / (2 * h)
        
        _, Fu, _ = ell.first_fundamental_form(u + h, v)
        _, Fu2, _ = ell.first_fundamental_form(u - h, v)
        dF_du = (Fu - Fu2) / (2 * h)
        
        _, Fv, _ = ell.first_fundamental_form(u, v + h)
        _, Fv2, _ = ell.first_fundamental_form(u, v - h)
        dF_dv = (Fv - Fv2) / (2 * h)
        
        det = E * G - F * F
        if abs(det) < 1e-15:
            return np.zeros((2, 2, 2))
        
        Gamma = np.zeros((2, 2, 2))
        # Γ^0_{ij}
        Gamma[0, 0, 0] = (G * dE_du - 2 * F * dF_du + F * dE_dv) / (2 * det)
        Gamma[0, 0, 1] = (G * dE_dv - F * dG_du) / (2 * det)
        Gamma[0, 1, 0] = Gamma[0, 0, 1]
        Gamma[0, 1, 1] = (2 * G * dF_dv - G * dG_du - F * dG_dv) / (2 * det)
        # Γ^1_{ij}
        Gamma[1, 0, 0] = (2 * E * dF_du - E * dE_dv - F * dE_du) / (2 * det)
        Gamma[1, 0, 1] = (E * dG_du - F * dE_dv) / (2 * det)
        Gamma[1, 1, 0] = Gamma[1, 0, 1]
        Gamma[1, 1, 1] = (E * dG_dv - 2 * F * dF_dv + F * dG_du) / (2 * det)
        
        return Gamma
    
    # Начальные контравариантные компоненты скорости
    E0, F0, G0 = ell.first_fundamental_form(u0, v0)
    # Единичный вектор: |dr/ds|² = E(du/ds)² + 2F(du/ds)(dv/ds) + G(dv/ds)² = 1
    # Угол alpha0 с меридианом (∂r/∂u направление):
    # du/ds = cosα / √E,  dv/ds = (sinα - F cosα / (√E √G)) * ... 
    # Упрощение для F ≈ 0 (ортогональная параметризация):
    du_ds = np.cos(alpha0) / np.sqrt(E0)
    dv_ds = np.sin(alpha0) / np.sqrt(G0)
    
    # Состояние: y = [u, v, du/ds, dv/ds]
    y = np.array([u0, v0, du_ds, dv_ds])
    
    ds = total_length / n_points
    s_arr = np.zeros(n_points + 1)
    u_arr = np.zeros(n_points + 1)
    v_arr = np.zeros(n_points + 1)
    
    u_arr[0], v_arr[0] = u0, v0
    
    def rhs(y):
        u, v, du, dv = y
        G = christoffel(ell, u, v)
        ddu = -(G[0, 0, 0] * du**2 + 2 * G[0, 0, 1] * du * dv + G[0, 1, 1] * dv**2)
        ddv = -(G[1, 0, 0] * du**2 + 2 * G[1, 0, 1] * du * dv + G[1, 1, 1] * dv**2)
        return np.array([du, dv, ddu, ddv])
    
    for i in range(n_points):
        # RK4
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * ds * k1)
        k3 = rhs(y + 0.5 * ds * k2)
        k4 = rhs(y + ds * k3)
        y = y + ds / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        
        s_arr[i + 1] = s_arr[i] + ds
        u_arr[i + 1] = y[0]
        v_arr[i + 1] = y[1]
    
    # 3D-точки
    pts = np.array([ell.position(u_arr[i], v_arr[i]) for i in range(n_points + 1)])
    
    return s_arr, u_arr, v_arr, pts


# ======================================================================
# 3. КЛАСС ТРАЕКТОРИИ (интерполяция 3D-кривой по длине дуги)
# ======================================================================

class Trajectory3D:
    """
    Траектория R(z), параметризованная длиной дуги z.
    Построена из массива 3D-точек.
    """
    def __init__(self, points):
        """points: (N, 3) массив."""
        self.points = np.array(points, dtype=float)
        N = len(self.points)
        
        # Длины дуг
        diffs = np.diff(self.points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.s = np.zeros(N)
        self.s[1:] = np.cumsum(seg_lengths)
        self.total_length = self.s[-1]
        
        # Кубическая интерполяция покомпонентно
        from scipy.interpolate import CubicSpline
        self._interp_x = CubicSpline(self.s, self.points[:, 0])
        self._interp_y = CubicSpline(self.s, self.points[:, 1])
        self._interp_z = CubicSpline(self.s, self.points[:, 2])
    
    def R(self, z):
        z = np.clip(z, 0, self.total_length)
        return np.array([self._interp_x(z),
                         self._interp_y(z),
                         self._interp_z(z)])
    
    def R_deriv(self, z):
        z = np.clip(z, 0, self.total_length)
        return np.array([self._interp_x(z, 1),
                         self._interp_y(z, 1),
                         self._interp_z(z, 1)])


# ======================================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАТНОЙ ЗАДАЧИ
# ======================================================================

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

def newton_corrector(surface, traj, u_pred, v_pred, z_target,
                     eps_Phi=1e-10, max_iter=7):
    """
    Корректор Ньютона: проецирует (u_pred, v_pred) на Φ = 0.
    Возвращает (u, v, Phi, n_iter, converged).
    """
    u_c, v_c = u_pred, v_pred
    
    for nit in range(max_iter):
        r_c = surface.position(u_c, v_c)
        m_c = surface.normal(u_c, v_c)
        R_t = traj.R(z_target)
        delta_c = R_t - r_c
        lam_c = np.linalg.norm(delta_c)
        Phi_c = np.dot(delta_c, m_c)
        
        if abs(Phi_c) < eps_Phi:
            return u_c, v_c, Phi_c, nit, True
        
        if lam_c < 1e-13:
            return u_c, v_c, Phi_c, nit, False
        
        tau_c = delta_c / lam_c
        try:
            up_c, vp_c = compute_tangent_components(surface, u_c, v_c, tau_c)
            dPdu, dPdv = compute_grad_Phi(surface, u_c, v_c, up_c, vp_c, lam_c)
            guu, guv, gvv = inverse_metric(surface, u_c, v_c)
        except ValueError:
            return u_c, v_c, Phi_c, nit, False
        
        Ng = (guu * dPdu**2 + 2 * guv * dPdu * dPdv + gvv * dPdv**2)
        
        if Ng < 1e-14:
            return u_c, v_c, Phi_c, nit, False
        
        u_c -= Phi_c / Ng * (guu * dPdu + guv * dPdv)
        v_c -= Phi_c / Ng * (guv * dPdu + gvv * dPdv)
    
    # Финальная невязка
    r_c = surface.position(u_c, v_c)
    m_c = surface.normal(u_c, v_c)
    Phi_c = np.dot(traj.R(z_target) - r_c, m_c)
    return u_c, v_c, Phi_c, max_iter, abs(Phi_c) < eps_Phi


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


# ======================================================================
# 7. ПАРАМЕТРЫ ЗАДАЧИ
# ======================================================================

# # Оправка (внутренний эллипсоид)
# a2, b2, c2 = 2.0, 1.5, 1.0
# E2 = Ellipsoid(a2, b2, c2)

# # Внешний эллипсоид (раскладчик движется по его геодезической)
# a1, b1, c1 = 3.0, 2.5, 2.0
# E1 = Ellipsoid(a1, b1, c1)

# # Начальная точка на внешнем эллипсоиде
# u0_ext = np.pi / 30       # θ
# v0_ext = np.pi / 60        # φ
# alpha_wind = np.pi / 6    # угол намотки

# # Геодезическая на E1 — траектория раскладчика
# n_traj_points = 1000
# total_geod_length = 25.0

# print("Вычисление геодезической на внешнем эллипсоиде...")
# s_geod, u_geod, v_geod, pts_geod = geodesic_on_ellipsoid(
#     E1, u0_ext, v0_ext, alpha_wind, total_geod_length, n_traj_points
# )
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
# ----------------------------------------------------------------------
# 1. Параметры эллипсоидов и прямая задача
# ----------------------------------------------------------------------
a1, b1, c1 = 2.0, 2.5, 5.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)
# scale = 0.8
# a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
# E2 = EllipsoidWithDerivatives(a2, b2, c2)
# Внутренний баллон E2 (цилиндр + полусферы)
R_cyl = 1
L_cyl = 6
z_cyl_min = -L_cyl / 2
z_cyl_max =  L_cyl / 2

cyl_seg = CylinderSegment(R_cyl, z_cyl_min, z_cyl_max)
lower_sphere = SphereSegment(R_cyl, z_cyl_min, is_upper=False)
upper_sphere = SphereSegment(R_cyl, z_cyl_max, is_upper=True)
E2 = CompositeSurface([lower_sphere, cyl_seg, upper_sphere])
print("===== Прямая задача =====")
deviation_law = ConstantDeviation(tan_theta=0)
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
forward_builder = ForwardWindingBuilder(
    surface=E1, deviation_law=deviation_law,
    solver=solver_forward, normalize_tangent=True, eps=1e-12
)
u0, v0 = 70*np.pi/180, -np.pi/6
alpha = np.pi / 6
s_end = 30.0
count_points = 200
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

# Траектория раскладчика
# traj = Trajectory3D(line_E1)
print(f"Длина траектории: {traj.total_length:.4f}")

# Начальная точка на оправке E2:
# Проецируем начальную точку траектории на E2 (те же угловые координаты)
u0_mandrel = u0
v0_mandrel = v0

# Проверка: начальная точка должна удовлетворять Φ ≈ 0
r0 = E2.position(u0_mandrel, v0_mandrel)
R0 = traj.R(0.0)
m0 = E2.normal(u0_mandrel, v0_mandrel)
Phi0 = np.dot(R0 - r0, m0)
print(f"Начальная невязка Φ₀ = {Phi0:.6e}")

# Если Φ₀ ≠ 0, корректируем начальную точку
if abs(Phi0) > 1e-8:
    print("Корректировка начальной точки...")
    u0_mandrel, v0_mandrel, Phi0_corr, _, conv0 = newton_corrector(
        E2, traj, u0_mandrel, v0_mandrel, 0.0,
        eps_Phi=1e-12, max_iter=20
    )
    print(f"  После коррекции: Φ₀ = {Phi0_corr:.6e}, сошёлся: {conv0}")


# ======================================================================
# 8. ЗАПУСК ОБРАТНОЙ ЗАДАЧИ
# ======================================================================

count_points = 300

print(f"\n===== Обратная задача v3 ({count_points} точек) =====")

result = inverse_winding_v3(
    E2, traj, u0_mandrel, v0_mandrel,
    count_points=count_points,
    eps_Phi=1e-10,
    max_newton=7,
    max_bisect=4,
    jump_threshold=3.0
)
pts_geod=line_E1
z_eval = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
lam_hist = result['lam']
flags = result['flags']
line_E2 = result['points_3d']

n_bisected = np.sum(flags == 1)
print(f"\nШагов с бисекцией: {n_bisected} из {count_points - 1}")
print(f"Максимальная невязка |Φ|: {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя невязка |Φ|:      {np.mean(np.abs(Phi_hist)):.2e}")
print(f"Среднее итераций Ньютона:  {np.mean(newton_iters_hist[1:]):.2f}")
print(f"Максимум итераций Ньютона: {np.max(newton_iters_hist[1:])}")


# ======================================================================
# 9. ВИЗУАЛИЗАЦИЯ 3D
# ======================================================================

def plot_ellipsoid_wireframe(ax, ell, color='blue', alpha=0.1, label=None):
    """Каркас эллипсоида."""
    u_grid = np.linspace(0, np.pi, 30)
    v_grid = np.linspace(0, 2 * np.pi, 40)
    U, V = np.meshgrid(u_grid, v_grid)
    X = ell.a * np.sin(U) * np.cos(V)
    Y = ell.b * np.sin(U) * np.sin(V)
    Z = ell.c * np.cos(U)
    ax.plot_surface(X, Y, Z, alpha=alpha, color=color, edgecolor='gray',
                    linewidth=0.2)
    if label:
        # Фиктивная линия для легенды
        ax.plot([], [], [], color=color, label=label)

# ----------------------------------------------------------------------
# 5. Визуализация
# ----------------------------------------------------------------------
print("\n===== Построение 3D-графика =====")

def surface_grid(surf, u_arr, v_arr):
    U, V = np.meshgrid(u_arr, v_arr)
    X, Y, Z = np.zeros_like(U), np.zeros_like(U), np.zeros_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            p = surf.position(U[i,j], V[i,j])
            X[i,j], Y[i,j], Z[i,j] = np.array(p)
    return X, Y, Z

u_grid = np.linspace(0, 2*np.pi, 80)
v_grid_E1 = np.linspace(-np.pi/2, np.pi/2, 50)
v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 80)

X1, Y1, Z1 = surface_grid(E1, u_grid, v_grid_E1)
X2, Y2, Z2 = surface_grid(E2, u_grid, v_grid_E2)

fig = go.Figure()
fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.2, colorscale='Blues', showscale=False, name='E1'))
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False, name='Баллон E2'))

# fig.add_trace(go.Scatter3d(x=line_E1[:,0], y=line_E1[:,1], z=line_E1[:,2],
#                            mode='lines', line=dict(color='blue', width=4), name='Траектория R(z)'))
# fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
#                            mode='lines', line=dict(color='red', width=4), name='Линия укладки на E2'))
from scipy.interpolate import CubicSpline
# line_E1 – массив точек (N,3)
# создаём сплайн по накопленной длине дуги вдоль линии
dist = np.zeros(len(line_E1))
dist[1:] = np.linalg.norm(np.diff(line_E1, axis=0), axis=1).cumsum()

cs_x = CubicSpline(dist, line_E1[:,0])
cs_y = CubicSpline(dist, line_E1[:,1])
cs_z = CubicSpline(dist, line_E1[:,2])

# генерируем в 5-10 раз больше точек для плавной картинки
dense_dist = np.linspace(dist[0], dist[-1], len(line_E1)*10)
smooth_x = cs_x(dense_dist)
smooth_y = cs_y(dense_dist)
smooth_z = cs_z(dense_dist)

# теперь рисуем сглаженную версию
fig.add_trace(go.Scatter3d(
    x=smooth_x, y=smooth_y, z=smooth_z,
   mode='lines', line=dict(color='blue', width=4), name='Траектория R(z)'))
# line_E2 – массив точек (N,3)
# создаём сплайн по накопленной длине дуги вдоль линии
dist = np.zeros(len(line_E2))
dist[1:] = np.linalg.norm(np.diff(line_E2, axis=0), axis=1).cumsum()

cs_x = CubicSpline(dist, line_E2[:,0])
cs_y = CubicSpline(dist, line_E2[:,1])
cs_z = CubicSpline(dist, line_E2[:,2])

# генерируем в 5-10 раз больше точек для плавной картинки
dense_dist = np.linspace(dist[0], dist[-1], len(line_E2)*10)
smooth_x = cs_x(dense_dist)
smooth_y = cs_y(dense_dist)
smooth_z = cs_z(dense_dist)

# теперь рисуем сглаженную версию
fig.add_trace(go.Scatter3d(
    x=smooth_x, y=smooth_y, z=smooth_z,
    mode='lines',
    line=dict(color='red', width=4),
    name='Линия укладки на E2'
))

# Начальные и конечные точки
# E1 (траектория)
start_E1 = line_E1[0]
end_E1   = line_E1[-1]
fig.add_trace(go.Scatter3d(x=[start_E1[0]], y=[start_E1[1]], z=[start_E1[2]],
                           mode='markers', marker=dict(color='green', size=4, symbol='circle'),
                           name='Начало R(z)'))
fig.add_trace(go.Scatter3d(x=[end_E1[0]], y=[end_E1[1]], z=[end_E1[2]],
                           mode='markers', marker=dict(color='black', size=4, symbol='x'),
                           name='Конец R(z)'))

# E2 (линия укладки)
start_E2 = line_E2[0]
end_E2   = line_E2[-1]
fig.add_trace(go.Scatter3d(x=[start_E2[0]], y=[start_E2[1]], z=[start_E2[2]],
                           mode='markers', marker=dict(color='lime', size=4, symbol='diamond'),
                           name='Старт укладки'))
fig.add_trace(go.Scatter3d(x=[end_E2[0]], y=[end_E2[1]], z=[end_E2[2]],
                           mode='markers', marker=dict(color='orange', size=4, symbol='cross'),
                           name='Финиш укладки'))

step = 5
for i in range(0, len(line_E1[:,0]), step):
    fig.add_trace(go.Scatter3d(x=[line_E1[i,0], line_E2[i,0]],
                               y=[line_E1[i,1], line_E2[i,1]],
                               z=[line_E1[i,2], line_E2[i,2]],
                               mode='lines', line=dict(color='green', width=2), showlegend=False))

fig.update_layout(title='Эллипсоид → баллон (прямая + обратная задача)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=2000, height=1600)
# fig.show()
fig.write_html('winding_analytic.html')
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111, projection='3d')

# # Эллипсоиды
# # plot_ellipsoid_wireframe(ax, E1, color='lightblue', alpha=0.08,
# #                          label=f'E1 (внешний) {a1}×{b1}×{c1}')
# # plot_ellipsoid_wireframe(ax, E2, color='lightsalmon', alpha=0.15,
# #                          label=f'E2 (оправка) {a2}×{b2}×{c2}')

# # Траектория раскладчика (геодезическая на E1)
# ax.plot(pts_geod[:, 0], pts_geod[:, 1], pts_geod[:, 2],
#         'b-', linewidth=2.0, label='Исходная линия (траектория)')

# # Восстановленная линия на E2
# ax.plot(line_E2[:, 0], line_E2[:, 1], line_E2[:, 2],
#         'r-', linewidth=2.0, label='Восстановленная линия')

# # Начальные точки
# ax.scatter(*traj.R(0.0), color='black', s=60, zorder=5)
# ax.scatter(*E2.position(u0_mandrel, v0_mandrel), color='black', s=60, zorder=5)
# ax.scatter(*traj.R(0.0), color='black', s=60, label='Начальные точки')

# # Несколько нитей (связь R ↔ r)
# n_threads = 10
# thread_indices = np.linspace(0, len(z_eval) - 1, n_threads, dtype=int)
# for idx in thread_indices:
#     R_pt = traj.R(z_eval[idx])
#     r_pt = line_E2[idx]
#     ax.plot([R_pt[0], r_pt[0]], [R_pt[1], r_pt[1]], [R_pt[2], r_pt[2]],
#             'g-', linewidth=0.8, alpha=0.5)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Обратная задача намотки v3')
# ax.legend(loc='upper right')

# # Равные масштабы осей
# max_range = max(a1, b1, c1) * 1.1
# ax.set_xlim(-max_range, max_range)
# ax.set_ylim(-max_range, max_range)
# ax.set_zlim(-max_range, max_range)

# plt.tight_layout()
# plt.show()


# ======================================================================
# 10. ДИАГНОСТИКА
# ======================================================================

fig_diag, axes = plt.subplots(2, 3, figsize=(18, 10))

# Невязка Φ (лог)
axes[0, 0].semilogy(z_eval[1:], np.abs(Phi_hist[1:]) + 1e-16, 'b-', lw=1.2)
axes[0, 0].axhline(y=1e-10, color='red', ls=':', lw=0.8, label='ε = 1e-10')
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('|Φ|')
axes[0, 0].set_title('Невязка |Φ| (лог. шкала)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Итерации Ньютона
dz_bar = 0.8 * (z_eval[1] - z_eval[0]) if len(z_eval) > 1 else 0.1
axes[0, 1].bar(z_eval[1:], newton_iters_hist[1:],
               width=dz_bar, color='steelblue', alpha=0.7)
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('Итерации')
axes[0, 1].set_title('Итерации Ньютона')
axes[0, 1].grid(True, axis='y')

# Флаги бисекции
axes[0, 2].bar(z_eval[1:], flags[1:],
               width=dz_bar, color='orange', alpha=0.7)
axes[0, 2].set_xlabel('z')
axes[0, 2].set_ylabel('Бисекция')
axes[0, 2].set_title('Бисекция (1 = да)')
axes[0, 2].grid(True, axis='y')

# κ_n
axes[1, 0].plot(z_eval, kappa_n_hist, 'g-', lw=1.2)
axes[1, 0].axhline(y=0, color='gray', ls='--', lw=0.5)
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('κ_n')
axes[1, 0].set_title('Нормальная кривизна')
axes[1, 0].grid(True)

# λ
axes[1, 1].plot(z_eval, lam_hist, 'm-', lw=1.2)
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('λ')
axes[1, 1].set_title('Длина свободного участка нити')
axes[1, 1].grid(True)

# u(z), v(z)
axes[1, 2].plot(z_eval, u_hist, 'r-', lw=1.2, label='u(z)')
axes[1, 2].plot(z_eval, v_hist, 'b-', lw=1.2, label='v(z)')
axes[1, 2].set_xlabel('z')
axes[1, 2].set_title('Параметры u(z), v(z)')
axes[1, 2].legend()
axes[1, 2].grid(True)

fig_diag.suptitle('Диагностика обратной задачи v3', fontsize=14)
plt.tight_layout()
plt.show()

print("\n===== Готово =====")
