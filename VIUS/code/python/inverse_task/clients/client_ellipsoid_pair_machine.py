#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_ellipsoid_pair_machine.py
================================
Интеграция fnc_2_1.py (пара эллипсоидов с собственной inverse_winding_v3)
с пространственной развёрткой станка (логика client_machine.py).

1. Прямая задача: геодезическая на E1 через ForwardWindingBuilder.
2. Обратная задача: собственная inverse_winding_v3 (RK4-предиктор + Ньютон + бисекция).
3. Кинематика: Machine3AxisExact_ODE — развёртка законов движения.
4. Верификация + визуализация + сохранение.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
from scipy.interpolate import CubicSpline
from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.ellipsoid import EllipsoidWithDerivatives
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from machine.machine3axis_exact import Machine3AxisExact_ODE, MachineState


# ======================================================================
# 1. ГЕОМЕТРИЯ (из fnc_2_1.py)
# ======================================================================
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)
scale = 0.8
a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)

print("===== 1. Прямая задача (геодезическая на E1) =====")
deviation_law = ConstantDeviation(tan_theta=0.0)
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)
forward_builder = ForwardWindingBuilder(
    surface=E1, deviation_law=deviation_law,
    solver=solver_forward, normalize_tangent=True, eps=1e-12
)

u0, v0 = np.pi / 3.0, np.pi / 6.0
alpha = np.pi / 6.0
s_end = 30.0
count_points = 100
s_eval = np.linspace(0.0, s_end, count_points)

s_vals_fwd, line_E1 = forward_builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
    eval_points=s_eval
)
if not forward_builder.last_run_successful:
    raise RuntimeError("Прямая задача завершилась с ошибкой")
print(f"Построено {len(s_vals_fwd)} точек на E1, длина s = {s_vals_fwd[-1]:.3f}")

traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории (сплайн): {traj.total_length:.4f}")


# ======================================================================
# 2. ОБРАТНАЯ ЗАДАЧА — inverse_winding_v3 (из fnc_2_1.py)
# ======================================================================
print("\n===== 2. Обратная задача v3 (собственная реализация) =====")

# --- Вспомогательные функции из fnc_2_1.py ---
def compute_tangent_components(surface, u, v, tau_3d):
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
    L, M, N_ff = surface.second_fundamental_form(u, v)
    E, F, G = surface.first_fundamental_form(u, v)
    II_val = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_ff * v_prime**2
    I_val = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
    if abs(I_val) < 1e-15:
        return 0.0
    return II_val / I_val

def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
    E, F, G = surface.first_fundamental_form(u, v)
    L, M, N_ff = surface.second_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    guu, guv, gvv = G / det, -F / det, E / det
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
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    return G / det, -F / det, E / det

def recompute_thread_geometry(surface, traj, u, v, z):
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
    m = surface.normal(u, v)
    return vec_3d - np.dot(vec_3d, m) * m

def compute_dr_dz(surface, traj, u, v, z):
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
    norm_grad_sq = guu * dPhi_du**2 + 2 * guv * dPhi_du * dPhi_dv + gvv * dPhi_dv**2
    if norm_grad_sq < 1e-14:
        return Rp_u, Rp_v
    mu = -residual / norm_grad_sq
    du_dz = Rp_u + mu * grad_u
    dv_dz = Rp_v + mu * grad_v
    return du_dz, dv_dz

def newton_corrector(surface, traj, u_pred, v_pred, z_target,
                     eps_Phi=1e-10, max_iter=7):
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
        Ng = guu * dPdu**2 + 2 * guv * dPdu * dPdv + gvv * dPdv**2
        if Ng < 1e-14:
            return u_c, v_c, Phi_c, nit, False
        u_c -= Phi_c / Ng * (guu * dPdu + guv * dPdv)
        v_c -= Phi_c / Ng * (guv * dPdu + gvv * dPdv)
    r_c = surface.position(u_c, v_c)
    m_c = surface.normal(u_c, v_c)
    Phi_c = np.dot(traj.R(z_target) - r_c, m_c)
    return u_c, v_c, Phi_c, max_iter, abs(Phi_c) < eps_Phi

def inverse_winding_v3(surface, traj, u0, v0, count_points=300,
                       eps_Phi=1e-10, max_newton=7,
                       max_bisect=4, jump_threshold=3.0):
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N)
    flags = np.zeros(N, dtype=int)
    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
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
        try:
            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
        except ValueError:
            du_dz_k, dv_dz_k = 0.0, 0.0
        E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
        speed_expected = np.sqrt(max(E_f * du_dz_k**2
            + 2 * F_f * du_dz_k * dv_dz_k + G_f * dv_dz_k**2, 0.0))
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
                u_c, v_c, Phi_c, nit, conv = newton_corrector(
                    surface, traj, u_p, v_p, z_b,
                    eps_Phi=eps_Phi, max_iter=max_newton
                )
                total_nit += nit
                du_j = u_c - u_s
                dv_j = v_c - v_s
                Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                ds_actual = np.sqrt(max(Ej * du_j**2 + 2*Fj * du_j * dv_j + Gj * dv_j**2, 0.0))
                ds_expect = speed_expected * dz_sub
                ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0
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

    points_3d = np.array([
        surface.position(u_hist[k], v_hist[k]) for k in range(N)
    ])
    return {
        'z_eval': z_eval, 'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }


# --- Запуск ---
u0_m = u0
v0_m = v0
r0 = E2.position(u0_m, v0_m)
R0 = traj.R(0.0)
m0 = E2.normal(u0_m, v0_m)
Phi0 = np.dot(R0 - r0, m0)
print(f"Начальная невязка Φ₀ = {Phi0:.6e}")

if abs(Phi0) > 1e-8:
    u0_m, v0_m, Phi0_c, _, conv0 = newton_corrector(
        E2, traj, u0_m, v0_m, 0.0, eps_Phi=1e-12, max_iter=20
    )
    print(f"После коррекции: Φ₀ = {Phi0_c:.6e}, conv = {conv0}")

result_inv = inverse_winding_v3(
    E2, traj, u0_m, v0_m,
    count_points=300, eps_Phi=1e-10, max_newton=7,
    max_bisect=4, jump_threshold=3.0
)

line_E2 = result_inv['points_3d']
print(f"Обратная задача: max |Φ| = {np.max(np.abs(result_inv['Phi'])):.2e}")
print(f"  Средняя |Φ| = {np.mean(np.abs(result_inv['Phi'])):.2e}")


# ======================================================================
# 3. ПОДГОТОВКА ДАННЫХ ДЛЯ СТАНКА
# ======================================================================
print("\n===== 3. Подготовка данных для станка =====")

z_offset = 0.0
n_pts = len(line_E2)
s_array = np.linspace(0, traj.total_length, n_pts)

tsn_pts = np.array([traj.R(s) for s in s_array])
mandrel_pts = line_E2.copy()

tsn_spline = CubicSpline(s_array, tsn_pts, axis=0, bc_type='natural')
mandrel_spline = CubicSpline(s_array, mandrel_pts, axis=0, bc_type='natural')
tsn_func = lambda s: tsn_spline(s)
mandrel_func = lambda s: mandrel_spline(s)
d_tsn_func = lambda s: tsn_spline(s, nu=1)
d_mandrel_func = lambda s: mandrel_spline(s, nu=1)

alpha_az = np.arctan2(mandrel_pts[:, 1], mandrel_pts[:, 0])
theta_array = np.unwrap(alpha_az)


# ======================================================================
# 4. ОБРАТНАЯ ЗАДАЧА КИНЕМАТИКИ
# ======================================================================
print("\n===== 4. Пространственная развёртка (станок) =====")

RING_RADIUS = 0.3
D_OFFSET = 0.6

machine_ode = Machine3AxisExact_ODE(ring_radius=RING_RADIUS, d_offset=D_OFFSET)

target0 = {
    'point': tsn_func(s_array[0]),
    'r_mandrel': mandrel_func(s_array[0])
}
initial_guess = MachineState([
    theta_array[0],
    mandrel_pts[0, 2],
    np.linalg.norm(mandrel_pts[0, :2]),
    0.0
])

q0 = machine_ode.inverse_first_point(target0, initial_guess)

deploy_result = machine_ode.integrate(
    (s_array[0], s_array[-1]),
    q0.coords,
    tsn_func, mandrel_func, d_tsn_func, d_mandrel_func,
    s_eval=s_array
)

coords = deploy_result['coords']
s_out = deploy_result['s_array']

theta_actual = coords[:, 0]
Z_actual = coords[:, 1] + z_offset
R_actual = coords[:, 2]
phi_actual_deg = np.degrees(coords[:, 3])

print(f"Развёртка: {len(s_out)} точек")


# ======================================================================
# 5. ВЕРИФИКАЦИЯ
# ======================================================================
print("\n===== 5. Верификация =====")

R_tsn_reconstructed = np.zeros_like(tsn_pts)
for i in range(len(s_out)):
    state = MachineState(coords[i])
    R_tsn_reconstructed[i] = machine_ode.forward(state)['point']

error = np.linalg.norm(tsn_pts - R_tsn_reconstructed, axis=1)
print(f"Средняя ошибка: {np.mean(error):.3e}")
print(f"Максимальная ошибка: {np.max(error):.3e}")


# ======================================================================
# 6. ВИЗУАЛИЗАЦИЯ 3D
# ======================================================================
print("\n===== 6. Визуализация =====")

fig3d = go.Figure()

# E1 — эллипсоид
u_e = np.linspace(0, 2*np.pi, 80)
v_e = np.linspace(-np.pi/2, np.pi/2, 50)
Ue, Ve = np.meshgrid(u_e, v_e)
# EllipsoidWithDerivatives: position(u,v) = (a*cos(u)*cos(v), b*sin(u)*cos(v), c*sin(v))
X1 = a1 * np.cos(Ue) * np.cos(Ve)
Y1 = b1 * np.sin(Ue) * np.cos(Ve)
Z1 = c1 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1, opacity=0.15, colorscale='Blues',
    showscale=False, name=f'E1 ({a1}×{b1}×{c1})'
))

# E2 — эллипсоид
X2 = a2 * np.cos(Ue) * np.cos(Ve)
Y2 = b2 * np.sin(Ue) * np.cos(Ve)
Z2 = c2 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2, opacity=0.2, colorscale='Reds',
    showscale=False, name=f'E2 ({a2}×{b2}×{c2})'
))

# ТСН
fig3d.add_trace(go.Scatter3d(
    x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
    mode='lines', line=dict(color='blue', width=3), name='ТСН (E1)'
))

# ЛУ
fig3d.add_trace(go.Scatter3d(
    x=mandrel_pts[:, 0], y=mandrel_pts[:, 1], z=mandrel_pts[:, 2],
    mode='lines', line=dict(color='green', width=3), name='ЛУ (E2)'
))

# Центр кольца
X_center = R_actual * np.cos(theta_actual)
Y_center = R_actual * np.sin(theta_actual)
fig3d.add_trace(go.Scatter3d(
    x=X_center, y=Y_center, z=Z_actual,
    mode='lines', line=dict(color='orange', width=3, dash='dot'),
    name='Центр кольца'
))

fig3d.update_layout(
    title='Развёртка: Пара эллипсоидов -> Станок',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    margin=dict(l=0, r=0, b=0, t=30)
)
fig3d.write_html('ellipsoid_pair_kinematics_3d.html')


# ======================================================================
# 7. 2D-графики
# ======================================================================
fig_2d = make_subplots(rows=4, cols=1,
    subplot_titles=('θ(s), рад', 'Z(s)', 'R(s)', 'φ(s), град'))

fig_2d.add_trace(go.Scatter(x=s_out, y=theta_actual, mode='lines+markers',
    name='θ(s)', line=dict(color='black', width=2)), row=1, col=1)
fig_2d.add_trace(go.Scatter(x=s_out, y=Z_actual, mode='lines+markers',
    name='Z(s)', line=dict(color='blue', width=2)), row=2, col=1)
fig_2d.add_trace(go.Scatter(x=s_out, y=R_actual, mode='lines+markers',
    name='R(s)', line=dict(color='green', width=2)), row=3, col=1)
fig_2d.add_trace(go.Scatter(x=s_out, y=phi_actual_deg, mode='lines+markers',
    name='φ(s)', line=dict(color='purple', width=2)), row=4, col=1)

fig_2d.update_xaxes(title_text='s', row=4, col=1)
fig_2d.update_yaxes(title_text='рад', row=1, col=1)
fig_2d.update_yaxes(title_text='мм', row=2, col=1)
fig_2d.update_yaxes(title_text='мм', row=3, col=1)
fig_2d.update_yaxes(title_text='град', row=4, col=1)
fig_2d.update_layout(height=1000, title_text='Законы движения (пара эллипсоидов)', showlegend=True)
fig_2d.write_html('ellipsoid_pair_kinematics_2d.html')


# ======================================================================
# 8. СОХРАНЕНИЕ
# ======================================================================
results = {
    's': s_out,
    'theta': theta_actual,
    'Z': Z_actual,
    'R': R_actual,
    'phi': coords[:, 3],
    'z_offset': z_offset,
    'tsn_pts': tsn_pts,
    'mandrel_pts': mandrel_pts,
    'ring_radius': RING_RADIUS,
    'd_offset': D_OFFSET,
    'error_mean': float(np.mean(error)),
    'error_max': float(np.max(error))
}
scipy.io.savemat('ellipsoid_pair_kinematics.mat', results)

print("\n===== Готово =====")
print("Графики: ellipsoid_pair_kinematics_3d.html, ellipsoid_pair_kinematics_2d.html")
print("Данные: ellipsoid_pair_kinematics.mat")
