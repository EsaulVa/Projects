#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_ellipsoid_compare_methods.py
====================================
Сравнение двух методов решения обратной задачи намотки
на примере пары эллипсоидов (E1 внешний, E2 внутренний).

Метод 1: Классический (Савин) — RightHandSideCalculator + SciPySolver.solve
  • Интегрирование системы (3.41) в параметрах (u,v)
  • Пропорциональная коррекция через k=1.0
  • Адаптивный шаг DOP853

Метод 2: DAE-предиктор-корректор (наш отчёт) — inverse_winding_v3
  • Явный предиктор Эйлера + корректор Ньютона
  • Адаптивная бисекция при скачках
  • Квадратичная сходимость корректора

ИСПРАВЛЕНИЕ: используем SciPySolver.solve() напрямую,
т.к. solve_with_diagnostics() отсутствует в текущей версии.
"""

import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.ellipsoid import EllipsoidWithDerivatives
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver

# --- Метод 1: классический ---
from inverse_winding.rhs_calculator import RightHandSideCalculator


# ======================================================================
# 1. ГЕОМЕТРИЯ
# ======================================================================
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)
scale = 0.8
a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)

print("=" * 60)
print("1. ГЕОМЕТРИЯ")
print(f"   E1 (внешний): a={a1}, b={b1}, c={c1}")
print(f"   E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")


# ======================================================================
# 2. ПРЯМАЯ ЗАДАЧА: геодезическая на E1
# ======================================================================
print("\n2. ПРЯМАЯ ЗАДАЧА (геодезическая на E1)")
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
assert forward_builder.last_run_successful, "Прямая задача не сошлась"

traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"   Построено {len(s_vals_fwd)} точек, длина ТСН = {traj.total_length:.3f}")


# ======================================================================
# 3. МЕТОД 1: Классический (Савин) — rhs_calculator + solve_ivp
# ======================================================================
print("\n3. МЕТОД 1: Классический (rhs_calculator + solve_ivp)")

rhs_calc = RightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=1.0,
    max_ds_dz=50.0,
    delta_clip=0.999,
    eps=1e-12
)

# Начальная точка на E2
u0_m = u0
v0_m = v0

# Корректировка начальной точки (Newton)
from helpers.inverse_method import newton_corrector as nc_savin
u0_m, v0_m, Phi0_c, _, conv0 = nc_savin(
    E2, traj, u0_m, v0_m, 0.0, eps_Phi=1e-12, max_iter=20
)
print(f"   Начальная точка: Φ₀ = {Phi0_c:.6e}, conv={conv0}")

# Интегрирование через solve_ivp напрямую
z_eval = np.linspace(0, traj.total_length, 300)

def rhs_wrapper(z, state):
    du, dv = rhs_calc(z, state)
    return [du, dv]

t1_start = time.time()
sol1 = solve_ivp(
    rhs_wrapper,
    [0, traj.total_length],
    [u0_m, v0_m],
    method='DOP853',
    t_eval=z_eval,
    rtol=1e-8,
    atol=1e-10
)
t1_end = time.time()

z1 = sol1.t
uv1 = sol1.y.T
lu1 = np.array([E2.position(u, v) for u, v in uv1])

# Невязки
Phi1 = np.zeros(len(z1))
for i in range(len(z1)):
    R = traj.R(z1[i])
    r = lu1[i]
    m = E2.normal(uv1[i, 0], uv1[i, 1])
    diff = R - r
    diff_norm = np.linalg.norm(diff)
    if diff_norm > 1e-12:
        Phi1[i] = np.dot(diff / diff_norm, m)

print(f"   Время: {t1_end - t1_start:.3f} с")
print(f"   Max |Φ| = {np.max(np.abs(Phi1)):.2e}")
print(f"   Mean |Φ| = {np.mean(np.abs(Phi1)):.2e}")
print(f"   Φ₀ = {Phi1[0]:.2e}, Φ_end = {Phi1[-1]:.2e}")


# ======================================================================
# 4. МЕТОД 2: DAE-предиктор-корректор (наш отчёт)
# ======================================================================
print("\n4. МЕТОД 2: DAE-предиктор-корректор (inverse_winding_v3)")

# --- Вспомогательные функции (копия из fnc_2_1.py) ---
def compute_tangent_components(surface, u, v, tau_3d):
    geom = surface.derivatives(u, v)
    ru, rv = geom['ru'], geom['rv']
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    b1 = np.dot(tau_3d, ru)
    b2 = np.dot(tau_3d, rv)
    return (G * b1 - F * b2) / det, (-F * b1 + E * b2) / det

def normal_curvature(surface, u, v, u_prime, v_prime):
    L, M, N_ff = surface.second_fundamental_form(u, v)
    E, F, G = surface.first_fundamental_form(u, v)
    II_val = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_ff * v_prime**2
    I_val = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
    return II_val / I_val if abs(I_val) > 1e-15 else 0.0

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
    tau = delta / lam
    return tau, lam, *compute_tangent_components(surface, u, v, tau)

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
    Ng = guu * dPhi_du**2 + 2 * guv * dPhi_du * dPhi_dv + gvv * dPhi_dv**2
    if Ng < 1e-14:
        return Rp_u, Rp_v
    mu = -residual / Ng
    return Rp_u + mu * grad_u, Rp_v + mu * grad_v

def newton_corrector_dae(surface, traj, u_pred, v_pred, z_target,
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
            print(f"   Шаг {i}: {e}")
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
                u_c, v_c, Phi_c, nit, conv = newton_corrector_dae(
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


# Запуск метода 2
t2_start = time.time()
result2 = inverse_winding_v3(
    E2, traj, u0_m, v0_m,
    count_points=300, eps_Phi=1e-10, max_newton=7,
    max_bisect=4, jump_threshold=3.0
)
t2_end = time.time()

lu2 = result2['points_3d']
Phi2 = result2['Phi']
print(f"   Время: {t2_end - t2_start:.3f} с")
print(f"   Max |Φ| = {np.max(np.abs(Phi2)):.2e}")
print(f"   Mean |Φ| = {np.mean(np.abs(Phi2)):.2e}")
print(f"   Φ₀ = {Phi2[0]:.2e}, Φ_end = {Phi2[-1]:.2e}")


# ======================================================================
# 5. СВОДНАЯ ТАБЛИЦА МЕТРИК
# ======================================================================
print("\n" + "=" * 60)
print("5. СВОДНАЯ ТАБЛИЦА МЕТРИК")
print("=" * 60)

# Интерполируем обе ЛУ на общую сетку для сравнения
s_common = np.linspace(0, traj.total_length, 300)
lu1_spline = CubicSpline(z1, lu1, axis=0, bc_type='natural')
lu2_spline = CubicSpline(result2['z_eval'], lu2, axis=0, bc_type='natural')
lu1_common = lu1_spline(s_common)
lu2_common = lu2_spline(s_common)

# Отклонение между методами
diff_12 = np.linalg.norm(lu1_common - lu2_common, axis=1)

# Невязки на общей сетке (интерполяция)
Phi1_spline = CubicSpline(z1, np.abs(Phi1), bc_type='natural')
Phi2_spline = CubicSpline(result2['z_eval'], np.abs(Phi2), bc_type='natural')
Phi1_common = Phi1_spline(s_common)
Phi2_common = Phi2_spline(s_common)

metrics = {
    'Метрика': [
        'Время расчёта, с',
        'Max |Φ|',
        'Mean |Φ|',
        'Φ в конце траектории',
        'Max отклонение ||r₁-r₂||, мм',
        'Mean отклонение ||r₁-r₂||, мм',
        'Max κ_n',
        'Среднее итераций Ньютона',
        'Шагов с бисекцией (%)'
    ],
    'Метод 1 (Савин)': [
        f"{t1_end - t1_start:.3f}",
        f"{np.max(np.abs(Phi1)):.2e}",
        f"{np.mean(np.abs(Phi1)):.2e}",
        f"{Phi1[-1]:.2e}",
        "—",
        "—",
        "—",
        "—",
        "—"
    ],
    'Метод 2 (DAE)': [
        f"{t2_end - t2_start:.3f}",
        f"{np.max(np.abs(Phi2)):.2e}",
        f"{np.mean(np.abs(Phi2)):.2e}",
        f"{Phi2[-1]:.2e}",
        f"{np.max(diff_12):.4f}",
        f"{np.mean(diff_12):.4f}",
        f"{np.max(np.abs(result2['kappa_n'])):.4f}",
        f"{np.mean(result2['newton_iters'][1:]):.2f}",
        f"{100*np.sum(result2['flags']==1)/len(result2['flags']):.1f}%"
    ]
}

print(f"{'Метрика':<40} {'Метод 1 (Савин)':>20} {'Метод 2 (DAE)':>20}")
print("-" * 80)
for i in range(len(metrics['Метрика'])):
    print(f"{metrics['Метрика'][i]:<40} {metrics['Метод 1 (Савин)'][i]:>20} {metrics['Метод 2 (DAE)'][i]:>20}")
print("=" * 80)


# ======================================================================
# 6. ВИЗУАЛИЗАЦИЯ 3D
# ======================================================================
print("\n6. ВИЗУАЛИЗАЦИЯ 3D")

fig3d = go.Figure()

# E1 — полупрозрачный
u_e = np.linspace(0, 2*np.pi, 60)
v_e = np.linspace(-np.pi/2, np.pi/2, 40)
Ue, Ve = np.meshgrid(u_e, v_e)
X1 = a1 * np.cos(Ue) * np.cos(Ve)
Y1 = b1 * np.sin(Ue) * np.cos(Ve)
Z1 = c1 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1, opacity=0.1, colorscale='Blues',
    showscale=False, name='E1 (внешний)'
))

# E2 — полупрозрачный
X2 = a2 * np.cos(Ue) * np.cos(Ve)
Y2 = b2 * np.sin(Ue) * np.cos(Ve)
Z2 = c2 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2, opacity=0.15, colorscale='Reds',
    showscale=False, name='E2 (внутренний)'
))

# ТСН
fig3d.add_trace(go.Scatter3d(
    x=line_E1[:,0], y=line_E1[:,1], z=line_E1[:,2],
    mode='lines', line=dict(color='black', width=4), name='ТСН (E1)'
))

# ЛУ метод 1
fig3d.add_trace(go.Scatter3d(
    x=lu1[:,0], y=lu1[:,1], z=lu1[:,2],
    mode='lines', line=dict(color='blue', width=3), name='ЛУ метод 1 (Савин)'
))

# ЛУ метод 2
fig3d.add_trace(go.Scatter3d(
    x=lu2[:,0], y=lu2[:,1], z=lu2[:,2],
    mode='lines', line=dict(color='green', width=3), name='ЛУ метод 2 (DAE)'
))

fig3d.update_layout(
    title='Сравнение методов: пара эллипсоидов',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=1200, height=900
)
fig3d.write_html('compare_methods_3d.html')
print("   3D-сцена: compare_methods_3d.html")


# ======================================================================
# 7. ГРАФИКИ СРАВНЕНИЯ
# ======================================================================
print("\n7. ГРАФИКИ СРАВНЕНИЯ")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Невязка Φ(s)
ax = axes[0, 0]
ax.semilogy(z1, np.abs(Phi1) + 1e-16, 'b-', linewidth=1.5, label='Метод 1 (Савин)')
ax.semilogy(result2['z_eval'], np.abs(Phi2) + 1e-16, 'g-', linewidth=1.5, label='Метод 2 (DAE)')
ax.axhline(1e-10, color='k', linestyle='--', linewidth=0.5)
ax.set_title('A. Невязка связи |Φ(s)|')
ax.set_xlabel('s')
ax.set_ylabel('|Φ|')
ax.legend()
ax.grid(True, alpha=0.3)

# B. Отклонение между методами
ax = axes[0, 1]
ax.plot(s_common, diff_12 * 1000, 'r-', linewidth=1.5)
ax.set_title('B. Отклонение ||r₁(s) - r₂(s)||, мкм')
ax.set_xlabel('s')
ax.set_ylabel('||Δr||, мкм')
ax.grid(True, alpha=0.3)

# C. Длина нити λ(s)
ax = axes[1, 0]
lam1 = np.linalg.norm(line_E1[:len(lu1)] - lu1, axis=1)
lam2 = result2['lam']
ax.plot(z1[:len(lam1)], lam1, 'b-', linewidth=1.5, label='Метод 1')
ax.plot(result2['z_eval'], lam2, 'g-', linewidth=1.5, label='Метод 2')
ax.set_title('C. Длина свободного участка λ(s)')
ax.set_xlabel('s')
ax.set_ylabel('λ, мм')
ax.legend()
ax.grid(True, alpha=0.3)

# D. Итерации Ньютона (метод 2)
ax = axes[1, 1]
ax.plot(result2['z_eval'][1:], result2['newton_iters'][1:], 'g.', markersize=3)
ax.set_title('D. Итерации корректора Ньютона (метод 2)')
ax.set_xlabel('s')
ax.set_ylabel('итерации')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('compare_methods_plots.png', dpi=150)
plt.show()
print("   Графики: compare_methods_plots.png")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)