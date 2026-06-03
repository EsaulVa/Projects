#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_ellipsoid_pair_inverse_compare.py
=========================================
Сравнительный анализ исходной и скорректированной намотки.

1. Загружает данные из ellipsoid_pair_kinematics.mat (исходная ЛУ + ТСН)
   и ellipsoid_pair_refined_kinematics.mat (новая ТСН после machine_change).
2. Решает обратную задачу от новой ТСН на E2 (inverse_winding_v3).
3. Строит:
   – 3D-сцену сравнения (6 объектов);
   – графики A–E (отклонение по нормали, Δu/Δv, Φ_new, λ, α);
   – таблицу количественных метрик.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from pathlib import Path
import sys

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.ellipsoid import EllipsoidWithDerivatives
from core.trajectory import Trajectory


# ======================================================================
# 0. ГЕОМЕТРИЯ
# ======================================================================
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)
scale = 0.8
a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)


# ======================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ======================================================================
# Исходные данные (до коррекции)
data_orig = scipy.io.loadmat('ellipsoid_pair_kinematics.mat')
s_orig = data_orig['s'].flatten()
tsn_orig = data_orig['tsn_pts']          # (N,3)
lu_orig = data_orig['mandrel_pts']       # (N,3)

# Новые данные (после machine_change)
data_new = scipy.io.loadmat('ellipsoid_pair_refined_kinematics.mat')
s_new = data_new['s'].flatten()
tsn_new_pts = np.column_stack([
    data_new['tsn_pts'][:,0] if 'tsn_pts' in data_new else data_orig['tsn_pts'][:,0],
    data_new['tsn_pts'][:,1] if 'tsn_pts' in data_new else data_orig['tsn_pts'][:,1],
    data_new['tsn_pts'][:,2] if 'tsn_pts' in data_new else data_orig['tsn_pts'][:,2]
])
# Если tsn_pts не сохранились в refined, используем forward для восстановления

# Восстановим новую ТСН через forward, если нет в файле
if 'tsn_pts' not in data_new or data_new['tsn_pts'].shape[0] != len(s_new):
    from machine.machine3axis_exact import Machine3AxisExact_ODE
    from machine.kinematics_base import MachineState
    RING = float(data_new['ring_radius'].flatten()[0])
    D = float(data_new['d_offset'].flatten()[0])
    z_offset = float(data_new['z_offset'].flatten()[0])
    machine = Machine3AxisExact_ODE(ring_radius=RING, d_offset=D)
    tsn_new_pts = np.zeros((len(s_new), 3))
    for i in range(len(s_new)):
        state = MachineState([
            data_new['theta'].flatten()[i],
            data_new['Z_carriage'].flatten()[i],
            data_new['R'].flatten()[i],
            data_new['phi'].flatten()[i]
        ])
        tsn_new_pts[i] = machine.forward(state)['point']
    print("ТСН восстановлена через forward (не найдена в .mat)")
else:
    tsn_new_pts = data_new['tsn_pts']
    print("ТСН загружена из refined .mat")

print(f"Исходная: {len(s_orig)} точек, Новая: {len(s_new)} точек")


# ======================================================================
# 2. ОБРАТНАЯ ЗАДАЧА ОТ НОВОЙ ТСН (inverse_winding_v3)
# ======================================================================
print("\n===== Обратная задача от новой ТСН =====")

traj_new = Trajectory.from_points(tsn_new_pts, method='cubic', bc_type='natural')

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


# --- Начальная точка ---
R0_new = traj_new.R(0.0)
u0_guess = np.arccos(a2 / a1) if R0_new[0] >= 0 else -np.arccos(a2 / a1)
v0_guess = R0_new[2]

# Корректировка начальной точки
u0, v0, Phi0, _, conv0 = newton_corrector(
    E2, traj_new, u0_guess, v0_guess, 0.0, eps_Phi=1e-12, max_iter=20
)
print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}, conv={conv0}")

# Запуск обратной задачи
result_new = inverse_winding_v3(
    E2, traj_new, u0, v0,
    count_points=300, eps_Phi=1e-10, max_newton=7,
    max_bisect=4, jump_threshold=3.0
)

lu_new = result_new['points_3d']
print(f"Обратная задача: max |Φ| = {np.max(np.abs(result_new['Phi'])):.2e}")


# ======================================================================
# 3. ИНТЕРПОЛЯЦИЯ НА ОБЩУЮ СЕТКУ
# ======================================================================
# Интерполируем исходную ЛУ на сетку новой обратной задачи
s_common = result_new['z_eval']

# Сплайны исходной ЛУ
lu_orig_spline = CubicSpline(s_orig, lu_orig, axis=0, bc_type='natural')
lu_orig_on_common = lu_orig_spline(s_common)

# Параметры исходной ЛУ на E2 (через uv_from_point)
u_orig = np.zeros(len(s_common))
v_orig = np.zeros(len(s_common))
for i in range(len(s_common)):
    u_orig[i], v_orig[i] = E2.uv_from_point(lu_orig_on_common[i])

# Параметры новой ЛУ
u_new = result_new['u']
v_new = result_new['v']


# ======================================================================
# 4. ВЫЧИСЛЕНИЕ МЕТРИК
# ======================================================================
N = len(s_common)

# A. Отклонение по нормали
Delta_n = np.zeros(N)
for i in range(N):
    m = E2.normal(u_orig[i], v_orig[i])
    Delta_n[i] = np.dot(lu_new[i] - lu_orig_on_common[i], m)

# B. Разница параметров
Delta_u = u_new - u_orig
Delta_v = v_new - v_orig

# C. Невязка связи новой пары
Phi_new = result_new['Phi']

# D. Длина нити
lam_new = result_new['lam']
lam_orig = np.zeros(N)
for i in range(N):
    lam_orig[i] = np.linalg.norm(tsn_orig[i] - lu_orig[i])

# E. Угол намотки (угол между нитью и меридианом E2)
def compute_alpha(surface, u, v, tau_3d):
    geom = surface.derivatives(u, v)
    ru = geom['ru']
    # меридиан = ru (нормализованный)
    merid = ru / np.linalg.norm(ru)
    tau_proj = tau_3d - np.dot(tau_3d, surface.normal(u, v)) * surface.normal(u, v)
    tau_proj = tau_proj / np.linalg.norm(tau_proj)
    cos_a = np.dot(tau_proj, merid)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

alpha_orig = np.zeros(N)
alpha_new = np.zeros(N)
for i in range(N):
    tau_orig = (tsn_orig[i] - lu_orig[i])
    if np.linalg.norm(tau_orig) > 1e-12:
        tau_orig /= np.linalg.norm(tau_orig)
        alpha_orig[i] = compute_alpha(E2, u_orig[i], v_orig[i], tau_orig)
    tau_new = (tsn_new_pts[i] - lu_new[i]) if i < len(tsn_new_pts) else tau_orig
    if i < len(tsn_new_pts) and np.linalg.norm(tau_new) > 1e-12:
        tau_new /= np.linalg.norm(tau_new)
        alpha_new[i] = compute_alpha(E2, u_new[i], v_new[i], tau_new)


# ======================================================================
# 5. ТАБЛИЦА МЕТРИК
# ======================================================================
print("\n" + "="*60)
print("КОЛИЧЕСТВЕННЫЕ МЕТРИКИ СРАВНЕНИЯ")
print("="*60)
print(f"{'Метрика':<45} {'Значение':>12}")
print("-"*60)
print(f"{'Макс. отклонение ЛУ (евклидово)':<45} {np.max(np.linalg.norm(lu_new - lu_orig_on_common, axis=1)):>12.4f} мм")
print(f"{'Среднее отклонение по нормали |Δn|':<45} {np.mean(np.abs(Delta_n)):>12.4f} мм")
print(f"{'Макс. отклонение по нормали |Δn|':<45} {np.max(np.abs(Delta_n)):>12.4f} мм")
print(f"{'Макс. невязка новой пары |Φ_new|':<45} {np.max(np.abs(Phi_new)):>12.2e}")
print(f"{'Средняя невязка новой пары |Φ_new|':<45} {np.mean(np.abs(Phi_new)):>12.2e}")
print(f"{'Макс. изменение длины нити |Δλ|':<45} {np.max(np.abs(lam_new - lam_orig)):>12.4f} мм")
print(f"{'Среднее изменение длины нити |Δλ|':<45} {np.mean(np.abs(lam_new - lam_orig)):>12.4f} мм")
print(f"{'Макс. изменение угла намотки |Δα|':<45} {np.max(np.abs(alpha_new - alpha_orig)):>12.2f}°")
print(f"{'Среднее изменение угла намотки |Δα|':<45} {np.mean(np.abs(alpha_new - alpha_orig)):>12.2f}°")
print(f"{'Макс. |Δu| (параметр поверхности)':<45} {np.max(np.abs(Delta_u)):>12.4f}")
print(f"{'Макс. |Δv| (параметр поверхности)':<45} {np.max(np.abs(Delta_v)):>12.4f}")
print("="*60)


# ======================================================================
# 6. 3D-ВИЗУАЛИЗАЦИЯ
# ======================================================================
print("\n===== 3D-сцена =====")

fig3d = go.Figure()

# E1
u_e = np.linspace(0, 2*np.pi, 60)
v_e = np.linspace(-np.pi/2, np.pi/2, 40)
Ue, Ve = np.meshgrid(u_e, v_e)
X1 = a1 * np.cos(Ue) * np.cos(Ve)
Y1 = b1 * np.sin(Ue) * np.cos(Ve)
Z1 = c1 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1, opacity=0.1, colorscale='Blues',
    showscale=False, name='E1 (outer)'
))

# E2
X2 = a2 * np.cos(Ue) * np.cos(Ve)
Y2 = b2 * np.sin(Ue) * np.cos(Ve)
Z2 = c2 * np.sin(Ve)
fig3d.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2, opacity=0.15, colorscale='Reds',
    showscale=False, name='E2 (inner)'
))

# Исходная ТСН
fig3d.add_trace(go.Scatter3d(
    x=tsn_orig[:,0], y=tsn_orig[:,1], z=tsn_orig[:,2],
    mode='lines', line=dict(color='red', width=3), name='ТСН исходная'
))

# Новая ТСН
fig3d.add_trace(go.Scatter3d(
    x=tsn_new_pts[:,0], y=tsn_new_pts[:,1], z=tsn_new_pts[:,2],
    mode='lines', line=dict(color='black', width=3, dash='dash'),
    name='ТСН после коррекции'
))

# Исходная ЛУ
fig3d.add_trace(go.Scatter3d(
    x=lu_orig[:,0], y=lu_orig[:,1], z=lu_orig[:,2],
    mode='lines', line=dict(color='green', width=3), name='ЛУ исходная'
))

# Новая ЛУ
fig3d.add_trace(go.Scatter3d(
    x=lu_new[:,0], y=lu_new[:,1], z=lu_new[:,2],
    mode='lines', line=dict(color='blue', width=3, dash='dash'),
    name='ЛУ после коррекции'
))

fig3d.update_layout(
    title='Сравнение до/после коррекции (пара эллипсоидов)',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=1200, height=900
)
fig3d.write_html('ellipsoid_pair_compare_3d.html')
print("3D-сцена: ellipsoid_pair_compare_3d.html")


# ======================================================================
# 7. ГРАФИКИ A–E
# ======================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# A. Отклонение по нормали
ax = axes[0, 0]
ax.plot(s_common, Delta_n * 1000, 'b-', linewidth=1.2, label='Δn(s)')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.fill_between(s_common, 0, Delta_n * 1000, where=(Delta_n > 0), alpha=0.3, color='red', label='наружу')
ax.fill_between(s_common, 0, Delta_n * 1000, where=(Delta_n < 0), alpha=0.3, color='blue', label='внутрь')
ax.set_title('A. Отклонение ЛУ по нормали к E2, мкм')
ax.set_xlabel('s')
ax.set_ylabel('Δn, мкм')
ax.legend(); ax.grid(True, alpha=0.3)

# B. Разница параметров u, v
ax = axes[0, 1]
ax.plot(s_common, Delta_u, 'g-', linewidth=1.2, label='Δu(s)')
ax.plot(s_common, Delta_v, 'm-', linewidth=1.2, label='Δv(s)')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_title('B. Разница параметров на E2')
ax.set_xlabel('s')
ax.set_ylabel('Δu, Δv')
ax.legend(); ax.grid(True, alpha=0.3)

# C. Невязка связи новой пары
ax = axes[1, 0]
ax.semilogy(s_common, np.abs(Phi_new) + 1e-16, 'r-', linewidth=1.2)
ax.axhline(1e-10, color='k', linestyle='--', linewidth=0.5)
ax.set_title('C. Невязка связи |Φ_new(s)|')
ax.set_xlabel('s')
ax.set_ylabel('|Φ|')
ax.grid(True, alpha=0.3)

# D. Длина нити
ax = axes[1, 1]
ax.plot(s_common, lam_orig, 'b-', linewidth=1.5, label='λ_orig(s)')
ax.plot(s_common, lam_new, 'r--', linewidth=1.5, label='λ_new(s)')
ax.set_title('D. Длина свободного участка нити')
ax.set_xlabel('s')
ax.set_ylabel('λ, мм')
ax.legend(); ax.grid(True, alpha=0.3)

# E. Угол намотки
ax = axes[2, 0]
ax.plot(s_common, alpha_orig, 'b-', linewidth=1.5, label='α_orig(s)')
ax.plot(s_common, alpha_new, 'r--', linewidth=1.5, label='α_new(s)')
ax.set_title('E. Угол намотки α(s)')
ax.set_xlabel('s')
ax.set_ylabel('α, град')
ax.legend(); ax.grid(True, alpha=0.3)

# Дополнительно: евклидово отклонение
ax = axes[2, 1]
dist_eucl = np.linalg.norm(lu_new - lu_orig_on_common, axis=1)
ax.plot(s_common, dist_eucl * 1000, 'k-', linewidth=1.2)
ax.set_title('Евклидово отклонение ||Δr||, мкм')
ax.set_xlabel('s')
ax.set_ylabel('||Δr||, мкм')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ellipsoid_pair_compare_AE.png', dpi=150)
plt.show()
print("Графики A–E: ellipsoid_pair_compare_AE.png")

print("\n===== Готово =====")
