#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_part2_1.py (v2)
======================
Обратная задача DAE с расширенной системой (u, v, s).

Изменения:
  • ds/dz = ||u'||_G  интегрируется параллельно с (u, v) методом трапеций.
  • Верификация по точному s: сплайны u(s), v(s) без искажения оси.
  • Проверка геодезичности через остатки уравнений на общей сетке s.
  • Трассировка ТСН от восстановленной ЛУ для кросс-валидации.
  • Убрано дублирование Ellipsoid (импорт из dae_helper_v1).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from pathlib import Path

from dae_helper_v1 import (
    Ellipsoid,
    SurfaceGeometryPack,
    TrajectoryByArcLength,
    DAEPredictor,
    NewtonCorrector,
    AdaptiveStepper,
)


# =======================================================================
# Геодезическая (эталон) — локальная реализация для автономности
# =======================================================================
def solve_geodesic(surface, u0, v0, alpha, s_end, num_points=3000):
    """
    Строит геодезическую на поверхности, параметризованную по длине дуги s.
    alpha — угол в касательной плоскости относительно направления ru.
    """
    E0, F0, G0 = surface.first_fundamental_form(u0, v0)
    det0 = E0 * G0 - F0**2
    if det0 <= 0:
        raise ValueError("Вырожденная метрика в начальной точке")

    # Начальные скорости (p=u', q=v'), |t|_G = 1
    p0 = np.cos(alpha) / np.sqrt(E0) - F0 * np.sin(alpha) / np.sqrt(E0 * det0)
    q0 = np.sin(alpha) * np.sqrt(E0 / det0)

    ru0 = surface.derivatives(u0, v0)["ru"]
    rv0 = surface.derivatives(u0, v0)["rv"]
    t0 = ru0 * p0 + rv0 * q0
    assert abs(np.linalg.norm(t0) - 1.0) < 1e-10

    def rhs(s, y):
        u, v, p, q = y
        try:
            Gamma = surface.christoffel_symbols(u, v)
        except ValueError:
            return [0.0, 0.0, 0.0, 0.0]
        dp = -Gamma[0, 0, 0] * p**2 - 2 * Gamma[0, 0, 1] * p * q - Gamma[0, 1, 1] * q**2
        dq = -Gamma[1, 0, 0] * p**2 - 2 * Gamma[1, 0, 1] * p * q - Gamma[1, 1, 1] * q**2
        return [p, q, dp, dq]

    s_eval = np.linspace(0, s_end, num_points)
    sol = solve_ivp(rhs, [0, s_end], [u0, v0, p0, q0],
                    method="DOP853", t_eval=s_eval, rtol=1e-8, atol=1e-10)
    if sol.status != 0:
        raise RuntimeError(f"Прямая задача не сошлась: {sol.message}")

    uv = sol.y.T
    points = np.array([surface.position(u, v) for u, v in uv[:, :2]])
    return sol.t, uv, points


# =======================================================================
# Трассировка луча к эллипсоиду
# =======================================================================
def trace_ray_to_ellipsoid(ellipsoid, origin, direction):
    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
    ox, oy, oz = origin
    dx, dy, dz = direction
    A = (dx / a)**2 + (dy / b)**2 + (dz / c)**2
    B = 2 * (ox * dx / a**2 + oy * dy / b**2 + oz * dz / c**2)
    C = (ox / a)**2 + (oy / b)**2 + (oz / c)**2 - 1.0

    if abs(A) < 1e-14:
        return None, None
    D = B**2 - 4 * A * C
    if D < 0:
        return None, None

    sqrtD = np.sqrt(D)
    t1 = (-B + sqrtD) / (2 * A)
    t2 = (-B - sqrtD) / (2 * A)
    t_candidates = [t for t in (t1, t2) if t > 1e-9]
    if not t_candidates:
        return None, None

    t = min(t_candidates)
    return t, origin + t * direction


# =======================================================================
# Основной блок
# =======================================================================
def main():
    # --- Геометрия ---
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 70)
    print("Часть 2: Обратная задача DAE (расширенная система u, v, s)")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")
    print("=" * 70)

    # --- Загрузка ТСН из CSV Part 1 ---
    csv_path = Path("part1_tsn_shadow.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите Part 1.")
        return

    df = pd.read_csv(csv_path)
    valid = df["valid"].values.astype(bool)
    tsn_pts = df[["X", "Y", "Z"]].values[valid]
    print(f"\nЗагружено {len(tsn_pts)} валидных точек ТСН")

    # --- Траектория раскладчика (по дуге ТСН, параметр z) ---
    traj = TrajectoryByArcLength(tsn_pts)
    z_end = traj.total_length
    print(f"Длина траектории ТСН (z): {z_end:.4f}")

    # --- Эталон: прямая задача на E2 ---
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = z_end
    num_points = len(tsn_pts)
    print(f"\nПостроение эталонной ЛУ (геодезическая на E2)...")
    s_ref, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Эталон: {len(lu_ref)} точек, длина геодезической s={s_ref[-1]:.4f}")

    # --- Обратная задача DAE с расширенной системой (u, v, s) ---
    print(f"\nРешение обратной задачи DAE (расширенная система)...")
    predictor = DAEPredictor(max_speed=50.0)
    corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)
    stepper = AdaptiveStepper(predictor, corrector, C_tol=3.0, max_bisect=4)

    # Начальная коррекция
    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr
    s_cur = 0.0
    print(f"Начальная точка: Ф0={corr0.Phi:.2e}, итераций={corr0.iterations}")

    # Равномерная сетка по z (параметр траектории ТСН)
    z_eval = np.linspace(0, z_end, num_points)
    dz = z_eval[1] - z_eval[0]

    N = num_points
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    s_hist = np.zeros(N)          # ТОЧНАЯ длина дуги ЛУ
    Phi_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    bisect_flags = np.zeros(N, dtype=int)
    ratio_hist = np.zeros(N)
    sigma_hist = np.zeros(N)      # ds/dz

    u_hist[0], v_hist[0], s_hist[0] = u_cur, v_cur, s_cur

    # Вычисляем начальную скорость и sigma
    du0, dv0, speed0, _ = predictor.compute_velocity(E2, traj, u_cur, v_cur, 0.0)
    geom0 = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
    sigma_hist[0] = geom0.metric_speed(np.array([du0, dv0]))

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        dz_step = z_next - z_k

        # --- Шаг DAE (предиктор + корректор) ---
        result = stepper.step(E2, traj, u_cur, v_cur, z_k, dz_step)
        u_next, v_next = result.u_next, result.v_next

        # --- Точное интегрирование ds/dz ---
        # Скорость в начале шага (уже известна)
        sigma_k = sigma_hist[i]

        # Скорость в конце шага
        du_next, dv_next, _, _ = predictor.compute_velocity(E2, traj, u_next, v_next, z_next)
        geom_next = SurfaceGeometryPack.from_surface(E2, u_next, v_next)
        sigma_next = geom_next.metric_speed(np.array([du_next, dv_next]))
        sigma_hist[i + 1] = sigma_next

        # Метод трапеций для ds/dz
        s_next = s_cur + 0.5 * (sigma_k + sigma_next) * dz_step

        # Сохранение
        u_cur, v_cur, s_cur = u_next, v_next, s_next
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        s_hist[i + 1] = s_cur
        Phi_hist[i + 1] = result.Phi
        newton_iters_hist[i + 1] = result.iterations
        bisect_flags[i + 1] = 1 if result.bisected else 0
        ratio_hist[i + 1] = result.ratio

    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])
    print(f"Восстановленная ЛУ: длина дуги s={s_hist[-1]:.4f}")

    # =====================================================================
    # ВЕРИФИКАЦИЯ 1: Прямое сравнение с геодезической по s
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 1: Сравнение восстановленной ЛУ с геодезической по s")
    print(f"{'='*70}")

    # Убираем возможные дубли s (хотя при dz>0 и sigma>0 их быть не должно)
    s_unique, idx_unique = np.unique(s_hist, return_index=True)
    u_unique = u_hist[idx_unique]
    v_unique = v_hist[idx_unique]

    # Сплайны восстановленной ЛУ по точному s
    u_inv_s = CubicSpline(s_unique, u_unique)
    v_inv_s = CubicSpline(s_unique, v_unique)

    # Сплайны эталона по s (уже по натуральному параметру)
    u_ref_s = CubicSpline(s_ref, uv_ref[:, 0])
    v_ref_s = CubicSpline(s_ref, uv_ref[:, 1])

    # Общая сетка по s
    s_max = min(s_unique[-1], s_ref[-1])
    s_common = np.linspace(0, s_max, 2000)

    lu_inv_at_s = np.array([E2.position(u_inv_s(s), v_inv_s(s)) for s in s_common])
    lu_ref_at_s = np.array([E2.position(u_ref_s(s), v_ref_s(s)) for s in s_common])
    diff_s = np.linalg.norm(lu_inv_at_s - lu_ref_at_s, axis=1)

    print(f"Общая длина s для сравнения: {s_max:.4f}")
    print(f"Max ||r_DAE(s) - r_geo(s)|| = {np.max(diff_s):.6f}")
    print(f"Mean ||r_DAE(s) - r_geo(s)|| = {np.mean(diff_s):.6f}")

    # =====================================================================
    # ВЕРИФИКАЦИЯ 2: Проверка геодезичности восстановленной ЛУ
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 2: Проверка геодезичности (остатки уравнений)")
    print(f"{'='*70}")

    du_ds = u_inv_s.derivative(1)
    dv_ds = v_inv_s.derivative(1)
    d2u_ds2 = u_inv_s.derivative(2)
    d2v_ds2 = v_inv_s.derivative(2)

    s_check = np.linspace(0, s_max, 500)
    res_u = np.zeros_like(s_check)
    res_v = np.zeros_like(s_check)

    for j, s in enumerate(s_check):
        u = float(u_inv_s(s))
        v = float(v_inv_s(s))
        p = float(du_ds(s))
        q = float(dv_ds(s))
        pp = float(d2u_ds2(s))
        qq = float(d2v_ds2(s))
        Gamma = E2.christoffel_symbols(u, v)
        res_u[j] = pp + Gamma[0, 0, 0] * p**2 + 2 * Gamma[0, 0, 1] * p * q + Gamma[0, 1, 1] * q**2
        res_v[j] = qq + Gamma[1, 0, 0] * p**2 + 2 * Gamma[1, 0, 1] * p * q + Gamma[1, 1, 1] * q**2

    print(f"Max |res_u| = {np.max(np.abs(res_u)):.2e}")
    print(f"Max |res_v| = {np.max(np.abs(res_v)):.2e}")
    print(f"Mean |res_u| = {np.mean(np.abs(res_u)):.2e}")
    print(f"Mean |res_v| = {np.mean(np.abs(res_v)):.2e}")

    # =====================================================================
    # ВЕРИФИКАЦИЯ 3: Интегрирование ОДУ геодезических с НУ из ЛУ
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 3: Интегрирование ОДУ геодезических с НУ восст. ЛУ")
    print(f"{'='*70}")

    u0_g = float(u_inv_s(0.0))
    v0_g = float(v_inv_s(0.0))
    p0_g = float(du_ds(0.0))
    q0_g = float(dv_ds(0.0))

    # Нормализация начальной скорости (должна быть единичной по метрике)
    E0, F0, G0 = E2.first_fundamental_form(u0_g, v0_g)
    speed0 = np.sqrt(E0 * p0_g**2 + 2 * F0 * p0_g * q0_g + G0 * q0_g**2)
    print(f"Начальная скорость |dr/ds| = {speed0:.6f}")
    if abs(speed0 - 1.0) > 1e-3:
        p0_g /= speed0
        q0_g /= speed0
        print(f"Перенормирована до 1.0")

    def geodesic_rhs(s, y):
        u, v, p, q = y
        Gamma = E2.christoffel_symbols(u, v)
        dp = -Gamma[0, 0, 0] * p**2 - 2 * Gamma[0, 0, 1] * p * q - Gamma[0, 1, 1] * q**2
        dq = -Gamma[1, 0, 0] * p**2 - 2 * Gamma[1, 0, 1] * p * q - Gamma[1, 1, 1] * q**2
        return [p, q, dp, dq]

    sol_geo = solve_ivp(
        geodesic_rhs,
        [0, s_max],
        [u0_g, v0_g, p0_g, q0_g],
        method="DOP853",
        t_eval=s_common,
        rtol=1e-8,
        atol=1e-10,
    )

    if sol_geo.status != 0:
        print(f"ОШИБКА интегрирования: {sol_geo.message}")
    else:
        u_geo = sol_geo.y[0]
        v_geo = sol_geo.y[1]
        pts_geo = np.array([E2.position(u, v) for u, v in zip(u_geo, v_geo)])
        diff_geo_inv = np.linalg.norm(pts_geo - lu_inv_at_s, axis=1)

        print(f"Max ||r_geo(s) - r_inv(s)|| = {np.max(diff_geo_inv):.6e}")
        print(f"Mean ||r_geo(s) - r_inv(s)|| = {np.mean(diff_geo_inv):.6e}")

    # =====================================================================
    # ВЕРИФИКАЦИЯ 4: Трассировка ТСН от восстановленной ЛУ
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 4: Трассировка ТСН от восстановленной ЛУ")
    print(f"{'='*70}")

    tsn_recon = np.zeros((N, 3))
    valid_recon = np.zeros(N, dtype=bool)

    for i in range(N):
        r = lu_inv[i]
        R_target = traj.R(z_eval[i])
        V_thread = R_target - r
        lam = np.linalg.norm(V_thread)
        if lam < 1e-12:
            continue
        tau = V_thread / lam
        m = E2.normal(u_hist[i], v_hist[i])
        tau_proj = tau - np.dot(tau, m) * m
        tau_norm = np.linalg.norm(tau_proj)
        if tau_norm > 1e-12:
            tau_proj = tau_proj / tau_norm
        t, R_pt = trace_ray_to_ellipsoid(E1, r, tau_proj)
        if t is not None:
            tsn_recon[i] = R_pt
            valid_recon[i] = True

    if np.sum(valid_recon) > 0:
        R_etalon = np.array([traj.R(z) for z in z_eval[valid_recon]])
        diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - R_etalon, axis=1)
        print(f"Валидных точек: {np.sum(valid_recon)}/{N}")
        print(f"Max ||ДТСН|| = {np.max(diff_tsn):.6f}")
        print(f"Mean ||ДТСН|| = {np.mean(diff_tsn):.6f}")

    # =====================================================================
    # Метрики DAE
    # =====================================================================
    print(f"\n{'='*70}")
    print("МЕТРИКИ DAE")
    print(f"{'='*70}")
    print(f"Max |Ф| = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Ф| = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Ф в конце = {Phi_hist[-1]:.2e}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Шагов с бисекцией = {np.sum(bisect_flags)}/{N}")
    print(f"Max ratio = {np.max(np.abs(ratio_hist[1:])):.4f}")
    print(f"Min ratio = {np.min(np.abs(ratio_hist[1:])):.4f}")
    print(f"Длина ЛУ (s_end) = {s_hist[-1]:.4f}")
    print(f"Длина ТСН (z_end) = {z_eval[-1]:.4f}")
    print(f"Среднее ds/dz = {np.mean(sigma_hist):.4f}")

    # =====================================================================
    # Сохранение CSV
    # =====================================================================
    df_out = pd.DataFrame({
        "z": z_eval,
        "s": s_hist,
        "u": u_hist,
        "v": v_hist,
        "X": lu_inv[:, 0],
        "Y": lu_inv[:, 1],
        "Z": lu_inv[:, 2],
        "Phi": Phi_hist,
        "sigma": sigma_hist,
        "newton_iters": newton_iters_hist,
        "bisected": bisect_flags,
        "ratio": ratio_hist,
    })
    df_out.to_csv("part2_inverse_dae.csv", index=False)
    print(f"\nCSV сохранен: part2_inverse_dae.csv")

    # =====================================================================
    # 3D Визуализация
    # =====================================================================
    print("\nПостроение 3D-сцены...")
    fig = go.Figure()

    u_e = np.linspace(0, 2 * np.pi, 60)
    v_e = np.linspace(-np.pi / 2, np.pi / 2, 40)
    Ue, Ve = np.meshgrid(u_e, v_e)

    X1 = a1 * np.cos(Ue) * np.cos(Ve)
    Y1 = b1 * np.sin(Ue) * np.cos(Ve)
    Z1 = c1 * np.sin(Ve)
    fig.add_trace(go.Surface(
        x=X1, y=Y1, z=Z1, opacity=0.15, colorscale="Blues",
        showscale=False, name="E1 (внешний)"
    ))

    X2 = a2 * np.cos(Ue) * np.cos(Ve)
    Y2 = b2 * np.sin(Ue) * np.cos(Ve)
    Z2 = c2 * np.sin(Ve)
    fig.add_trace(go.Surface(
        x=X2, y=Y2, z=Z2, opacity=0.25, colorscale="Reds",
        showscale=False, name="E2 (внутренний)"
    ))

    fig.add_trace(go.Scatter3d(
        x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
        mode="lines", line=dict(color="black", width=4),
        name="ТСН исходная"
    ))
    fig.add_trace(go.Scatter3d(
        x=lu_ref[:, 0], y=lu_ref[:, 1], z=lu_ref[:, 2],
        mode="lines", line=dict(color="blue", width=3, dash="dash"),
        name="ЛУ эталон (геодезическая)"
    ))
    fig.add_trace(go.Scatter3d(
        x=lu_inv[:, 0], y=lu_inv[:, 1], z=lu_inv[:, 2],
        mode="lines", line=dict(color="green", width=3),
        name="ЛУ DAE (восстановленная)"
    ))

    if np.sum(valid_recon) > 0:
        fig.add_trace(go.Scatter3d(
            x=tsn_recon[valid_recon, 0],
            y=tsn_recon[valid_recon, 1],
            z=tsn_recon[valid_recon, 2],
            mode="lines", line=dict(color="magenta", width=3, dash="dot"),
            name="ТСН reconstructed"
        ))

    # Отрезки нити
    for i in range(0, N, 20):
        fig.add_trace(go.Scatter3d(
            x=[lu_inv[i, 0], tsn_pts[i, 0]],
            y=[lu_inv[i, 1], tsn_pts[i, 1]],
            z=[lu_inv[i, 2], tsn_pts[i, 2]],
            mode="lines", line=dict(color="gray", width=1), showlegend=False
        ))

    fig.update_layout(
        title="Часть 2: Обратная задача DAE (расширенная система u, v, s)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_inverse_dae_3d.html")
    print("3D-сцена сохранена: part2_inverse_dae_3d.html")

    # =====================================================================
    # Графики диагностики
    # =====================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(s_common, diff_s * 1000, "b-", lw=1.0)
    ax.set_title("A. Отклонение ||r_DAE - r_geo|| по s, мм")
    ax.set_xlabel("s"); ax.set_ylabel("мм")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(z_eval, np.abs(Phi_hist) + 1e-16, "g-", lw=1.5)
    ax.axhline(1e-10, color="k", ls="--", lw=0.5)
    ax.set_title("B. Невязка |Ф(z)|")
    ax.set_xlabel("z"); ax.set_ylabel("|Ф|")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(z_eval, s_hist, "c-", lw=1.5, label="s(z)")
    ax.plot(z_eval, z_eval, "k--", lw=0.8, label="s=z")
    ax.set_title("C. Накопленная длина ЛУ s(z)")
    ax.set_xlabel("z"); ax.set_ylabel("s")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(z_eval, sigma_hist, "m-", lw=1.0)
    ax.set_title("D. Метрическая скорость ds/dz")
    ax.set_xlabel("z"); ax.set_ylabel("ds/dz")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(s_check, np.abs(res_u) + 1e-16, "b-", lw=0.8, label="|res_u|")
    ax.semilogy(s_check, np.abs(res_v) + 1e-16, "r-", lw=0.8, label="|res_v|")
    ax.axhline(1e-6, color="k", ls="--", lw=0.5)
    ax.set_title("E. Остатки геодезичности")
    ax.set_xlabel("s"); ax.set_ylabel("|остаток|")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(z_eval[1:], newton_iters_hist[1:], "m.", markersize=4)
    ax.set_title("F. Итерации корректора Ньютона")
    ax.set_xlabel("z"); ax.set_ylabel("итерации")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_diagnostics.png", dpi=150)
    print("Графики сохранены: part2_diagnostics.png")
    plt.show()

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()
