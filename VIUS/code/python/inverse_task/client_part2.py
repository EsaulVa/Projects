#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_part2.py
===============
Клиент для решения обратной задачи намотки на паре эллипсоидов
методом DAE-предиктор–корректор с адаптивной бисекцией.

Импортирует модули из пакета inverse_dae.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from pathlib import Path

from dae_helper_v1 import (
    Ellipsoid, SurfaceGeometryPack,
    TrajectoryByArcLength,
    DAEPredictor, NewtonCorrector, AdaptiveStepper,
)


# ----------------------------------------------------------------------
# Прямая задача: геодезическая (для эталона)
# ----------------------------------------------------------------------
def solve_geodesic(surface, u0, v0, alpha, s_end, num_points=300):
    E0, F0, G0 = surface.first_fundamental_form(u0, v0)
    det0 = E0 * G0 - F0**2
    if det0 <= 0:
        raise ValueError("Вырожденная метрика в начальной точке")
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


# ----------------------------------------------------------------------
# Трассировка луча к эллипсоиду (для верификации)
# ----------------------------------------------------------------------
def trace_ray_to_ellipsoid(ellipsoid, origin, direction):
    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
    ox, oy, oz = origin
    dx, dy, dz = direction
    A = (dx/a)**2 + (dy/b)**2 + (dz/c)**2
    B = 2 * (ox*dx/a**2 + oy*dy/b**2 + oz*dz/c**2)
    C = (ox/a)**2 + (oy/b)**2 + (oz/c)**2 - 1.0
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


# ----------------------------------------------------------------------
# Основной блок
# ----------------------------------------------------------------------
def main():
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 60)
    print("Часть 2: Обратная задача DAE (модуль inverse_dae)")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")

    # Загрузка ТСН из CSV
    csv_path = Path("part1_tsn_shadow.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите Part 1.")
        return

    df = pd.read_csv(csv_path)
    valid = df["valid"].values.astype(bool)
    tsn_pts = df[["X", "Y", "Z"]].values[valid]
    print(f"Загружено {len(tsn_pts)} валидных точек ТСН")

    # Траектория раскладчика
    traj = TrajectoryByArcLength(tsn_pts)
    print(f"Длина траектории ТСН: {traj.total_length:.3f}")

    # Эталон: прямая задача на E2
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = traj.total_length
    num_points = len(tsn_pts)
    print(f"Построение эталонной ЛУ (геодезическая на E2)...")
    _, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Эталон построен: {len(lu_ref)} точек")

    # Обратная задача DAE с адаптивной бисекцией
    print(f"Решение обратной задачи DAE (адаптивная бисекция)...")
    predictor = DAEPredictor(max_speed=50.0)
    corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)
    stepper = AdaptiveStepper(predictor, corrector, C_tol=3.0, max_bisect=4)

    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr
    print(f"Начальная точка скорректирована: Ф0={corr0.Phi:.2e}, итераций={corr0.iterations}")

    z_eval = np.linspace(0, traj.total_length, num_points)
    N = len(z_eval)
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    bisect_flags = np.zeros(N, dtype=int)
    ratio_hist = np.zeros(N)
    u_hist[0], v_hist[0] = u_cur, v_cur

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        dz = z_next - z_k

        result = stepper.step(E2, traj, u_cur, v_cur, z_k, dz)
        u_cur, v_cur = result.u_next, result.v_next
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = result.Phi
        newton_iters_hist[i + 1] = result.iterations
        bisect_flags[i + 1] = 1 if result.bisected else 0
        ratio_hist[i + 1] = result.ratio

        # kappa_n для диагностики
        geom_f = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
        R_f = traj.R(z_next)
        V_thread = R_f - geom_f.r
        lam = np.linalg.norm(V_thread)
        if lam > 1e-12:
            tau = V_thread / lam
            P = geom_f.project_on_basis(tau)
            tau_contra = geom_f.G_inv @ P
            kappa_n_hist[i + 1] = geom_f.normal_curvature(tau_contra)

    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])

    # Верификация: трассировка ТСН от восстановленной ЛУ
    print(f"Верификация: трассировка ТСН от восстановленной ЛУ...")
    tsn_recon = np.zeros((N, 3))
    valid_recon = np.zeros(N, dtype=bool)
    for i, z in enumerate(z_eval):
        r = lu_inv[i]
        R_target = traj.R(z)
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

    # Метрики
    print(f"--- Результаты обратной задачи ---")
    print(f"Max |Ф|        = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Ф|       = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Ф в конце      = {Phi_hist[-1]:.2e}")
    print(f"Max к_n        = {np.max(np.abs(kappa_n_hist)):.4f}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Шагов с бисекцией = {np.sum(bisect_flags)}/{N}")
    print(f"Max ratio      = {np.max(np.abs(ratio_hist[1:])):.4f}")
    print(f"Min ratio      = {np.min(np.abs(ratio_hist[1:])):.4f}")

    if np.sum(valid_recon) > 0:
        R_etalon = traj.R_array(z_eval[valid_recon])
        diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - R_etalon, axis=1)
        print(f"--- Верификация (ТСН по вектору нити) ---")
        print(f"Валидных точек: {np.sum(valid_recon)}/{N}")
        print(f"Max ||ДТСН||   = {np.max(diff_tsn):.6f}")
        print(f"Mean ||ДТСН||  = {np.mean(diff_tsn):.6f}")

    # Сохранение CSV
    df_out = pd.DataFrame({
        "z": z_eval,
        "u": u_hist, "v": v_hist,
        "X": lu_inv[:, 0], "Y": lu_inv[:, 1], "Z": lu_inv[:, 2],
        "Phi": Phi_hist,
        "kappa_n": kappa_n_hist,
        "newton_iters": newton_iters_hist,
        "bisected": bisect_flags,
        "ratio": ratio_hist,
    })
    df_out.to_csv("part2_inverse_dae.csv", index=False)
    print(f"CSV сохранен: part2_inverse_dae.csv")

    # 3D Визуализация
    print("Построение 3D-сцены...")
    fig = go.Figure()
    u_e = np.linspace(0, 2 * np.pi, 60)
    v_e = np.linspace(-np.pi / 2, np.pi / 2, 40)
    Ue, Ve = np.meshgrid(u_e, v_e)

    X1 = a1 * np.cos(Ue) * np.cos(Ve)
    Y1 = b1 * np.sin(Ue) * np.cos(Ve)
    Z1 = c1 * np.sin(Ve)
    fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.15, colorscale="Blues",
                             showscale=False, name="E1 (внешний)"))

    X2 = a2 * np.cos(Ue) * np.cos(Ve)
    Y2 = b2 * np.sin(Ue) * np.cos(Ve)
    Z2 = c2 * np.sin(Ve)
    fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.25, colorscale="Reds",
                             showscale=False, name="E2 (внутренний)"))

    fig.add_trace(go.Scatter3d(x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
                               mode="lines", line=dict(color="black", width=4),
                               name="ТСН исходная"))
    fig.add_trace(go.Scatter3d(x=lu_ref[:, 0], y=lu_ref[:, 1], z=lu_ref[:, 2],
                               mode="lines", line=dict(color="blue", width=3, dash="dash"),
                               name="ЛУ эталон (геодезическая)"))
    fig.add_trace(go.Scatter3d(x=lu_inv[:, 0], y=lu_inv[:, 1], z=lu_inv[:, 2],
                               mode="lines", line=dict(color="green", width=3),
                               name="ЛУ DAE (восстановленная)"))

    if np.sum(valid_recon) > 0:
        fig.add_trace(go.Scatter3d(
            x=tsn_recon[valid_recon, 0],
            y=tsn_recon[valid_recon, 1],
            z=tsn_recon[valid_recon, 2],
            mode="lines", line=dict(color="magenta", width=3, dash="dot"),
            name="ТСН reconstructed"
        ))

    for i in range(0, N, 10):
        fig.add_trace(go.Scatter3d(
            x=[lu_inv[i, 0], tsn_pts[i, 0]],
            y=[lu_inv[i, 1], tsn_pts[i, 1]],
            z=[lu_inv[i, 2], tsn_pts[i, 2]],
            mode="lines", line=dict(color="gray", width=1), showlegend=False
        ))

    fig.update_layout(
        title="Часть 2: Обратная задача DAE с адаптивной бисекцией",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_inverse_dae_3d.html")
    print("3D-сцена сохранена: part2_inverse_dae_3d.html")

    # Графики диагностики
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(z_eval, np.abs(Phi_hist) + 1e-16, "g-", lw=1.5, label="|Ф|")
    ax.axhline(1e-10, color="k", ls="--", lw=0.5)
    ax.set_title("A. Невязка связи |Ф(z)|")
    ax.set_xlabel("z"); ax.set_ylabel("|Ф|")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if np.sum(valid_recon) > 0:
        diff_tsn_full = np.linalg.norm(tsn_recon - tsn_pts, axis=1)
        ax.plot(z_eval[valid_recon], diff_tsn_full[valid_recon] * 1000, "r-", lw=1.5)
    ax.set_title("B. Отклонение ||ТСН_recon - ТСН_orig||, мкм")
    ax.set_xlabel("z"); ax.set_ylabel("||ДТСН||, мкм")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(z_eval[1:], kappa_n_hist[1:], "b-", lw=1.5)
    ax.set_title("C. Нормальная кривизна к_n(z)")
    ax.set_xlabel("z"); ax.set_ylabel("к_n")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(z_eval[1:], newton_iters_hist[1:], "m.", markersize=4)
    ax.set_title("D. Итерации корректора Ньютона")
    ax.set_xlabel("z"); ax.set_ylabel("итерации")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_diagnostics.png", dpi=150)
    print("Графики сохранены: part2_diagnostics.png")
    plt.show()

    print("" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
