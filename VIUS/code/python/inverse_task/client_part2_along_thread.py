#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_part2_along_thread.py
============================
Вариант A: обратная задача с параметром z = s (длина дуги ЛУ).
Траектория раскладчика синтезирована так, что R(s) = r(s) + lambda(s)*tau(s),
а скорость раскладчика R'(s) = tau(s) — единичный вектор вдоль нити.

Это позволяет тестировать DAE-модуль на известной геодезической,
хотя на эллипсоиде (kappa_g != 0) восстановленная ЛУ будет отличаться
от геодезической из-за условия сохранения контакта Phi=0.
Для ИДЕАЛЬНОГО совпадения замените Ellipsoid на сферу (a=b=c).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from pathlib import Path

from dae_helper_v1 import (
    Ellipsoid, SurfaceGeometryPack,
    DAEPredictor, NewtonCorrector, AdaptiveStepper,
)


# =======================================================================
# Класс траектории: раскладчик движется ВДОЛЬ НИТИ (а не вдоль ТСН)
# =======================================================================
class TrajectoryAlongThread:
    """
    Траектория раскладчика, параметризованная по s (длина дуги ЛУ).
    R(s) = r(s) + lambda(s) * tau(s)  (точки ТСН)
    R'(s) = tau(s)  (направление нити, единичное)
    """
    def __init__(self, s_vals, r_vals, tau_vals, lambda_vals):
        s_vals = np.asarray(s_vals)
        r_vals = np.asarray(r_vals)
        tau_vals = np.asarray(tau_vals)
        lambda_vals = np.asarray(lambda_vals)

        # Точки ТСН
        R_vals = r_vals + lambda_vals[:, None] * tau_vals

        self._s = s_vals
        self._total = float(s_vals[-1])

        # Сплайны для R(s)
        self._sx = CubicSpline(s_vals, R_vals[:, 0])
        self._sy = CubicSpline(s_vals, R_vals[:, 1])
        self._sz = CubicSpline(s_vals, R_vals[:, 2])

        # Сплайны для tau(s) — направление нити
        self._tx = CubicSpline(s_vals, tau_vals[:, 0])
        self._ty = CubicSpline(s_vals, tau_vals[:, 1])
        self._tz = CubicSpline(s_vals, tau_vals[:, 2])

    def R(self, s):
        s = float(np.clip(s, 0.0, self._total))
        return np.array([self._sx(s), self._sy(s), self._sz(s)])

    def R_deriv(self, s):
        """Единичный вектор вдоль нити tau(s) (НЕ касательная к ТСН)."""
        s = float(np.clip(s, 0.0, self._total))
        tau = np.array([self._tx(s), self._ty(s), self._tz(s)])
        norm = np.linalg.norm(tau)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return tau / norm

    @property
    def total_length(self):
        return self._total


# =======================================================================
# Геодезическая (копия из Part1 для автономности)
# =======================================================================
def solve_geodesic(surface, u0, v0, alpha, s_end, num_points=3000):
    E0, F0, G0 = surface.first_fundamental_form(u0, v0)
    det0 = E0 * G0 - F0**2
    if det0 <= 0:
        raise ValueError("Вырожденная метрика")

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
        raise RuntimeError(f"Геодезическая не сошлась: {sol.message}")

    uv = sol.y.T
    points = np.array([surface.position(u, v) for u, v in uv[:, :2]])
    return sol.t, uv, points


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
    print("ВАРИАНТ A: z = s (длина дуги ЛУ), раскладчик движется вдоль нити")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")
    print("=" * 70)

    # --- 1. Строим геодезическую (эталон) ---
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = 30.0
    num_points = 3000

    print(f"\nПостроение геодезической на E2...")
    s_vals, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"  Геодезическая: {len(s_vals)} точек, s_end={s_vals[-1]:.4f}")

    # --- 2. Вычисляем tau(s) и lambda(s) ---
    tau_vals = np.zeros((num_points, 3))
    lambda_vals = np.zeros(num_points)
    valid_mask = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        u, v = uv_ref[i, 0], uv_ref[i, 1]
        p, q = uv_ref[i, 2], uv_ref[i, 3]
        geom = E2.derivatives(u, v)
        tau = geom["ru"] * p + geom["rv"] * q
        tau_norm = np.linalg.norm(tau)
        if tau_norm > 1e-12:
            tau = tau / tau_norm
        tau_vals[i] = tau

        t, R_pt = trace_ray_to_ellipsoid(E1, lu_ref[i], tau)
        if t is not None:
            lambda_vals[i] = t
            valid_mask[i] = True

    valid_count = np.sum(valid_mask)
    print(f"  Валидных точек ТСН: {valid_count}/{num_points}")

    # --- 3. Траектория раскладчика: движется вдоль нити ---
    traj = TrajectoryAlongThread(s_vals, lu_ref, tau_vals, lambda_vals)
    print(f"  TrajectoryAlongThread: total_length={traj.total_length:.4f}")

    # --- 4. Обратная задача DAE по s ---
    print(f"\nРешение обратной задачи DAE (по s)...")
    predictor = DAEPredictor(max_speed=50.0)
    corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)
    stepper = AdaptiveStepper(predictor, corrector, C_tol=3.0, max_bisect=4)

    # Начальная коррекция
    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr
    print(f"  Начальная точка: Ф0={corr0.Phi:.2e}, итераций={corr0.iterations}")

    N = num_points
    s_eval = s_vals  # z = s
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    bisect_flags = np.zeros(N, dtype=int)
    ratio_hist = np.zeros(N)

    u_hist[0], v_hist[0] = u_cur, v_cur

    for i in range(N - 1):
        s_k = s_eval[i]
        s_next = s_eval[i + 1]
        ds = s_next - s_k

        result = stepper.step(E2, traj, u_cur, v_cur, s_k, ds)
        u_cur, v_cur = result.u_next, result.v_next

        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = result.Phi
        newton_iters_hist[i + 1] = result.iterations
        bisect_flags[i + 1] = 1 if result.bisected else 0
        ratio_hist[i + 1] = result.ratio

    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])

    # --- 5. Сравнение с геодезической (по s, напрямую) ---
    print(f"\n{'='*70}")
    print("СРАВНЕНИЕ: восстановленная ЛУ vs эталонная геодезическая")
    print(f"{'='*70}")

    diff_3d = np.linalg.norm(lu_inv - lu_ref, axis=1)
    diff_u = np.abs(u_hist - uv_ref[:, 0])
    diff_v = np.abs(v_hist - uv_ref[:, 1])

    print(f"Max ||r_DAE - r_geo|| = {np.max(diff_3d):.6e}")
    print(f"Mean ||r_DAE - r_geo|| = {np.mean(diff_3d):.6e}")
    print(f"Max |u_DAE - u_geo|   = {np.max(diff_u):.6e}")
    print(f"Max |v_DAE - v_geo|   = {np.max(diff_v):.6e}")

    # --- 6. Метрики DAE ---
    print(f"\n--- Метрики DAE ---")
    print(f"Max |Ф|        = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Ф|       = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Шагов с бисекцией = {np.sum(bisect_flags)}/{N}")

    # --- 7. Верификация: трассировка ТСН от восстановленной ЛУ ---
    print(f"\nВерификация: трассировка ТСН от восстановленной ЛУ...")
    tsn_recon = np.zeros((N, 3))
    valid_recon = np.zeros(N, dtype=bool)

    for i in range(N):
        r = lu_inv[i]
        R_target = traj.R(s_eval[i])
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
        R_etalon = np.array([traj.R(s) for s in s_eval[valid_recon]])
        diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - R_etalon, axis=1)
        print(f"Валидных точек: {np.sum(valid_recon)}/{N}")
        print(f"Max ||ДТСН||   = {np.max(diff_tsn):.6f}")
        print(f"Mean ||ДТСН||  = {np.mean(diff_tsn):.6f}")

    # --- 8. Сохранение CSV ---
    df = pd.DataFrame({
        "s": s_eval,
        "u_geo": uv_ref[:, 0], "v_geo": uv_ref[:, 1],
        "u_dae": u_hist, "v_dae": v_hist,
        "X_geo": lu_ref[:, 0], "Y_geo": lu_ref[:, 1], "Z_geo": lu_ref[:, 2],
        "X_dae": lu_inv[:, 0], "Y_dae": lu_inv[:, 1], "Z_dae": lu_inv[:, 2],
        "Phi": Phi_hist,
        "diff_3d": diff_3d,
    })
    df.to_csv("part2_along_thread.csv", index=False)
    print(f"\nCSV сохранен: part2_along_thread.csv")

    # --- 9. 3D-визуализация ---
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

    fig.add_trace(go.Scatter3d(x=lu_ref[:, 0], y=lu_ref[:, 1], z=lu_ref[:, 2],
                                 mode="lines", line=dict(color="blue", width=4, dash="dash"),
                                 name="ЛУ эталон (геодезическая)"))
    fig.add_trace(go.Scatter3d(x=lu_inv[:, 0], y=lu_inv[:, 1], z=lu_inv[:, 2],
                                 mode="lines", line=dict(color="green", width=4),
                                 name="ЛУ DAE (восстановленная)"))

    # Отрезки нити
    for i in range(0, N, 20):
        fig.add_trace(go.Scatter3d(
            x=[lu_inv[i, 0], lu_ref[i, 0] + lambda_vals[i] * tau_vals[i, 0]],
            y=[lu_inv[i, 1], lu_ref[i, 1] + lambda_vals[i] * tau_vals[i, 1]],
            z=[lu_inv[i, 2], lu_ref[i, 2] + lambda_vals[i] * tau_vals[i, 2]],
            mode="lines", line=dict(color="gray", width=1), showlegend=False
        ))

    fig.update_layout(
        title="Вариант A: z = s, раскладчик вдоль нити",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_along_thread_3d.html")
    print("3D-сцена сохранена: part2_along_thread_3d.html")

    # --- 10. Графики ---
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(s_eval, diff_3d * 1000, "b-", lw=1.0)
    ax.set_title("A. Отклонение ||r_DAE - r_geo||, мм")
    ax.set_xlabel("s"); ax.set_ylabel("мм")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(s_eval, np.abs(Phi_hist) + 1e-16, "g-", lw=1.5)
    ax.axhline(1e-10, color="k", ls="--", lw=0.5)
    ax.set_title("B. Невязка |Ф(s)|")
    ax.set_xlabel("s"); ax.set_ylabel("|Ф|")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(s_eval, diff_u, "r-", lw=1.0, label="|Δu|")
    ax.plot(s_eval, diff_v, "m-", lw=1.0, label="|Δv|")
    ax.set_title("C. Отклонение параметров |Δu|, |Δv|")
    ax.set_xlabel("s"); ax.set_ylabel("|Δ|")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(s_eval[1:], ratio_hist[1:], "c.-", markersize=3, lw=0.8)
    ax.axhline(3.0, color="r", ls="--", lw=0.5, label="C_tol")
    ax.axhline(1/3.0, color="r", ls="--", lw=0.5)
    ax.set_title("D. Адаптивный ratio")
    ax.set_xlabel("s"); ax.set_ylabel("ratio")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_along_thread_diagnostics.png", dpi=150)
    print("Графики сохранены: part2_along_thread_diagnostics.png")
    plt.show()

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()
