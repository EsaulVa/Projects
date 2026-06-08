#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_part2_collinear.py
=========================
Обратная задача с предиктором на базе коллинеарного замыкания (п.6).

Предиктор двигает точку укладки строго вдоль направления нити tau(s),
используя явную формулу:

    u'(z) = <R'(z), m> / (P^T G^{-1} B G^{-1} P) * G^{-1} P

где P = (<V_thread, e1>, <V_thread, e2>)^T.

Это частное решение уравнения связи dPhi/dz=0 при условии,
что скорость точки укладки коллинеарна направлению нити.

В паре: PredictorCollinear + NewtonCorrector + AdaptiveStepper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass

from dae_helper_v1 import (
    Ellipsoid,
    SurfaceGeometryPack,
    TrajectoryByArcLength,
    NewtonCorrector,    
)
from dae_helper_v1.adaptive_stepper_1 import AdaptiveStepper


# =======================================================================
# Вспомогательный класс результата Эйлера
# =======================================================================
class EulerResult(NamedTuple):
    u_pred: float
    v_pred: float
    z_pred: float
    Phi_pred: float
    V_thread: np.ndarray
    du_dz: float
    dv_dz: float


# =======================================================================
# Предиктор: коллинеарное замыкание (движение вдоль нити)
# =======================================================================
@dataclass
class PredictorCollinear:
    """
    Предиктор, основанный на формуле (4):
    u' = (dPhi/dz) / (P^T G^{-1} B G^{-1} P) * G^{-1} P
    """
    max_speed: float = 50.0
    eps: float = 1e-14

    def compute_velocity(self, surface, traj, u, v, z):
        """
        Вычисляет скорость (du/dz, dv/dz) по коллинеарной формуле.
        Возвращает: du, dv, Phi, V_thread
        """
        geom = SurfaceGeometryPack.from_surface(surface, u, v)
        R = traj.R(z)
        r = surface.position(u, v)
        V_thread = R - r
        lam = np.linalg.norm(V_thread)
        if lam < self.eps:
            return 0.0, 0.0, 0.0, V_thread

        # Проекции нити на базис
        P = np.array([
            float(np.dot(V_thread, geom.ru)),
            float(np.dot(V_thread, geom.rv))
        ])

        # Частная производная зазора по z
        dPhi_dz = float(np.dot(traj.R_deriv(z), geom.normal))

        # Знаменатель: P^T G^{-1} B G^{-1} P
        GinvP = geom.G_inv @ P
        BGinvP = geom.B @ GinvP
        denom = float(GinvP @ BGinvP)

        if abs(denom) < self.eps:
            # Вырождение: поверхность локально плоская (B ~ 0)
            # Возвращаем нулевую скорость, корректор будет удерживать связь
            u_prime = np.array([0.0, 0.0])
        else:
            factor = dPhi_dz / denom
            u_prime = factor * GinvP

        # Невязка связи
        Phi = float(np.dot(V_thread, geom.normal))

        # Ограничение скорости
        speed = geom.metric_speed(u_prime)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            u_prime *= scale

        return float(u_prime[0]), float(u_prime[1]), Phi, V_thread

    def predict_step(self, surface, traj, u, v, z, dz):
        du, dv, Phi, V_thread = self.compute_velocity(surface, traj, u, v, z)
        u_pred = u + du * dz
        v_pred = v + dv * dz
        z_pred = z + dz
        return EulerResult(u_pred, v_pred, z_pred, Phi, V_thread, du, dv)

    def _euler_step(self, surface, traj, u, v, z, dz):
        return self.predict_step(surface, traj, u, v, z, dz)


# =======================================================================
# Геодезическая (эталон)
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
        raise RuntimeError(f"Прямая задача не сошлась: {sol.message}")

    uv = sol.y.T
    points = np.array([surface.position(u, v) for u, v in uv[:, :2]])
    return sol.t, uv, points


# =======================================================================
# Трассировка луча
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
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 70)
    print("Предиктор: коллинеарное замыкание (движение вдоль нити)")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")
    print("=" * 70)

    # --- Загрузка ТСН ---
    csv_path = Path("part1_tsn_shadow.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите Part 1.")
        return

    df = pd.read_csv(csv_path)
    valid = df["valid"].values.astype(bool)
    tsn_pts = df[["X", "Y", "Z"]].values[valid]
    print(f"\nЗагружено {len(tsn_pts)} валидных точек ТСН")

    traj = TrajectoryByArcLength(tsn_pts)
    z_end = traj.total_length
    print(f"Длина траектории ТСН (z): {z_end:.4f}")

    # --- Эталон ---
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = z_end
    num_points = len(tsn_pts)
    print(f"\nПостроение эталонной ЛУ...")
    s_ref, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Эталон: {len(lu_ref)} точек, s={s_ref[-1]:.4f}")

    # --- Обратная задача: коллинеарный предиктор ---
    print(f"\nРешение обратной задачи (коллинеарный предиктор)...")
    predictor = PredictorCollinear(max_speed=100.0)
    corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)
    stepper = AdaptiveStepper(predictor, corrector=None, C_tol=3.0, max_bisect=4,use_corrector=False)

    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr
    s_cur = 0.0
    print(f"Начальная точка: Ф0={corr0.Phi:.2e}, итераций={corr0.iterations}")

    z_eval = np.linspace(0, z_end, num_points)
    dz = z_eval[1] - z_eval[0]
    N = num_points

    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    s_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    bisect_flags = np.zeros(N, dtype=int)
    ratio_hist = np.zeros(N)
    sigma_hist = np.zeros(N)

    u_hist[0], v_hist[0], s_hist[0] = u_cur, v_cur, s_cur

    du0, dv0, _, _ = predictor.compute_velocity(E2, traj, u_cur, v_cur, 0.0)
    geom0 = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
    sigma_hist[0] = geom0.metric_speed(np.array([du0, dv0]))

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        dz_step = z_next - z_k

        result = stepper.step(E2, traj, u_cur, v_cur, z_k, dz_step)
        u_next, v_next = result.u_next, result.v_next

        sigma_k = sigma_hist[i]
        du_next, dv_next, _, _ = predictor.compute_velocity(E2, traj, u_next, v_next, z_next)
        geom_next = SurfaceGeometryPack.from_surface(E2, u_next, v_next)
        sigma_next = geom_next.metric_speed(np.array([du_next, dv_next]))
        sigma_hist[i + 1] = sigma_next

        s_next = s_cur + 0.5 * (sigma_k + sigma_next) * dz_step

        u_cur, v_cur, s_cur = u_next, v_next, s_next
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        s_hist[i + 1] = s_cur
        Phi_hist[i + 1] = result.Phi
        newton_iters_hist[i + 1] = result.iterations
        bisect_flags[i + 1] = 1 if result.bisected else 0
        ratio_hist[i + 1] = result.ratio

    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])
    print(f"Восстановленная ЛУ: s={s_hist[-1]:.4f}")

    # =====================================================================
    # Верификация 1: сравнение с геодезической по s
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 1: Сравнение по s")
    print(f"{'='*70}")

    s_unique, idx_unique = np.unique(s_hist, return_index=True)
    u_unique = u_hist[idx_unique]
    v_unique = v_hist[idx_unique]

    u_inv_s = CubicSpline(s_unique, u_unique)
    v_inv_s = CubicSpline(s_unique, v_unique)
    u_ref_s = CubicSpline(s_ref, uv_ref[:, 0])
    v_ref_s = CubicSpline(s_ref, uv_ref[:, 1])

    s_max = min(s_unique[-1], s_ref[-1])
    s_common = np.linspace(0, s_max, 2000)

    lu_inv_at_s = np.array([E2.position(u_inv_s(s), v_inv_s(s)) for s in s_common])
    lu_ref_at_s = np.array([E2.position(u_ref_s(s), v_ref_s(s)) for s in s_common])
    diff_s = np.linalg.norm(lu_inv_at_s - lu_ref_at_s, axis=1)

    print(f"Max ||r_col - r_geo|| = {np.max(diff_s):.6f}")
    print(f"Mean ||r_col - r_geo|| = {np.mean(diff_s):.6f}")

    # =====================================================================
    # Верификация 2: остатки геодезичности
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 2: Остатки геодезичности")
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
    # Верификация 3: трассировка ТСН
    # =====================================================================
    print(f"\n{'='*70}")
    print("ВЕРИФИКАЦИЯ 3: Трассировка ТСН")
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
        print(f"Валидных: {np.sum(valid_recon)}/{N}")
        print(f"Max ||ДТСН|| = {np.max(diff_tsn):.6f}")
        print(f"Mean ||ДТСН|| = {np.mean(diff_tsn):.6f}")

    # =====================================================================
    # Метрики
    # =====================================================================
    print(f"\n{'='*70}")
    print("МЕТРИКИ")
    print(f"{'='*70}")
    print(f"Max |Ф| = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Ф| = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Шагов с бисекцией = {np.sum(bisect_flags)}/{N}")
    print(f"Max ratio = {np.max(np.abs(ratio_hist[1:])):.4f}")
    print(f"Min ratio = {np.min(np.abs(ratio_hist[1:])):.4f}")
    print(f"Длина ЛУ s = {s_hist[-1]:.4f}")
    print(f"Длина ТСН z = {z_eval[-1]:.4f}")
    print(f"Среднее ds/dz = {np.mean(sigma_hist):.4f}")

    # =====================================================================
    # Сохранение
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
    df_out.to_csv("part2_collinear.csv", index=False)
    print(f"\nCSV сохранен: part2_collinear.csv")

    # =====================================================================
    # 3D
    # =====================================================================
    fig = go.Figure()
    u_e = np.linspace(0, 2 * np.pi, 60)
    v_e = np.linspace(-np.pi / 2, np.pi / 2, 40)
    Ue, Ve = np.meshgrid(u_e, v_e)

    X1 = a1 * np.cos(Ue) * np.cos(Ve)
    Y1 = b1 * np.sin(Ue) * np.cos(Ve)
    Z1 = c1 * np.sin(Ve)
    fig.add_trace(go.Surface(x=X1, y=Y1, z=Z1, opacity=0.15, colorscale="Blues",
                             showscale=False, name="E1"))

    X2 = a2 * np.cos(Ue) * np.cos(Ve)
    Y2 = b2 * np.sin(Ue) * np.cos(Ve)
    Z2 = c2 * np.sin(Ve)
    fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.25, colorscale="Reds",
                             showscale=False, name="E2"))

    fig.add_trace(go.Scatter3d(x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
                                 mode="lines", line=dict(color="black", width=4),
                                 name="ТСН"))
    fig.add_trace(go.Scatter3d(x=lu_ref[:, 0], y=lu_ref[:, 1], z=lu_ref[:, 2],
                                 mode="lines", line=dict(color="blue", width=3, dash="dash"),
                                 name="ЛУ эталон"))
    fig.add_trace(go.Scatter3d(x=lu_inv[:, 0], y=lu_inv[:, 1], z=lu_inv[:, 2],
                                 mode="lines", line=dict(color="orange", width=3),
                                 name="ЛУ коллинеарный"))

    fig.update_layout(
        title="Коллинеарный предиктор (движение вдоль нити)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_collinear_3d.html")
    print("3D-сцена: part2_collinear_3d.html")

    # =====================================================================
    # Графики
    # =====================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(s_common, diff_s * 1000, "b-", lw=1.0)
    ax.set_title("A. Отклонение ||r_col - r_geo||, мм")
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
    ax.set_title("C. Накопленная длина ЛУ")
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
    ax.set_title("F. Итерации корректора")
    ax.set_xlabel("z"); ax.set_ylabel("итерации")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_collinear_diagnostics.png", dpi=150)
    print("Графики: part2_collinear_diagnostics.png")
    plt.show()

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)


if __name__ == "__main__":
    main()