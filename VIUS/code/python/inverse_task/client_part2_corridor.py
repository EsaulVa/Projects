#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_part2_corridor_v2.py
===========================
Обратная задача DAE для данных client_corridor_3.py (масштаб ~мм).

Исправления:
  • Интерполяция ТСН на мелкую сетку (N=5000, dz~0.5 мм).
  • Умное начальное условие: проекция R0 на E2.
  • Допуски и скорости масштабированы под мм (eps_Phi=1e-5, max_speed=5000).
  • Отладочный вывод первых 10 шагов.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go
from pathlib import Path
import scipy.io

from dae_helper_v1 import (
    SurfaceGeometryPack,
    TrajectoryByArcLength,
    DAEPredictor,
    NewtonCorrector,
    AdaptiveStepper,
)

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from helpers.intersection import RayTracer, PiecewisePolynomialIntersection


# =============================================================================
# Численный адаптер
# =============================================================================
class NumericalSurfaceAdapter:
    def __init__(self, surface, du=1e-6, dv=1e-6):
        self.surface = surface
        self.du = du
        self.dv = dv

    def position(self, u, v):
        return np.asarray(self.surface.position(u, v), dtype=float)

    def derivatives(self, u, v):
        du, dv = self.du, self.dv
        r = self.position(u, v)
        ru = (self.position(u + du, v) - self.position(u - du, v)) / (2 * du)
        rv = (self.position(u, v + dv) - self.position(u, v - dv)) / (2 * dv)
        n = np.cross(ru, rv)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / norm
        return {"r": r, "ru": ru, "rv": rv, "normal": n}

    def normal(self, u, v):
        return self.derivatives(u, v)["normal"]

    def first_fundamental_form(self, u, v):
        d = self.derivatives(u, v)
        E = float(np.dot(d["ru"], d["ru"]))
        F = float(np.dot(d["ru"], d["rv"]))
        G = float(np.dot(d["rv"], d["rv"]))
        return E, F, G

    def second_fundamental_form(self, u, v):
        du, dv = self.du, self.dv
        r_uu = (self.position(u + du, v) - 2 * self.position(u, v) + self.position(u - du, v)) / (du ** 2)
        r_vv = (self.position(u, v + dv) - 2 * self.position(u, v) + self.position(u, v - dv)) / (dv ** 2)
        r_uv = (self.position(u + du, v + dv) - self.position(u + du, v - dv)
                - self.position(u - du, v + dv) + self.position(u - du, v - dv)) / (4 * du * dv)
        n = self.normal(u, v)
        L = float(np.dot(r_uu, n))
        M = float(np.dot(r_uv, n))
        N = float(np.dot(r_vv, n))
        return L, M, N


# =============================================================================
# Трассировка
# =============================================================================
def trace_ray_to_surface(tracer, surface, origin, direction, t_min=1.0, t_max=2000.0):
    try:
        t, pt = tracer.trace(surface, origin, direction, t_min=t_min, t_max=t_max)
        return t, pt
    except Exception:
        return None, None


# =============================================================================
# Проекция точки на поверхность вращения (поиск ближайшей)
# =============================================================================
def project_point_to_surface(surface_adapter, point_3d, v0=None):
    """
    Для поверхности вращения: ищем u (высоту), минимизируя расстояние.
    v0 = atan2(y,x) — угол, если известен.
    """
    x, y, z = point_3d
    if v0 is None:
        v0 = np.arctan2(y, x)

    def dist_u(u):
        try:
            r = surface_adapter.position(u, v0)
        except Exception:
            return 1e9
        return np.linalg.norm(r - point_3d)

    # Поиск по диапазону u
    u_min, u_max = getattr(surface_adapter.surface, 'u_min', 0.0), getattr(surface_adapter.surface, 'u_max', 1000.0)
    res = minimize_scalar(dist_u, bounds=(u_min, u_max), method='bounded')
    u_opt = res.x
    r_opt = surface_adapter.position(u_opt, v0)
    return u_opt, v0, r_opt, res.fun


# =============================================================================
# Основной блок
# =============================================================================
def main():
    # --- Поверхности ---
    phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                     -0.0099656628535, 2.9503573330764]
    R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
                   39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
    bound_opravka = [0, 234.27, 534.27, 768.54]
    cyl_r_opravka = 251.705

    phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076,
                  -0.0066486075257, 2.9473869159379]
    R_c_safe = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463,
                27152.4105364360366, -31195.5446114188999, 14397.6607910855146]
    bound_safe = [0, 327.978, 627.978, 955.956]
    cyl_r_safe = 352.387

    E2_raw = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)
    E1_raw = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)
    E2 = NumericalSurfaceAdapter(E2_raw)
    E1 = NumericalSurfaceAdapter(E1_raw)

    print("=" * 70)
    print("Обратная задача DAE для corridor_3 (масштаб мм)")
    print(f"E2 u-range: {getattr(E2_raw, 'u_min', 0):.1f} .. {getattr(E2_raw, 'u_max', 0):.1f}")
    print("=" * 70)

    # --- Загрузка ТСН ---
    csv_path = Path("tsn_shadow.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите client_corridor_3.py.")
        return

    df_tsn = pd.read_csv(csv_path)
    valid = df_tsn["valid"].values.astype(bool)
    tsn_pts_raw = df_tsn[["X", "Y", "Z"]].values[valid]
    print(f"Загружено {len(tsn_pts_raw)} валидных точек ТСН")

    

    # # --- Интерполяция ТСН на мелкую сетку ---
    N_fine = 5000
    tsn_pts=tsn_pts_raw
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter

    # --- Вариант А: Гаусс (σ = 3 точки) ---
    sigma_pts = 3.0
    tsn_smooth = np.zeros_like(tsn_pts)
    for dim in range(3):
        tsn_smooth[:, dim] = gaussian_filter1d(tsn_pts[:, dim], sigma=sigma_pts)

    # --- Вариант Б: Савицки-Голай (окно 15, полином 3-й степени) ---
    # Сохраняет первые производные лучше, чем Гаусс
    window = min(15, len(tsn_pts) // 2 * 2 + 1)  # нечётное, < len
    tsn_smooth = savgol_filter(tsn_pts, window_length=window, polyorder=3, axis=0)

    traj = TrajectoryByArcLength(tsn_smooth)  # вместо tsn_pts
    # s_raw = np.linspace(0, 1, len(tsn_pts_raw))
    # s_fine = np.linspace(0, 1, N_fine)

    # tsn_fine = np.zeros((N_fine, 3))
    # for dim in range(3):
    #     cs = CubicSpline(s_raw, tsn_pts_raw[:, dim])
    #     tsn_fine[:, dim] = cs(s_fine)

    # print(f"Интерполировано на {N_fine} точек (шаг ~{np.linalg.norm(tsn_fine[-1]-tsn_fine[0])/N_fine:.2f} мм)")

    # # --- Траектория ---
    # traj = TrajectoryByArcLength(tsn_fine)
    z_end = traj.total_length
    # print(f"Длина траектории ТСН (z): {z_end:.2f} мм")
    

    # --- Начальное условие: проекция R0 на E2 ---
    R0 = tsn_smooth[0]
    u0, v0, r0_proj, err_proj = project_point_to_surface(E2, R0)
    print(f"\nПроекция R0 на E2: u0={u0:.4f}, v0={v0:.4f}, ошибка={err_proj:.4f} мм")

    # Проверка Phi в начальной точке
    Phi0 = np.dot(R0 - r0_proj, E2.normal(u0, v0))
    print(f"Начальная Φ = {Phi0:.4f} мм")

    # --- Загрузка эталона ЛУ ---
    mat_path = Path("LU_data.mat")
    lu_ref = None
    s_ref = None
    if mat_path.exists():
        data = scipy.io.loadmat(str(mat_path))
        lu_ref = np.asarray(data["r"], dtype=float)
        if lu_ref.ndim == 2 and lu_ref.shape[0] == 3 and lu_ref.shape[1] != 3:
            lu_ref = lu_ref.T
        print(f"Загружена эталонная ЛУ: {lu_ref.shape[0]} точек")
        s_ref = np.zeros(len(lu_ref))
        for i in range(1, len(lu_ref)):
            s_ref[i] = s_ref[i - 1] + np.linalg.norm(lu_ref[i] - lu_ref[i - 1])
    else:
        print("LU_data.mat не найден — сравнение с эталоном отключено")

    # --- Обратная задача DAE ---
    print(f"\nРешение обратной задачи DAE...")
    predictor = DAEPredictor(max_speed=5000.0)
    corrector = NewtonCorrector(eps_Phi=1e-5, max_iter=20)
    stepper = AdaptiveStepper(predictor, corrector, C_tol=10.0, max_bisect=6)

    # Начальная коррекция
    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    if not corr0.converged:
        print(f"ВНИМАНИЕ: начальная коррекция не сошлась! Φ={corr0.Phi:.4f}")
        u_cur, v_cur = u0, v0
    else:
        u_cur, v_cur = corr0.u_corr, corr0.v_corr
        print(f"Начальная коррекция: Ф0={corr0.Phi:.2e}, итераций={corr0.iterations}")

    num_points = N_fine
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
    conv_flags = np.zeros(N, dtype=bool)

    u_hist[0], v_hist[0], s_hist[0] = u_cur, v_cur, 0.0

    du0, dv0, _, _ = predictor.compute_velocity(E2, traj, u_cur, v_cur, 0.0)
    geom0 = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
    sigma_hist[0] = geom0.metric_speed(np.array([du0, dv0]))

    # --- Цикл интегрирования ---
    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        dz_step = z_next - z_k

        result = stepper.step(E2, traj, u_cur, v_cur, z_k, dz_step)
        u_next, v_next = result.u_next, result.v_next

        # Точное накопление s
        sigma_k = sigma_hist[i]
        du_next, dv_next, _, _ = predictor.compute_velocity(E2, traj, u_next, v_next, z_next)
        try:
            geom_next = SurfaceGeometryPack.from_surface(E2, u_next, v_next)
            sigma_next = geom_next.metric_speed(np.array([du_next, dv_next]))
        except Exception:
            sigma_next = sigma_k
        sigma_hist[i + 1] = sigma_next
        s_next = s_hist[i] + 0.5 * (sigma_k + sigma_next) * dz_step

        u_cur, v_cur = u_next, v_next
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        s_hist[i + 1] = s_next
        Phi_hist[i + 1] = result.Phi
        newton_iters_hist[i + 1] = result.iterations
        bisect_flags[i + 1] = 1 if result.bisected else 0
        ratio_hist[i + 1] = result.ratio
        conv_flags[i + 1] = result.converged

        # Отладка первых 10 шагов
        if i < 10:
            print(f"  step {i}: z={z_k:.2f}, Φ={result.Phi:.2e}, "
                  f"it={result.iterations}, bisect={result.bisected}, "
                  f"conv={result.converged}, ratio={result.ratio:.3f}")

    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])
    print(f"\nВосстановленная ЛУ: {N} точек, длина s={s_hist[-1]:.2f} мм")
    print(f"Несошедших шагов: {np.sum(~conv_flags)}/{N}")

    # =====================================================================
    # Верификация: трассировка ТСН от восстановленной ЛУ
    # =====================================================================
    print(f"\n{'='*70}")
    print("Верификация: трассировка ТСН")
    print(f"{'='*70}")

    tracer = RayTracer()
    tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())

    tsn_recon = np.zeros((N, 3))
    valid_recon = np.zeros(N, dtype=bool)
    lambda_recon = np.zeros(N)

    for i in range(N):
        r = lu_inv[i]
        R_target = traj.R(z_eval[i])
        V_thread = R_target - r
        lam = np.linalg.norm(V_thread)
        if lam < 1e-9:
            continue
        tau = V_thread / lam
        m = E2.normal(u_hist[i], v_hist[i])
        tau_proj = tau - np.dot(tau, m) * m
        norm_proj = np.linalg.norm(tau_proj)
        if norm_proj < 1e-9:
            continue
        tau_proj = tau_proj / norm_proj

        t, pt = trace_ray_to_surface(tracer, E1_raw, r, tau_proj, t_min=1.0, t_max=2000.0)
        if t is not None:
            tsn_recon[i] = pt
            lambda_recon[i] = t
            valid_recon[i] = True

    if np.sum(valid_recon) > 0:
        R_etalon = np.array([traj.R(z) for z in z_eval[valid_recon]])
        diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - R_etalon, axis=1)
        print(f"Валидных точек: {np.sum(valid_recon)}/{N}")
        print(f"Max ||ДТСН|| = {np.max(diff_tsn):.3f} мм")
        print(f"Mean ||ДТСН|| = {np.mean(diff_tsn):.3f} мм")
    else:
        print("Трассировка ТСН не дала валидных точек!")

    # =====================================================================
    # Сравнение с эталоном
    # =====================================================================
    diff_lu = None
    if lu_ref is not None and s_ref is not None:
        print(f"\n{'='*70}")
        print("Сравнение с эталонной ЛУ")
        print(f"{'='*70}")
        s_max = min(s_hist[-1], s_ref[-1])
        if s_max > 1e-3:
            n_cmp = min(N, len(lu_ref))
            s_common = np.linspace(0, s_max, n_cmp)

            u_inv_s = CubicSpline(s_hist, u_hist)
            v_inv_s = CubicSpline(s_hist, v_hist)
            lu_inv_at_s = np.array([E2.position(u_inv_s(s), v_inv_s(s)) for s in s_common])

            fx = interp1d(s_ref, lu_ref[:, 0], kind="cubic", fill_value="extrapolate")
            fy = interp1d(s_ref, lu_ref[:, 1], kind="cubic", fill_value="extrapolate")
            fz = interp1d(s_ref, lu_ref[:, 2], kind="cubic", fill_value="extrapolate")
            lu_ref_at_s = np.column_stack([fx(s_common), fy(s_common), fz(s_common)])

            diff_lu = np.linalg.norm(lu_inv_at_s - lu_ref_at_s, axis=1)
            print(f"Max ||r_DAE - r_ref|| = {np.max(diff_lu):.3f} мм")
            print(f"Mean ||r_DAE - r_ref|| = {np.mean(diff_lu):.3f} мм")
        else:
            print("Недостаточная длина для сравнения")

    # =====================================================================
    # Метрики
    # =====================================================================
    print(f"\n{'='*70}")
    print("МЕТРИКИ DAE")
    print(f"{'='*70}")
    print(f"Max |Ф|        = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Ф|       = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Шагов с бисекцией = {np.sum(bisect_flags)}/{N}")
    print(f"Max ratio      = {np.max(np.abs(ratio_hist[1:])):.4f}")
    print(f"Min ratio      = {np.min(np.abs(ratio_hist[1:])):.4f}")
    print(f"Длина ЛУ s     = {s_hist[-1]:.2f} мм")
    print(f"Длина ТСН z    = {z_eval[-1]:.2f} мм")
    print(f"Среднее ds/dz  = {np.mean(sigma_hist):.4f}")

    # =====================================================================
    # CSV
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
        "converged": conv_flags,
    })
    df_out.to_csv("part2_corridor_inverse.csv", index=False)
    print(f"\nCSV: part2_corridor_inverse.csv")

    # =====================================================================
    # 3D
    # =====================================================================
    fig = go.Figure()

    if lu_ref is not None:
        fig.add_trace(go.Scatter3d(
            x=lu_ref[:, 0], y=lu_ref[:, 1], z=lu_ref[:, 2],
            mode="lines", line=dict(color="blue", width=4, dash="dash"),
            name="ЛУ эталон"
        ))

    fig.add_trace(go.Scatter3d(
        x=lu_inv[:, 0], y=lu_inv[:, 1], z=lu_inv[:, 2],
        mode="lines", line=dict(color="green", width=4),
        name="ЛУ DAE"
    ))
    fig.add_trace(go.Scatter3d(
        x=tsn_fine[:, 0], y=tsn_fine[:, 1], z=tsn_fine[:, 2],
        mode="lines", line=dict(color="red", width=3),
        name="ТСН"
    ))

    if np.sum(valid_recon) > 0:
        fig.add_trace(go.Scatter3d(
            x=tsn_recon[valid_recon, 0],
            y=tsn_recon[valid_recon, 1],
            z=tsn_recon[valid_recon, 2],
            mode="lines", line=dict(color="magenta", width=3, dash="dot"),
            name="ТСН recon"
        ))

    step = max(1, N // 50)
    for i in range(0, N, step):
        fig.add_trace(go.Scatter3d(
            x=[lu_inv[i, 0], tsn_fine[i, 0]],
            y=[lu_inv[i, 1], tsn_fine[i, 1]],
            z=[lu_inv[i, 2], tsn_fine[i, 2]],
            mode="lines", line=dict(color="gray", width=1), showlegend=False
        ))

    fig.update_layout(
        title="Обратная задача DAE (corridor_3, мм)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_corridor_3d.html")
    print("3D: part2_corridor_3d.html")
    fig.show()

    # =====================================================================
    # Графики
    # =====================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(z_eval, np.abs(Phi_hist) + 1e-16, "g-", lw=1.0)
    ax.axhline(1e-5, color="k", ls="--", lw=0.5)
    ax.set_title("Невязка |Ф(z)|")
    ax.set_xlabel("z, мм")
    ax.set_ylabel("|Ф|, мм")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(z_eval, s_hist, "c-", lw=1.5, label="s(z)")
    ax.plot(z_eval, z_eval, "k--", lw=0.8, label="s=z")
    ax.set_title("Накопленная длина ЛУ")
    ax.set_xlabel("z, мм")
    ax.set_ylabel("s, мм")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(z_eval[1:], newton_iters_hist[1:], "m.", markersize=3)
    ax.set_title("Итерации корректора")
    ax.set_xlabel("z, мм")
    ax.set_ylabel("итерации")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if diff_lu is not None:
        ax.plot(s_common, diff_lu, "b-", lw=1.0)
        ax.set_title("Отклонение от эталона")
        ax.set_xlabel("s, мм")
        ax.set_ylabel("||Δr||, мм")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Эталон не загружен", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("part2_corridor_diagnostics.png", dpi=150)
    print("Графики: part2_corridor_diagnostics.png")
    plt.show()

    print("\nГОТОВО")


if __name__ == "__main__":
    main()