#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part2_ellipsoid_inverse_dae.py
================================
Часть 2:
1. Загружает ТСН из CSV (part1_tsn_shadow.csv).
2. Строит траекторию раскладчика R(z) по точкам ТСН.
3. Решает обратную задачу намотки на внутренний эллипсоид E2
   методом DAE-предиктор–корректор (отчёт 5.06.2026).
4. Сравнивает восстановленную ЛУ с эталонной (геодезическая на E2).
5. Сохраняет результаты и строит визуализацию.

Параметры:
  E1 (внешний): a=3.0, b=2.5, c=2.0
  E2 (внутренний): scale=0.8
  Начальные условия: u0=π/3, v0=π/6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from dataclasses import dataclass
from pathlib import Path
from dae_helper.dae_predictor import *
from dae_helper.newton_corrector import *


# =====================================================================
# 1. КЛАСС ЭЛЛИПСОИДА (совместимый с SurfaceGeometryPack)
# =====================================================================
class Ellipsoid:
    def __init__(self, a, b, c):
        self.a, self.b, self.c = a, b, c

    def position(self, u, v):
        a, b, c = self.a, self.b, self.c
        return np.array([
            a * np.cos(u) * np.cos(v),
            b * np.sin(u) * np.cos(v),
            c * np.sin(v)
        ])

    def normal(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        n = np.array([cos_u * cos_v / a, sin_u * cos_v / b, sin_v / c])
        return n / np.linalg.norm(n)

    def derivatives(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        r = self.position(u, v)
        ru = np.array([-a * sin_u * cos_v, b * cos_u * cos_v, 0.0])
        rv = np.array([-a * cos_u * sin_v, -b * sin_u * sin_v, c * cos_v])
        return {"r": r, "ru": ru, "rv": rv, "normal": self.normal(u, v)}

    def first_fundamental_form(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        E = a**2 * sin_u**2 * cos_v**2 + b**2 * cos_u**2 * cos_v**2
        F = (a**2 - b**2) * sin_u * cos_u * sin_v * cos_v
        G = a**2 * cos_u**2 * sin_v**2 + b**2 * sin_u**2 * sin_v**2 + c**2 * cos_v**2
        return E, F, G
    
    def second_fundamental_form(self, u, v):
        """Правильные аналитические формулы для эллипсоида."""
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        denom = np.sqrt(
            (cos_u * cos_v / self.a)**2 +
            (sin_u * cos_v / self.b)**2 +
            (sin_v / self.c)**2
        )
        # r_uu = (-a cos_u cos_v, -b sin_u cos_v, 0)
        # r_vv = (-a cos_u cos_v, -b sin_u cos_v, -c sin_v)
        # m = (cos_u cos_v / a, sin_u cos_v / b, sin_v / c) / denom  (наружу)
        L = -cos_v**2 / denom
        M = 0.0
        N = -1.0 / denom
        return L, M, N

    # def second_fundamental_form(self, u, v):
    #     """Аналитические формулы для эллипсоида (как в старом проекте)."""
    #     cos_u, sin_u = np.cos(u), np.sin(u)
    #     cos_v, sin_v = np.cos(v), np.sin(v)
    #     denom = np.sqrt(
    #         (cos_u * cos_v / self.a)**2 +
    #         (sin_u * cos_v / self.b)**2 +
    #         (sin_v / self.c)**2
    #     )
    #     L = self.a * self.b * self.c * cos_v / (self.a**2 * denom**3)
    #     M = 0.0
    #     N = self.a * self.b * self.c / (self.c**2 * denom**3)
    #     return L, M, N

    def metric_derivatives(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        E_u = 2 * (a**2 - b**2) * sin_u * cos_u * cos_v**2
        E_v = -2 * cos_v * sin_v * (a**2 * sin_u**2 + b**2 * cos_u**2)
        F_u = (a**2 - b**2) * np.cos(2*u) * sin_v * cos_v
        F_v = (a**2 - b**2) * sin_u * cos_u * np.cos(2*v)
        G_u = 2 * (b**2 - a**2) * sin_u * cos_u * sin_v**2
        G_v = 2 * sin_v * cos_v * (a**2 * cos_u**2 + b**2 * sin_u**2 - c**2)
        return E_u, E_v, F_u, F_v, G_u, G_v

    def christoffel_symbols(self, u, v):
        E, F, G = self.first_fundamental_form(u, v)
        E_u, E_v, F_u, F_v, G_u, G_v = self.metric_derivatives(u, v)
        det = E * G - F**2
        if abs(det) < 1e-14:
            raise ValueError("Вырожденная метрика")
        inv_det = 1.0 / det
        g11, g12, g22 = G * inv_det, -F * inv_det, E * inv_det
        Gamma = np.zeros((2, 2, 2))
        Gamma[0, 0, 0] = 0.5 * (g11 * E_u + g12 * (2.0 * F_u - E_v))
        Gamma[0, 0, 1] = 0.5 * (g11 * E_v + g12 * G_u)
        Gamma[0, 1, 0] = Gamma[0, 0, 1]
        Gamma[0, 1, 1] = 0.5 * (g11 * (2.0 * F_v - G_u) + g12 * G_v)
        Gamma[1, 0, 0] = 0.5 * (g12 * E_u + g22 * (2.0 * F_u - E_v))
        Gamma[1, 0, 1] = 0.5 * (g12 * E_v + g22 * G_u)
        Gamma[1, 1, 0] = Gamma[1, 0, 1]
        Gamma[1, 1, 1] = 0.5 * (g12 * (2.0 * F_v - G_u) + g22 * G_v)
        return Gamma


# =====================================================================
# 2. ПРЯМАЯ ЗАДАЧА: геодезическая (для эталона)
# =====================================================================
def solve_geodesic(surface, u0, v0, alpha, s_end, num_points=300):
    """Строит геодезическую на поверхности."""
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


# =====================================================================
# 3. SurfaceGeometryPack (DAE-ядро)
# =====================================================================
@dataclass(frozen=True)
class SurfaceGeometryPack:
    r: np.ndarray
    ru: np.ndarray
    rv: np.ndarray
    normal: np.ndarray
    E: float
    F: float
    G: float
    L: float
    M: float
    N: float
    G_inv: np.ndarray
    B: np.ndarray
    det_G: float

    @classmethod
    def from_surface(cls, surface, u, v):
        d = surface.derivatives(u, v)
        E, F, G = surface.first_fundamental_form(u, v)
        L, M, N = surface.second_fundamental_form(u, v)
        det = E * G - F * F
        if abs(det) < 1e-14:
            raise ValueError(f"Вырожденная метрика: det={det}")
        inv = 1.0 / det
        return cls(
            r=d["r"], ru=d["ru"], rv=d["rv"], normal=d["normal"],
            E=E, F=F, G=G, L=L, M=M, N=N,
            G_inv=np.array([[G*inv, -F*inv], [-F*inv, E*inv]]),
            B=np.array([[L, M], [M, N]]),
            det_G=det
        )

    def project_on_basis(self, vec):
        return np.array([np.dot(vec, self.ru), np.dot(vec, self.rv)])

    def grad_Phi(self, V_thread):
        P = self.project_on_basis(V_thread)
        return -self.B @ self.G_inv @ P

    def surface_gradient(self, grad_u):
        return self.G_inv @ grad_u

    def norm_grad_sq(self, grad_u):
        return float(grad_u @ self.surface_gradient(grad_u))

    def base_velocity(self, R_prime):
        P_R = self.project_on_basis(R_prime)
        return self.G_inv @ P_R

    def dPhi_dz(self, R_prime):
        return float(np.dot(R_prime, self.normal))

    def compute_mu(self, R_prime, V_thread):
        dphi = self.dPhi_dz(R_prime)
        grad_u = self.grad_Phi(V_thread)
        Rp = self.base_velocity(R_prime)
        Ng = self.norm_grad_sq(grad_u)
        if Ng < 1e-14:
            return 0.0
        return -(dphi + float(grad_u @ Rp)) / Ng

    def winding_velocity(self, R_prime, V_thread):
        Rp = self.base_velocity(R_prime)
        grad_u = self.grad_Phi(V_thread)
        grad_s = self.surface_gradient(grad_u)
        mu = self.compute_mu(R_prime, V_thread)
        return Rp + mu * grad_s

    def normal_curvature(self, direction):
        du, dv = direction[0], direction[1]
        II = self.L * du**2 + 2.0 * self.M * du * dv + self.N * dv**2
        I = self.E * du**2 + 2.0 * self.F * du * dv + self.G * dv**2
        return II / I if abs(I) > 1e-15 else 0.0

    def metric_speed(self, vec_uv):
        du, dv = vec_uv[0], vec_uv[1]
        return np.sqrt(max(self.E * du**2 + 2*self.F * du * dv + self.G * dv**2, 0.0))

    @staticmethod
    def compute_Phi(R, r, normal):
        return float(np.dot(R - r, normal))


# =====================================================================
# 4. DAEPredictor
# =====================================================================
@dataclass(frozen=True)
class PredictorResult:
    u_pred: float; v_pred: float
    du_dz: float; dv_dz: float
    R_prime: np.ndarray; V_thread: np.ndarray
    mu: float; grad_u: np.ndarray; grad_s: np.ndarray
    Ng: float; dPhi_dz: float; Rp_parallel: np.ndarray
    speed: float; optical_fallback: bool


# class DAEPredictor:
#     def __init__(self, max_speed=50.0):
#         self.max_speed = max_speed

#     def predict_step(self, geom, traj, u, v, z, dz):
#         R = traj.R(z)
#         V_thread = R - geom.r
#         R_prime = traj.R_deriv(z)
#         grad_u = geom.grad_Phi(V_thread)
#         grad_s = geom.surface_gradient(grad_u)
#         Ng = geom.norm_grad_sq(grad_u)
#         dphi_dz = geom.dPhi_dz(R_prime)
#         Rp_parallel = geom.base_velocity(R_prime)
#         optical_fallback = Ng < 1e-14
#         mu = 0.0 if optical_fallback else -(dphi_dz + float(grad_u @ Rp_parallel)) / Ng
#         u_prime = Rp_parallel + mu * grad_s
#         du_dz, dv_dz = float(u_prime[0]), float(u_prime[1])
#         speed = geom.metric_speed(u_prime)
#         if self.max_speed is not None and speed > self.max_speed:
#             scale = self.max_speed / speed
#             du_dz *= scale; dv_dz *= scale
#             u_prime = np.array([du_dz, dv_dz])
#             speed = self.max_speed
#         return PredictorResult(
#             u_pred=u + du_dz * dz, v_pred=v + dv_dz * dz,
#             du_dz=du_dz, dv_dz=dv_dz, R_prime=R_prime, V_thread=V_thread,
#             mu=mu, grad_u=grad_u, grad_s=grad_s, Ng=Ng, dPhi_dz=dphi_dz,
#             Rp_parallel=Rp_parallel, speed=speed, optical_fallback=optical_fallback
#         )


# =====================================================================
# 5. NewtonCorrector
# =====================================================================
@dataclass(frozen=True)
class CorrectorResult:
    u_corr: float; v_corr: float
    Phi: float; iterations: int; converged: bool


# class NewtonCorrector:
#     def __init__(self, eps_Phi=1e-10, max_iter=7, v_periodic=False, v_period=2*np.pi):
#         self.eps_Phi = eps_Phi
#         self.max_iter = max_iter
#         self.v_periodic = v_periodic
#         self.v_period = v_period

#     def correct(self, surface, traj, u_pred, v_pred, z_target):
#         u_c, v_c = float(u_pred), float(v_pred)
#         for nit in range(self.max_iter):
#             try:
#                 geom = SurfaceGeometryPack.from_surface(surface, u_c, v_c)
#             except ValueError:
#                 return CorrectorResult(u_c, v_c, np.nan, nit, False)
#             R_t = traj.R(z_target)
#             Phi = SurfaceGeometryPack.compute_Phi(R_t, geom.r, geom.normal)
#             if abs(Phi) < self.eps_Phi:
#                 return CorrectorResult(u_c, v_c, Phi, nit, True)
#             V_thread = R_t - geom.r
#             grad_u = geom.grad_Phi(V_thread)
#             grad_s = geom.surface_gradient(grad_u)
#             Ng = geom.norm_grad_sq(grad_u)
#             if Ng < 1e-14:
#                 return CorrectorResult(u_c, v_c, Phi, nit, False)
#             delta = -Phi / Ng * grad_s
#             u_c += float(delta[0])
#             v_c += float(delta[1])
#             if self.v_periodic:
#                 half = self.v_period / 2.0
#                 v_c = ((v_c + half) % self.v_period) - half
#         # Финальная проверка
#         try:
#             geom = SurfaceGeometryPack.from_surface(surface, u_c, v_c)
#         except ValueError:
#             return CorrectorResult(u_c, v_c, np.nan, self.max_iter, False)
#         Phi = SurfaceGeometryPack.compute_Phi(traj.R(z_target), geom.r, geom.normal)
#         return CorrectorResult(u_c, v_c, Phi, self.max_iter, abs(Phi) < self.eps_Phi)


# =====================================================================
# 6. Траектория по точкам ТСН
# =====================================================================
class SimpleTrajectory:
    """Траектория раскладчика, построенная по точкам ТСН через кубические сплайны."""
    def __init__(self, s_data, points):
        self.s_data = np.asarray(s_data)
        self.total_length = float(self.s_data[-1])
        self._sx = CubicSpline(self.s_data, points[:, 0], bc_type="natural")
        self._sy = CubicSpline(self.s_data, points[:, 1], bc_type="natural")
        self._sz = CubicSpline(self.s_data, points[:, 2], bc_type="natural")

    def R(self, s):
        s = np.clip(float(s), 0.0, self.total_length)
        return np.array([self._sx(s), self._sy(s), self._sz(s)])

    def R_deriv(self, s):
        s = np.clip(float(s), 0.0, self.total_length)
        return np.array([self._sx(s, 1), self._sy(s, 1), self._sz(s, 1)])


# =====================================================================
# 7. ОСНОВНОЙ БЛОК
# =====================================================================
def main():
    # --- Геометрия ---------------------------------------------------
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 60)
    print("ЧАСТЬ 2: Обратная задача DAE по ТСН из CSV")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")

    # --- Загрузка ТСН из CSV ----------------------------------------
    csv_path = Path("part1_tsn_shadow.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите Part 1.")
        return

    df = pd.read_csv(csv_path)
    # Берём только валидные точки
    valid = df["valid"].values.astype(bool)
    s_vals = df["s"].values[valid]
    # tsn_pts = df[["X", "Y", "Z"]].values[valid]
    # --- Перепараметризация ТСН по длине дуги -------------------------
    tsn_pts = df[["X", "Y", "Z"]].values[valid]
    # Вычисляем кумулятивную длину дуги ТСН
    deltas = np.diff(tsn_pts, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    s_tsn = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    # Теперь s_tsn — это истинная длина дуги ТСН
    # traj = SimpleTrajectory(s_tsn, tsn_pts)
    traj=Trajectory.from_points(tsn_pts,method='nurbs',degree=5)
    # print(f"\nЗагружено {len(s_vals)} валидных точек ТСН из {csv_path}")

    
    # --- Траектория раскладчика --------------------------------------
    # traj = SimpleTrajectory(s_vals, tsn_pts)
    # traj=Trajectory.from_points(tsn_pts)
    print(f"Длина траектории ТСН: {traj.total_length:.3f}")

    # --- Эталон: прямая задача на E2 ---------------------------------
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = float(s_vals[-1])
    num_points = len(s_vals)
    print(f"\nПостроение эталонной ЛУ (геодезическая на E2)...")
    _, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Эталон построен: {len(lu_ref)} точек")

    # --- Обратная задача DAE -----------------------------------------
    print(f"\nРешение обратной задачи DAE...")
    predictor = DAEPredictor(max_speed=1.0)
    # corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7, v_periodic=True)
    corrector = NewtonCorrector(
    eps_Phi=1e-10, max_iter=7,
    v_periodic=False,  # v ∈ [-π/2, π/2], не периодична
    # u_periodic не реализован в текущем корректоре, 
    # но для данной геодезики u не выйдет за 2π
)

    # Корректировка начальной точки на многообразие Φ=0
    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr
    print(f"Начальная точка скорректирована: Φ₀={corr0.Phi:.2e}, итераций={corr0.iterations}")

    z_eval = s_vals  # используем ту же сетку
    N = len(z_eval)
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    optical_flags = np.zeros(N, dtype=bool)
    u_hist[0], v_hist[0] = u_cur, v_cur

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        dz = z_next - z_k

        # Предиктор
        geom = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
        pred = predictor.predict_step(geom, traj, u_cur, v_cur, z_k, dz)
        optical_flags[i + 1] = pred.optical_fallback

        # Корректор
        corr = corrector.correct(E2, traj, pred.u_pred, pred.v_pred, z_next)
        u_cur, v_cur = corr.u_corr, corr.v_corr
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = corr.Phi
        newton_iters_hist[i + 1] = corr.iterations

        # Нормальная кривизна (для диагностики)
        geom_f = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
        R_f = traj.R(z_next)
        V_thread = R_f - geom_f.r
        lam = np.linalg.norm(V_thread)
        if lam > 1e-12:
            tau = V_thread / lam
            P = geom_f.project_on_basis(tau)
            tau_contra = geom_f.G_inv @ P
            kappa_n_hist[i + 1] = geom_f.normal_curvature(tau_contra)

    # 3D-точки восстановленной ЛУ
    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])

    # --- Метрики -----------------------------------------------------
    print(f"\n--- Результаты обратной задачи ---")
    print(f"Max |Φ|        = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Φ|       = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Φ в конце      = {Phi_hist[-1]:.2e}")
    print(f"Max κ_n        = {np.max(np.abs(kappa_n_hist)):.4f}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Optical fallback шагов   = {np.sum(optical_flags)}/{N}")

    # Отклонение от эталона
    diff = np.linalg.norm(lu_inv - lu_ref, axis=1)
    print(f"Max ||Δr||     = {np.max(diff):.4f}")
    print(f"Mean ||Δr||    = {np.mean(diff):.4f}")

    # --- Сохранение CSV ----------------------------------------------
    df_out = pd.DataFrame({
        "z": z_eval,
        "u": u_hist, "v": v_hist,
        "X": lu_inv[:, 0], "Y": lu_inv[:, 1], "Z": lu_inv[:, 2],
        "Phi": Phi_hist,
        "kappa_n": kappa_n_hist,
        "newton_iters": newton_iters_hist,
        "optical_flag": optical_flags,
    })
    df_out.to_csv("part2_inverse_dae.csv", index=False)
    print(f"\nCSV сохранён: part2_inverse_dae.csv")

    # --- 3D Визуализация ---------------------------------------------
    print("\nПостроение 3D-сцены...")
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

    # ТСН
    fig.add_trace(go.Scatter3d(x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
                               mode="lines", line=dict(color="black", width=4),
                               name="ТСН (загружена из CSV)"))

    # Эталонная ЛУ
    fig.add_trace(go.Scatter3d(x=lu_ref[:, 0], y=lu_ref[:, 1], z=lu_ref[:, 2],
                               mode="lines", line=dict(color="blue", width=3, dash="dash"),
                               name="ЛУ эталон (геодезическая)"))

    # Восстановленная ЛУ
    fig.add_trace(go.Scatter3d(x=lu_inv[:, 0], y=lu_inv[:, 1], z=lu_inv[:, 2],
                               mode="lines", line=dict(color="green", width=3),
                               name="ЛУ DAE (восстановленная)"))

    # Лучи (каждый 10-й)
    for i in range(0, N, 10):
        fig.add_trace(go.Scatter3d(
            x=[lu_inv[i, 0], tsn_pts[i, 0]],
            y=[lu_inv[i, 1], tsn_pts[i, 1]],
            z=[lu_inv[i, 2], tsn_pts[i, 2]],
            mode="lines", line=dict(color="gray", width=1),
            showlegend=False
        ))

    fig.update_layout(
        title="Часть 2: Обратная задача DAE — восстановление ЛУ на E2",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_inverse_dae_3d.html")
    print("3D-сцена сохранена: part2_inverse_dae_3d.html")

    # --- Графики диагностики -----------------------------------------
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(z_eval, np.abs(Phi_hist) + 1e-16, "g-", lw=1.5, label="|Φ|")
    ax.axhline(1e-10, color="k", ls="--", lw=0.5)
    ax.set_title("A. Невязка связи |Φ(z)|")
    ax.set_xlabel("z"); ax.set_ylabel("|Φ|")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(z_eval, diff * 1000, "r-", lw=1.5)
    ax.set_title("B. Отклонение ||r_DAE − r_эталон||, мкм")
    ax.set_xlabel("z"); ax.set_ylabel("||Δr||, мкм")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(z_eval[1:], kappa_n_hist[1:], "b-", lw=1.5)
    ax.set_title("C. Нормальная кривизна κ_n(z)")
    ax.set_xlabel("z"); ax.set_ylabel("κ_n")
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

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()