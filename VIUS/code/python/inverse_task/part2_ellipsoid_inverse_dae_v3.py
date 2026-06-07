#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part2_ellipsoid_inverse_dae_v3.py
==================================
Часть 2 (v3): Обратная задача DAE с адаптивной бисекцией.

1. Загружает ТСН из CSV (part1_tsn_shadow.csv).
2. Строит траекторию раскладчика R(z) по точкам ТСН с перепараметризацией
   по длине дуги.
3. Решает обратную задачу намотки на внутренний эллипсоид E2
   методом DAE-предиктор–корректор с адаптивной бисекцией шага.
4. Верификация: от восстановленной ЛУ трассирует ТСН на E1 и сравнивает
   с исходной ТСН в единой параметризации.
5. Сохраняет результаты и строит визуализацию.

Параметры:
  E1 (внешний): a=3.0, b=2.5, c=2.0
  E2 (внутренний): scale=0.8
  Начальные условия: u0=π/3, v0=π/6
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import solve_ivp
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
import plotly.graph_objects as go
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =====================================================================
# 1. КЛАСС ЭЛЛИПСОИДА (исправленный second_fundamental_form)
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
        """Исправленные аналитические формулы для эллипсоида."""
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        denom = np.sqrt(
            (cos_u * cos_v / self.a)**2 +
            (sin_u * cos_v / self.b)**2 +
            (sin_v / self.c)**2
        )
        L = -cos_v**2 / denom
        M = 0.0
        N = -1.0 / denom
        return L, M, N

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
# 4. DAEPredictor (с compute_velocity для AdaptiveStepper)
# =====================================================================
@dataclass(frozen=True)
class PredictorResult:
    u_pred: float
    v_pred: float
    du_dz: float
    dv_dz: float
    R_prime: np.ndarray
    V_thread: np.ndarray
    mu: float
    grad_u: np.ndarray
    grad_s: np.ndarray
    Ng: float
    dPhi_dz: float
    Rp_parallel: np.ndarray
    speed: float
    optical_fallback: bool


class DAEPredictor:
    def __init__(self, solver_method: Optional[str] = None,
                 max_speed: Optional[float] = 50.0,
                 rtol: float = 1e-6, atol: float = 1e-8):
        self.solver_method = solver_method
        self.max_speed = max_speed
        self.rtol = rtol
        self.atol = atol

    def predict_step(self, surface, traj, u, v, z, dz):
        if self.solver_method is None:
            return self._euler_step(surface, traj, u, v, z, dz)
        else:
            return self._scipy_step(surface, traj, u, v, z, dz)

    def compute_velocity(self, surface, traj, u, v, z):
        """Вычислить скорость u' и ее длину s_dot в точке (u,v,z) без шага."""
        geom = SurfaceGeometryPack.from_surface(surface, u, v)
        R = traj.R(z)
        V_thread = R - geom.r
        R_prime = traj.R_deriv(z)
        grad_u = geom.grad_Phi(V_thread)
        grad_s = geom.surface_gradient(grad_u)
        Ng = geom.norm_grad_sq(grad_u)
        dphi_dz = geom.dPhi_dz(R_prime)
        Rp_parallel = geom.base_velocity(R_prime)
        optical_fallback = Ng < 1e-14
        if optical_fallback:
            mu = 0.0
        else:
            residual = dphi_dz + float(grad_u @ Rp_parallel)
            mu = -residual / Ng
        u_prime = Rp_parallel + mu * grad_s
        du_dz, dv_dz = float(u_prime[0]), float(u_prime[1])
        speed = geom.metric_speed(u_prime)
        if self.max_speed is not None and speed > self.max_speed:
            scale = self.max_speed / speed
            du_dz *= scale; dv_dz *= scale
            u_prime = np.array([du_dz, dv_dz])
            speed = self.max_speed
        return du_dz, dv_dz, speed, optical_fallback

    def _euler_step(self, surface, traj, u, v, z, dz):
        geom = SurfaceGeometryPack.from_surface(surface, u, v)
        R = traj.R(z)
        V_thread = R - geom.r
        R_prime = traj.R_deriv(z)
        grad_u = geom.grad_Phi(V_thread)
        grad_s = geom.surface_gradient(grad_u)
        Ng = geom.norm_grad_sq(grad_u)
        dphi_dz = geom.dPhi_dz(R_prime)
        Rp_parallel = geom.base_velocity(R_prime)
        optical_fallback = Ng < 1e-14
        mu = 0.0 if optical_fallback else -(dphi_dz + float(grad_u @ Rp_parallel)) / Ng
        u_prime = Rp_parallel + mu * grad_s
        du_dz, dv_dz = float(u_prime[0]), float(u_prime[1])
        speed = geom.metric_speed(u_prime)
        if self.max_speed is not None and speed > self.max_speed:
            scale = self.max_speed / speed
            du_dz *= scale; dv_dz *= scale
            u_prime = np.array([du_dz, dv_dz])
            speed = self.max_speed
        return PredictorResult(
            u_pred=u + du_dz * dz, v_pred=v + dv_dz * dz,
            du_dz=du_dz, dv_dz=dv_dz, R_prime=R_prime, V_thread=V_thread,
            mu=mu, grad_u=grad_u, grad_s=grad_s, Ng=Ng, dPhi_dz=dphi_dz,
            Rp_parallel=Rp_parallel, speed=speed, optical_fallback=optical_fallback
        )

    def _scipy_step(self, surface, traj, u, v, z, dz):
        z_next = z + dz

        def rhs(z_local, y):
            u_loc, v_loc = y
            z_loc = np.clip(float(z_local), z, z_next)
            geom = SurfaceGeometryPack.from_surface(surface, u_loc, v_loc)
            R_loc = traj.R(z_loc)
            V_thread = R_loc - geom.r
            R_prime = traj.R_deriv(z_loc)
            return geom.winding_velocity(R_prime, V_thread)

        sol = solve_ivp(
            rhs, [z, z_next], [float(u), float(v)],
            method=self.solver_method, t_eval=[z_next],
            rtol=self.rtol, atol=self.atol, max_step=dz,
        )
        if not sol.success:
            return self._euler_step(surface, traj, u, v, z, dz)
        u_pred, v_pred = sol.y[:, -1]
        geom_f = SurfaceGeometryPack.from_surface(surface, u_pred, v_pred)
        R_f = traj.R(z_next)
        V_thread_f = R_f - geom_f.r
        R_prime_f = traj.R_deriv(z_next)
        grad_u = geom_f.grad_Phi(V_thread_f)
        grad_s = geom_f.surface_gradient(grad_u)
        Ng = geom_f.norm_grad_sq(grad_u)
        dphi_dz = geom_f.dPhi_dz(R_prime_f)
        Rp_parallel = geom_f.base_velocity(R_prime_f)
        optical_fallback = Ng < 1e-14
        mu = 0.0 if optical_fallback else -(dphi_dz + float(grad_u @ Rp_parallel)) / Ng
        du_dz = (u_pred - u) / dz
        dv_dz = (v_pred - v) / dz
        speed = geom_f.metric_speed(np.array([du_dz, dv_dz]))
        return PredictorResult(
            u_pred=float(u_pred), v_pred=float(v_pred),
            du_dz=du_dz, dv_dz=dv_dz, R_prime=R_prime_f, V_thread=V_thread_f,
            mu=mu, grad_u=grad_u, grad_s=grad_s, Ng=Ng, dPhi_dz=dphi_dz,
            Rp_parallel=Rp_parallel, speed=speed, optical_fallback=optical_fallback
        )


# =====================================================================
# 5. NewtonCorrector
# =====================================================================
@dataclass(frozen=True)
class CorrectorResult:
    u_corr: float
    v_corr: float
    Phi: float
    iterations: int
    converged: bool


class NewtonCorrector:
    def __init__(self, eps_Phi=1e-10, max_iter=7):
        self.eps_Phi = eps_Phi
        self.max_iter = max_iter

    def correct(self, surface, traj, u_pred, v_pred, z_target):
        u_c, v_c = float(u_pred), float(v_pred)
        for nit in range(self.max_iter):
            try:
                geom = SurfaceGeometryPack.from_surface(surface, u_c, v_c)
            except ValueError:
                return CorrectorResult(u_c, v_c, np.nan, nit, False)
            R_t = traj.R(z_target)
            Phi = SurfaceGeometryPack.compute_Phi(R_t, geom.r, geom.normal)
            if abs(Phi) < self.eps_Phi:
                return CorrectorResult(u_c, v_c, Phi, nit, True)
            V_thread = R_t - geom.r
            grad_u = geom.grad_Phi(V_thread)
            grad_s = geom.surface_gradient(grad_u)
            Ng = geom.norm_grad_sq(grad_u)
            if Ng < 1e-14:
                return CorrectorResult(u_c, v_c, Phi, nit, False)
            delta = -Phi / Ng * grad_s
            u_c += float(delta[0])
            v_c += float(delta[1])
        try:
            geom = SurfaceGeometryPack.from_surface(surface, u_c, v_c)
        except ValueError:
            return CorrectorResult(u_c, v_c, np.nan, self.max_iter, False)
        Phi = SurfaceGeometryPack.compute_Phi(traj.R(z_target), geom.r, geom.normal)
        return CorrectorResult(u_c, v_c, Phi, self.max_iter, abs(Phi) < self.eps_Phi)


# =====================================================================
# 6. TrajectoryByArcLength
# =====================================================================
class TrajectoryByArcLength:
    def __init__(self, points):
        points = np.asarray(points)
        self._build(points)

    def _build(self, points):
        n = len(points)
        chord_lengths = np.zeros(n)
        for i in range(1, n):
            chord_lengths[i] = chord_lengths[i-1] + np.linalg.norm(points[i] - points[i-1])
        total_chord = chord_lengths[-1]
        if total_chord < 1e-12:
            raise ValueError("Слишком короткая траектория")
        u_param = chord_lengths / total_chord
        self._sx = CubicSpline(u_param, points[:, 0], bc_type="natural")
        self._sy = CubicSpline(u_param, points[:, 1], bc_type="natural")
        self._sz = CubicSpline(u_param, points[:, 2], bc_type="natural")
        self._u_param = u_param
        u_fine = np.linspace(0, 1, 5000)
        dx = self._sx(u_fine, 1)
        dy = self._sy(u_fine, 1)
        dz = self._sz(u_fine, 1)
        speed = np.sqrt(dx**2 + dy**2 + dz**2)
        s_fine = cumtrapz(speed, u_fine, initial=0)
        self.total_length = float(s_fine[-1])
        self._u_of_s = interp1d(s_fine, u_fine, kind="cubic",
                                 bounds_error=False, fill_value=(0.0, 1.0))

    def R(self, s):
        s = np.clip(float(s), 0.0, self.total_length)
        u = float(self._u_of_s(s))
        return np.array([self._sx(u), self._sy(u), self._sz(u)])

    def R_deriv(self, s):
        s = np.clip(float(s), 0.0, self.total_length)
        u = float(self._u_of_s(s))
        dx = self._sx(u, 1)
        dy = self._sy(u, 1)
        dz = self._sz(u, 1)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return np.array([dx, dy, dz]) / norm

    def R_array(self, s_array):
        return np.array([self.R(s) for s in s_array])



# =====================================================================
# 7. AdaptiveStepper (адаптивная бисекция шага)
# =====================================================================
@dataclass(frozen=True)
class StepResult:
    u_next: float
    v_next: float
    Phi: float
    iterations: int
    bisected: bool
    substeps: int
    ratio: float
    speed_expected: float
    speed_actual: float
    converged: bool


class AdaptiveStepper:
    def __init__(self, predictor, corrector, C_tol=3.0, max_bisect=4):
        self.predictor = predictor
        self.corrector = corrector
        self.C_tol = C_tol
        self.max_bisect = max_bisect

    def step(self, surface, traj, u, v, z, dz):
        # Ожидаемая скорость в начале шага
        _, _, speed_expected, _ = self.predictor.compute_velocity(
            surface, traj, u, v, z
        )

        n_sub = 1
        best_u, best_v = u, v
        best_Phi = 1.0
        best_nit = 0
        best_ratio = 1.0
        bisected = False
        converged = False

        for bisect_level in range(self.max_bisect + 1):
            sub_z = np.linspace(z, z + dz, n_sub + 1)
            u_s, v_s = u, v
            total_nit = 0
            jump_detected = False
            last_corr = None
            last_ratio = 1.0
            last_speed_actual = 0.0

            for j in range(n_sub):
                z_a = float(sub_z[j])
                z_b = float(sub_z[j + 1])
                dz_sub = z_b - z_a

                # Предиктор
                # geom = SurfaceGeometryPack.from_surface(surface, u_s, v_s)
                # pred = self.predictor.predict_step(geom, traj, u_s, v_s, z_a, dz_sub)
                # ДОЛЖНО БЫТЬ:
                pred = self.predictor.predict_step(surface, traj, u_s, v_s, z_a, dz_sub)

                # Корректор
                corr = self.corrector.correct(
                    surface, traj, pred.u_pred, pred.v_pred, z_b
                )
                total_nit += corr.iterations
                last_corr = corr

                if not corr.converged:
                    jump_detected = True
                    break

                # Фактическое перемещение
                du_j = corr.u_corr - u_s
                dv_j = corr.v_corr - v_s
                try:
                    geom_new = SurfaceGeometryPack.from_surface(
                        surface, corr.u_corr, corr.v_corr
                    )
                except ValueError:
                    jump_detected = True
                    break

                ds_actual = np.sqrt(max(
                    geom_new.E * du_j**2
                    + 2.0 * geom_new.F * du_j * dv_j
                    + geom_new.G * dv_j**2,
                    0.0
                ))

                # Ожидаемая скорость в начале подшага
                _, _, speed_sub, _ = self.predictor.compute_velocity(
                    surface, traj, u_s, v_s, z_a
                )
                ds_expect = speed_sub * abs(dz_sub)
                ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0
                last_ratio = ratio
                last_speed_actual = ds_actual / abs(dz_sub) if dz_sub != 0 else 0.0

                # Критерий скачка
                if (ratio > self.C_tol or ratio < 1.0 / self.C_tol) \
                        and bisect_level < self.max_bisect:
                    jump_detected = True
                    break

                u_s, v_s = corr.u_corr, corr.v_corr

            if not jump_detected:
                best_u, best_v = u_s, v_s
                best_Phi = last_corr.Phi if last_corr is not None else np.nan
                best_nit = total_nit
                best_ratio = last_ratio
                converged = True
                if bisect_level > 0:
                    bisected = True
                break
            else:
                n_sub *= 2

        return StepResult(
            u_next=best_u,
            v_next=best_v,
            Phi=best_Phi,
            iterations=best_nit,
            bisected=bisected,
            substeps=n_sub,
            ratio=best_ratio,
            speed_expected=speed_expected,
            speed_actual=last_speed_actual,
            converged=converged,
        )


# =====================================================================
# 8. Трассировка луча к эллипсоиду (для верификации)
# =====================================================================
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


# =====================================================================
# 9. ОСНОВНОЙ БЛОК
# =====================================================================
def main():
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 60)
    print("ЧАСТЬ 2 (v3): Обратная задача DAE с адаптивной бисекцией")
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
    print(f"\nЗагружено {len(tsn_pts)} валидных точек ТСН")

    # Траектория раскладчика
    traj = TrajectoryByArcLength(tsn_pts)
    print(f"Длина траектории ТСН: {traj.total_length:.3f}")

    # Эталон: прямая задача на E2
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = traj.total_length
    num_points = len(tsn_pts)
    print(f"\nПостроение эталонной ЛУ (геодезическая на E2)...")
    _, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Эталон построен: {len(lu_ref)} точек")

    # Обратная задача DAE с адаптивной бисекцией
    print(f"\nРешение обратной задачи DAE (адаптивная бисекция)...")
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
    print(f"\nВерификация: трассировка ТСН от восстановленной ЛУ...")
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
    print(f"\n--- Результаты обратной задачи ---")
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
        print(f"\n--- Верификация (ТСН по вектору нити) ---")
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
    df_out.to_csv("part2_inverse_dae_v3.csv", index=False)
    print(f"\nCSV сохранен: part2_inverse_dae_v3.csv")

    # 3D Визуализация
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
        title="Часть 2 (v3): Обратная задача DAE с адаптивной бисекцией",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_inverse_dae_v3_3d.html")
    print("3D-сцена сохранена: part2_inverse_dae_v3_3d.html")

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
    plt.savefig("part2_diagnostics_v3.png", dpi=150)
    print("Графики сохранены: part2_diagnostics_v3.png")
    plt.show()

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()