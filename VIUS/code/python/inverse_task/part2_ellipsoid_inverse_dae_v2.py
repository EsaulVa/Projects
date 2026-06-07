#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part2_ellipsoid_inverse_dae_v2.py
==================================
Часть 2 (исправленная):
1. Загружает ТСН из CSV (part1_tsn_shadow.csv).
2. Строит траекторию раскладчика R(z) по точкам ТСН с перепараметризацией
   по длине дуги (хордовая параметризация + интегрирование).
3. Решает обратную задачу намотки на внутренний эллипсоид E2
   методом DAE-предиктор–корректор (отчёт 5.06.2026).
4. Верификация: от восстановленной ЛУ трассирует ТСН на E1 и сравнивает
   с исходной ТСН (а не сравнивает ЛУ напрямую из-за разной параметризации).
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
# 2. ПРЯМАЯ ЗАДАЧА: геодезическая (для эталона, опционально)
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

# from dae_helper.dae_predictor_v1 import DAEPredictor
# from dae_helper.newton_corrector import NewtonCorrector
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


class DAEPredictor:
    """
    Предиктор DAE-схемы (Метод Хойна / RK2).
    """
    def __init__(self, max_speed=50.0):
        self.max_speed = max_speed

    def predict_step(self, surface, traj, u, v, z, dz):
        """
        Выполнить один шаг предиктора RK2 от (u, v) при параметре z на шаг dz.
        """
        z_next = z + dz
        
        # ==================================================================
        # ШАГ 1: Вычисление скорости в НАЧАЛЕ интервала (точка k)
        # ==================================================================
        geom_k = SurfaceGeometryPack.from_surface(surface, u, v)
        R_k = traj.R(z)
        V_thread_k = R_k - geom_k.r
        R_prime_k = traj.R_deriv(z)
        u_prime_k = geom_k.winding_velocity(R_prime_k, V_thread_k)
        
        # Сохраняем диагностику начала шага для отчета
        grad_u_k = geom_k.grad_Phi(V_thread_k)
        grad_s_k = geom_k.surface_gradient(grad_u_k)
        Ng_k = geom_k.norm_grad_sq(grad_u_k)
        dPhi_dz_k = geom_k.dPhi_dz(R_prime_k)
        Rp_parallel_k = geom_k.base_velocity(R_prime_k)
        optical_fallback = Ng_k < 1e-14
        mu_k = 0.0 if optical_fallback else -(dPhi_dz_k + float(grad_u_k @ Rp_parallel_k)) / Ng_k

        # ==================================================================
        # ШАГ 2: Предиктор Эйлера (промежуточная точка tilde)
        # ==================================================================
        u_tilde_arr = np.array([u, v]) + u_prime_k * dz
        u_tilde, v_tilde = float(u_tilde_arr[0]), float(u_tilde_arr[1])

        # ==================================================================
        # ШАГ 3: Вычисление скорости в КОНЦЕ интервала (точка tilde)
        # ==================================================================
        try:
            geom_tilde = SurfaceGeometryPack.from_surface(surface, u_tilde, v_tilde)
            R_next = traj.R(z_next)
            V_thread_tilde = R_next - geom_tilde.r
            R_prime_next = traj.R_deriv(z_next)
            u_prime_tilde = geom_tilde.winding_velocity(R_prime_next, V_thread_tilde)
        except ValueError:
            # Если промежуточная точка вылетела за пределы параметризации
            # (маловероятно на эллипсоиде, но для безопасности), падаем обратно на Эйлера
            u_prime_tilde = u_prime_k

        # ==================================================================
        # ШАГ 4: Усреднение скоростей и итоговый шаг (Коррекция Хойна)
        # ==================================================================
        u_prime_avg = 0.5 * (u_prime_k + u_prime_tilde)
        
        u_pred_arr = np.array([u, v]) + u_prime_avg * dz
        u_pred, v_pred = float(u_pred_arr[0]), float(u_pred_arr[1])

        # --- Ограничение скорости (защита от катастрофических шагов) ---
        speed = geom_k.metric_speed(u_prime_avg)
        du_dz, dv_dz = float(u_prime_avg[0]), float(u_prime_avg[1])
        
        if self.max_speed is not None and speed > self.max_speed:
            scale = self.max_speed / speed
            du_dz *= scale
            dv_dz *= scale
            u_prime_avg = np.array([du_dz, dv_dz])
            speed = self.max_speed
            # Пересчитываем итоговую точку с ограниченной скоростью
            u_pred_arr = np.array([u, v]) + u_prime_avg * dz
            u_pred, v_pred = float(u_pred_arr[0]), float(u_pred_arr[1])

        return PredictorResult(
            u_pred=u_pred, v_pred=v_pred,
            du_dz=du_dz, dv_dz=dv_dz,
            R_prime=R_prime_k, V_thread=V_thread_k,
            mu=mu_k, grad_u=grad_u_k, grad_s=grad_s_k,
            Ng=Ng_k, dPhi_dz=dPhi_dz_k,
            Rp_parallel=Rp_parallel_k, speed=speed,
            optical_fallback=optical_fallback
        )


# =====================================================================
# 5. NewtonCorrector (v_periodic убран)
# =====================================================================
@dataclass(frozen=True)
class CorrectorResult:
    u_corr: float; v_corr: float
    Phi: float; iterations: int; converged: bool


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
# 6. TrajectoryByArcLength — перепараметризация по длине дуги
# =====================================================================
class TrajectoryByArcLength:
    """
    Траектория, параметризованная по длине дуги.
    Аналог Trajectory.from_points из проекта.
    """
    def __init__(self, points):
        points = np.asarray(points)
        self._build(points)

    def _build(self, points):
        # Хордовая параметризация u ∈ [0, 1]
        n = len(points)
        chord_lengths = np.zeros(n)
        for i in range(1, n):
            chord_lengths[i] = chord_lengths[i-1] + np.linalg.norm(points[i] - points[i-1])
        total_chord = chord_lengths[-1]
        if total_chord < 1e-12:
            raise ValueError("Слишком короткая траектория")
        u_param = chord_lengths / total_chord

        # Кубические сплайны по u
        self._sx = CubicSpline(u_param, points[:, 0], bc_type="natural")
        self._sy = CubicSpline(u_param, points[:, 1], bc_type="natural")
        self._sz = CubicSpline(u_param, points[:, 2], bc_type="natural")
        self._u_param = u_param

        # Интегрирование длины дуги: s(u) = ∫₀^u |r'(τ)| dτ
        u_fine = np.linspace(0, 1, 5000)
        dx = self._sx(u_fine, 1)
        dy = self._sy(u_fine, 1)
        dz = self._sz(u_fine, 1)
        speed = np.sqrt(dx**2 + dy**2 + dz**2)
        s_fine = cumtrapz(speed, u_fine, initial=0)
        self.total_length = float(s_fine[-1])

        # Отображение s -> u (обратное)
        self._u_of_s = interp1d(s_fine, u_fine, kind="cubic",
                                 bounds_error=False, fill_value=(0.0, 1.0))

    def R(self, s):
        s = np.clip(float(s), 0.0, self.total_length)
        u = float(self._u_of_s(s))
        return np.array([self._sx(u), self._sy(u), self._sz(u)])

    def R_deriv(self, s):
        """Единичный вектор касательной dr/ds."""
        s = np.clip(float(s), 0.0, self.total_length)
        u = float(self._u_of_s(s))
        dx = self._sx(u, 1)
        dy = self._sy(u, 1)
        dz = self._sz(u, 1)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0])
        return np.array([dx, dy, dz]) / norm


# =====================================================================
# 7. Трассировка луча к эллипсоиду (для верификации)
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


def verify_by_tracing(lu_points, E2, E1, num_check=300):
    """
    От точек ЛУ строит траекторию, проецирует касательную на касательную
    плоскость E2 и трассирует луч к E1. Возвращает восстановленные точки ТСН.
    """
    traj_lu = TrajectoryByArcLength(lu_points)
    s_check = np.linspace(0, traj_lu.total_length, num_check)
    tsn_recon = np.zeros((num_check, 3))
    for i, s in enumerate(s_check):
        r = traj_lu.R(s)
        tau = traj_lu.R_deriv(s)
        # Проекция на касательную плоскость E2 в точке r
        # Находим ближайшую точку на E2 (параметры u,v)
        # Для простоты: ищем по нормали (итерационно, 1 шаг Ньютона)
        # или используем сохранённые u,v. Здесь упрощённый подход:
        # считаем, что r лежит на E2, и нормаль известна приблизительно.
        # Более точно: используем u,v из обратной задачи.
        pass
    return tsn_recon

def bisection_step(surface, traj, predictor, corrector, u, v, z, dz, C_tol=1.5, min_dz=1e-6):
    """
    Адаптивный шаг DAE с бисекцией по скорости перемещения по поверхности.
    Возвращает (u_next, v_next, PredictorResult, CorrectorResult)
    """
    # 1. Выполняем базовый шаг предиктора
    pred = predictor.predict_step(surface, traj, u, v, z, dz)
    
    # 2. Выполняем корректор
    corr = corrector.correct(surface, traj, pred.u_pred, pred.v_pred, z + dz)
    if not corr.converged:
        # Если Ньютон не сошелся, шаг слишком большой - делим пополам
        if dz / 2 < min_dz:
            return corr.u_corr, corr.v_corr, pred, corr
        return bisection_step(surface, traj, predictor, corrector, u, v, z, dz / 2, C_tol, min_dz)
        
    u_next, v_next = corr.u_corr, corr.v_corr
    
    # 3. Вычисляем ожидаемую скорость (в начале шага)
    geom_k = SurfaceGeometryPack.from_surface(surface, u, v)
    u_prime_k = np.array([pred.du_dz, pred.dv_dz])
    expected_speed = geom_k.metric_speed(u_prime_k)  # ds/dz
    
    # 4. Вычисляем фактическое перемещение по поверхности (в конце шага)
    try:
        geom_next = SurfaceGeometryPack.from_surface(surface, u_next, v_next)
    except ValueError:
         if dz / 2 < min_dz: return u_next, v_next, pred, corr
         return bisection_step(surface, traj, predictor, corrector, u, v, z, dz / 2, C_tol, min_dz)
         
    delta_u = np.array([u_next - u, v_next - v])
    actual_ds = geom_next.metric_speed(delta_u / dz) if dz > 0 else 0.0 # Фактическая скорость
    
    # 5. Проверяем критерий скачка (отношение скоростей)
    if expected_speed > 1e-9:
        ratio = actual_ds / expected_speed
        # Если фактическая скорость в C_tol раз больше или меньше ожидаемой
        if ratio > C_tol or ratio < 1.0 / C_tol:
            if dz / 2 < min_dz:
                # Шаг уже минимальный, принимаем как есть
                return u_next, v_next, pred, corr
            # Рекурсивно делим шаг пополам
            return bisection_step(surface, traj, predictor, corrector, u, v, z, dz / 2, C_tol, min_dz)
            
    return u_next, v_next, pred, corr
# =====================================================================
# 8. ОСНОВНОЙ БЛОК
# =====================================================================
def main():
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 60)
    print("ЧАСТЬ 2 (v2): Обратная задача DAE по ТСН из CSV")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")

    # --- Загрузка ТСН из CSV ----------------------------------------
    csv_path = Path("part1_tsn_shadow.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите Part 1.")
        return

    df = pd.read_csv(csv_path)
    valid = df["valid"].values.astype(bool)
    tsn_pts = df[["X", "Y", "Z"]].values[valid]
    print(f"Загружено {len(tsn_pts)} валидных точек ТСН")

    # --- Траектория раскладчика (перепараметризация по дуге) ---------
    traj = TrajectoryByArcLength(tsn_pts)
    print(f"Длина траектории ТСН: {traj.total_length:.3f}")

    # --- Эталон: прямая задача на E2 ---------------------------------
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = traj.total_length
    num_points = len(tsn_pts)
    print(f"Построение эталонной ЛУ (геодезическая на E2)...")
    _, uv_ref, lu_ref = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Эталон построен: {len(lu_ref)} точек")

    # --- Обратная задача DAE -----------------------------------------
    print(f"Решение обратной задачи DAE...")
    predictor = DAEPredictor(max_speed=50.0)
    corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)

    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr
    print(f"Начальная точка скорректирована: Φ₀={corr0.Phi:.2e}, итераций={corr0.iterations}")

    z_eval = np.linspace(0, traj.total_length, num_points)
    N = len(z_eval)
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    optical_flags = np.zeros(N, dtype=bool)
    u_hist[0], v_hist[0] = u_cur, v_cur

        # --- Обратная задача DAE с адаптивным шагом ---
    print(f"Решение обратной задачи DAE (Адаптивный шаг)...")
    # predictor = DAEPredictor(max_speed=50.0)
    # corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)

    # Начальная коррекция
    corr0 = corrector.correct(E2, traj, u0, v0, 0.0)
    u_cur, v_cur = corr0.u_corr, corr0.v_corr

    N = 30000 # Количество точек вывода
    z_eval = np.linspace(0, traj.total_length, N)
    
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    u_hist[0], v_hist[0] = u_cur, v_cur

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        dz = z_next - z_k
        
        # Вызов адаптивного шага!
        u_cur, v_cur, pred, corr = bisection_step(
            E2, traj, predictor, corrector, u_cur, v_cur, z_k, dz, C_tol=1.5
        )
        
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur

    # for i in range(N - 1):
    #     z_k = z_eval[i]
    #     z_next = z_eval[i + 1]
    #     dz = z_next - z_k
    #     # geom = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
    #     # pred = predictor.predict_step(geom, traj, u_cur, v_cur, z_k, dz)
    #      # Передаем E2 вместо geom:
    #     pred = predictor.predict_step(E2, traj, u_cur, v_cur, z_k, dz)
    #     optical_flags[i + 1] = pred.optical_fallback
    #     corr = corrector.correct(E2, traj, pred.u_pred, pred.v_pred, z_next)
    #     u_cur, v_cur = corr.u_corr, corr.v_corr
    #     u_hist[i + 1] = u_cur
    #     v_hist[i + 1] = v_cur
    #     Phi_hist[i + 1] = corr.Phi
    #     newton_iters_hist[i + 1] = corr.iterations
    #     # κ_n для диагностики
    #     geom_f = SurfaceGeometryPack.from_surface(E2, u_cur, v_cur)
    #     R_f = traj.R(z_next)
    #     V_thread = R_f - geom_f.r
    #     lam = np.linalg.norm(V_thread)
    #     if lam > 1e-12:
    #         tau = V_thread / lam
    #         P = geom_f.project_on_basis(tau)
    #         tau_contra = geom_f.G_inv @ P
    #         kappa_n_hist[i + 1] = geom_f.normal_curvature(tau_contra)

    lu_inv = np.array([E2.position(u_hist[k], v_hist[k]) for k in range(N)])

    # --- Верификация: трассировка от восстановленной ЛУ к ТСН ---------
    print(f"\nВерификация: трассировка ТСН от восстановленной ЛУ...")
    tsn_recon = np.zeros((N, 3))
    valid_recon = np.zeros(N, dtype=bool)
    for i, z in enumerate(z_eval):
        r = lu_inv[i]
        R_target = traj.R(z)  # исходная точка ТСН
        V_thread = R_target - r
        lam = np.linalg.norm(V_thread)
        if lam < 1e-12:
            continue
        tau = V_thread / lam  # единичный вектор нити
        # tau уже лежит в касательной плоскости (Φ ≈ 0), но для надёжности проецируем
        m = E2.normal(u_hist[i], v_hist[i])
        tau_proj = tau - np.dot(tau, m) * m
        tau_norm = np.linalg.norm(tau_proj)
        if tau_norm > 1e-12:
            tau_proj = tau_proj / tau_norm
        t, R_pt = trace_ray_to_ellipsoid(E1, r, tau_proj)
        if t is not None:
            tsn_recon[i] = R_pt
            valid_recon[i] = True

        # --- Метрики -----------------------------------------------------
    print(f"\n--- Результаты обратной задачи ---")
    print(f"Max |Φ|        = {np.max(np.abs(Phi_hist)):.2e}")
    print(f"Mean |Φ|       = {np.mean(np.abs(Phi_hist)):.2e}")
    print(f"Φ в конце      = {Phi_hist[-1]:.2e}")
    print(f"Max κ_n        = {np.max(np.abs(kappa_n_hist)):.4f}")
    print(f"Среднее итераций Ньютона = {np.mean(newton_iters_hist[1:]):.2f}")
    print(f"Optical fallback шагов   = {np.sum(optical_flags)}/{N}")

    # Отклонение восстановленной ТСН от исходной (по вектору нити)
    if np.sum(valid_recon) > 0:
        # diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - tsn_pts[valid_recon], axis=1)
        # Вместо diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - tsn_pts[valid_recon], axis=1)
        # Сравниваем с непрерывной траекторией:
        # R_etalon = traj.R(z_eval[valid_recon])
        R_etalon = np.array([traj.R(z) for z in z_eval[valid_recon]])
        diff_tsn = np.linalg.norm(tsn_recon[valid_recon] - R_etalon, axis=1)
        print(f"\n--- Верификация (ТСН по вектору нити) ---")
        print(f"Валидных точек: {np.sum(valid_recon)}/{N}")
        print(f"Max ||ΔТСН||   = {np.max(diff_tsn):.6f}")
        print(f"Mean ||ΔТСН||  = {np.mean(diff_tsn):.6f}")

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
    df_out.to_csv("part2_inverse_dae_v2.csv", index=False)
    print(f"CSV сохранён: part2_inverse_dae_v2.csv")

    

    # --- 3D Визуализация ---------------------------------------------
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
        fig.add_trace(go.Scatter3d(x=tsn_recon[valid_recon, 0],
                                   y=tsn_recon[valid_recon, 1],
                                   z=tsn_recon[valid_recon, 2],
                                   mode="lines", line=dict(color="magenta", width=3, dash="dot"),
                                   name="ТСН reconstructed"))
    for i in range(0, N, 10):
        fig.add_trace(go.Scatter3d(
            x=[lu_inv[i, 0], tsn_pts[i, 0]],
            y=[lu_inv[i, 1], tsn_pts[i, 1]],
            z=[lu_inv[i, 2], tsn_pts[i, 2]],
            mode="lines", line=dict(color="gray", width=1), showlegend=False
        ))
    fig.update_layout(
        title="Часть 2 (v2): Обратная задача DAE — восстановление ЛУ на E2",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        width=1200, height=900
    )
    fig.write_html("part2_inverse_dae_v2_3d.html")
    print("3D-сцена сохранена: part2_inverse_dae_v2_3d.html")

    # --- Графики диагностики -----------------------------------------
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.semilogy(z_eval, np.abs(Phi_hist) + 1e-16, "g-", lw=1.5, label="|Φ|")
    ax.axhline(1e-10, color="k", ls="--", lw=0.5)
    ax.set_title("A. Невязка связи |Φ(z)|")
    ax.set_xlabel("z"); ax.set_ylabel("|Φ|")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if np.sum(valid_recon) > 0:
        diff_tsn_full = np.linalg.norm(tsn_recon - tsn_pts, axis=1)
        ax.plot(z_eval[valid_recon], diff_tsn_full[valid_recon] * 1000, "r-", lw=1.5)
    ax.set_title("B. Отклонение ||ТСН_recon − ТСН_orig||, мкм")
    ax.set_xlabel("z"); ax.set_ylabel("||ΔТСН||, мкм")
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
    plt.savefig("part2_diagnostics_v2.png", dpi=150)
    print("Графики сохранены: part2_diagnostics_v2.png")
    plt.show()

    print("" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
