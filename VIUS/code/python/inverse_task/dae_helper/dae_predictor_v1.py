#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dae_predictor.py
================
Предиктор для DAE-схемы обратной задачи намотки.

Поддерживает два режима:
  • None (по умолчанию) — явный метод Эйлера (1-й порядок).
  • 'RK45', 'DOP853', 'RK23', 'Radau', 'BDF', 'LSODA' — интегрирование
    одного шага через scipy.integrate.solve_ivp с указанным методом.

Все формулы соответствуют отчёту «Обратная задача намотки нити…», 5 июня 2026 г.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.integrate import solve_ivp

from dae_helper.surface_geometry_pack import SurfaceGeometryPack


# ----------------------------------------------------------------------
# Результат работы предиктора
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# DAEPredictor
# ----------------------------------------------------------------------
class DAEPredictor:
    """
    Предиктор DAE-схемы.

    Parameters
    ----------
    solver_method : str | None
        Если None — явный Эйлер (1-й порядок).
        Если 'RK45', 'DOP853', 'RK23', 'Radau', 'BDF', 'LSODA' —
        шаг интегрируется через solve_ivp с указанным методом.
    max_speed : float | None
        Ограничение скорости перемещения по поверхности
        (только для режима Эйлера; для solve_ivp используйте max_step).
    rtol, atol : float
        Параметры solve_ivp (игнорируются при solver_method=None).
    """

    def __init__(
        self,
        solver_method: Optional[str] = None,
        max_speed: Optional[float] = 50.0,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ):
        self.solver_method = solver_method
        self.max_speed = max_speed
        self.rtol = rtol
        self.atol = atol

    # ==================================================================
    # Публичный интерфейс
    # ==================================================================
    def predict_step(self, surface, traj, u, v, z, dz):
        """
        Выполнить один шаг предиктора.

        Parameters
        ----------
        surface
            Поверхность с интерфейсом SurfaceGeometryPack.
        traj
            Траектория раскладчика с методами R(z), R_deriv(z).
        u, v : float
            Текущие координаты на поверхности.
        z : float
            Текущий параметр траектории.
        dz : float
            Шаг интегрирования (z_next − z).

        Returns
        -------
        PredictorResult
        """
        if self.solver_method is None:
            return self._euler_step(surface, traj, u, v, z, dz)
        else:
            return self._scipy_step(surface, traj, u, v, z, dz)

    # ==================================================================
    # Режим 1: Эйлер (явный, 1-й порядок)
    # ==================================================================
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
            du_dz *= scale
            dv_dz *= scale
            u_prime = np.array([du_dz, dv_dz])
            speed = self.max_speed

        return PredictorResult(
            u_pred=u + du_dz * dz,
            v_pred=v + dv_dz * dz,
            du_dz=du_dz,
            dv_dz=dv_dz,
            R_prime=R_prime,
            V_thread=V_thread,
            mu=mu,
            grad_u=grad_u,
            grad_s=grad_s,
            Ng=Ng,
            dPhi_dz=dphi_dz,
            Rp_parallel=Rp_parallel,
            speed=speed,
            optical_fallback=optical_fallback,
        )

    # ==================================================================
    # Режим 2: SciPy solve_ivp (высокий порядок)
    # ==================================================================
    def _scipy_step(self, surface, traj, u, v, z, dz):
        z_next = z + dz

        def rhs(z_local, y):
            u_loc, v_loc = y
            # Защита от выхода за пределы шага
            z_loc = np.clip(float(z_local), z, z_next)
            geom = SurfaceGeometryPack.from_surface(surface, u_loc, v_loc)
            R_loc = traj.R(z_loc)
            V_thread = R_loc - geom.r
            R_prime = traj.R_deriv(z_loc)
            return geom.winding_velocity(R_prime, V_thread)

        sol = solve_ivp(
            rhs,
            [z, z_next],
            [float(u), float(v)],
            method=self.solver_method,
            t_eval=[z_next],
            rtol=self.rtol,
            atol=self.atol,
            max_step=dz,
        )

        if not sol.success:
            # Fallback на Эйлера при неудаче solve_ivp
            return self._euler_step(surface, traj, u, v, z, dz)

        u_pred, v_pred = sol.y[:, -1]

        # Диагностика в конечной точке (для единообразия PredictorResult)
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
            u_pred=float(u_pred),
            v_pred=float(v_pred),
            du_dz=du_dz,
            dv_dz=dv_dz,
            R_prime=R_prime_f,
            V_thread=V_thread_f,
            mu=mu,
            grad_u=grad_u,
            grad_s=grad_s,
            Ng=Ng,
            dPhi_dz=dphi_dz,
            Rp_parallel=Rp_parallel,
            speed=speed,
            optical_fallback=optical_fallback,
        )