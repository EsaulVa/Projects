#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_stepper.py
===================
Адаптивный шаг DAE-схемы с бисекцией.

Реализует алгоритм раздела 7.2 отчёта:
 • ожидаемая скорость ṡ_k через метрику в начале шага;
 • фактическое перемещение Δs_fact через метрику в новой точке;
 • контроль отношения r = Δs_fact / (ṡ_k · dz);
 • рекурсивная бисекция при скачке (r вне [1/C_tol, C_tol]).

Дополнительно: режим без корректора (use_corrector=False),
полезный для отладки и сравнения предикторов.
"""

import numpy as np
from dataclasses import dataclass

from .geometry import SurfaceGeometryPack
from .dae_predictor import DAEPredictor
from .newton_corrector import NewtonCorrector


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
    def __init__(
        self,
        predictor: DAEPredictor,
        corrector: NewtonCorrector | None = None,
        C_tol: float = 3.0,
        max_bisect: int = 4,
        use_corrector: bool = True,
    ):
        self.predictor = predictor
        self.corrector = corrector
        self.C_tol = C_tol
        self.max_bisect = max_bisect
        self.use_corrector = use_corrector

    @staticmethod
    def _compute_phi(surface, traj, u, v, z):
        """Скалярное произведение (R(z) - r(u,v), m(u,v))."""
        r = surface.position(u, v)
        m = surface.normal(u, v)
        R = traj.R(z)
        return float(np.dot(R - r, m))

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
            last_Phi = np.nan
            last_ratio = 1.0
            last_speed_actual = 0.0

            for j in range(n_sub):
                z_a = float(sub_z[j])
                z_b = float(sub_z[j + 1])
                dz_sub = z_b - z_a

                pred = self.predictor.predict_step(
                    surface, traj, u_s, v_s, z_a, dz_sub
                )

                # --- Корректор (опционально) ---
                if self.use_corrector and self.corrector is not None:
                    corr = self.corrector.correct(
                        surface, traj, pred.u_pred, pred.v_pred, z_b
                    )
                    total_nit += corr.iterations
                    u_j, v_j = corr.u_corr, corr.v_corr
                    last_Phi = corr.Phi

                    if not corr.converged:
                        jump_detected = True
                        break
                else:
                    u_j, v_j = pred.u_pred, pred.v_pred
                    last_Phi = self._compute_phi(surface, traj, u_j, v_j, z_b)
                    total_nit += 0

                # Фактическое перемещение
                du_j = u_j - u_s
                dv_j = v_j - v_s
                try:
                    geom_new = SurfaceGeometryPack.from_surface(
                        surface, u_j, v_j
                    )
                except ValueError:
                    jump_detected = True
                    break

                ds_actual = np.sqrt(
                    max(
                        geom_new.E * du_j**2
                        + 2.0 * geom_new.F * du_j * dv_j
                        + geom_new.G * dv_j**2,
                        0.0,
                    )
                )

                _, _, speed_sub, _ = self.predictor.compute_velocity(
                    surface, traj, u_s, v_s, z_a
                )
                ds_expect = speed_sub * abs(dz_sub)
                ratio = (
                    ds_actual / ds_expect if ds_expect > 1e-12 else 1.0
                )
                last_ratio = ratio
                last_speed_actual = (
                    ds_actual / abs(dz_sub) if dz_sub != 0 else 0.0
                )

                if (
                    ratio > self.C_tol or ratio < 1.0 / self.C_tol
                ) and bisect_level < self.max_bisect:
                    jump_detected = True
                    break

                u_s, v_s = u_j, v_j

            if not jump_detected:
                best_u, best_v = u_s, v_s
                best_Phi = last_Phi
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