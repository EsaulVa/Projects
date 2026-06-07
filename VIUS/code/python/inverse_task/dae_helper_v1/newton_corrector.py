#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newton_corrector.py
===================
Корректор Ньютона для проекции на многообразие связи Φ = 0.
"""

import numpy as np
from dataclasses import dataclass

from dae_helper_v1.geometry import SurfaceGeometryPack


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
