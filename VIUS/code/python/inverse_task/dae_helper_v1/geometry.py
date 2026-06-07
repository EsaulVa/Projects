#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py
===========
Поверхность (Ellipsoid) и геометрический пакет (SurfaceGeometryPack)
для DAE-обратной задачи намотки.
"""

import numpy as np
from dataclasses import dataclass


class Ellipsoid:
    """Эллипсоид с аналитическими производными."""

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
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        n = np.array([cu * cv / a, su * cv / b, sv / c])
        return n / np.linalg.norm(n)

    def derivatives(self, u, v):
        a, b, c = self.a, self.b, self.c
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        r = self.position(u, v)
        ru = np.array([-a * su * cv, b * cu * cv, 0.0])
        rv = np.array([-a * cu * sv, -b * su * sv, c * cv])
        return {"r": r, "ru": ru, "rv": rv, "normal": self.normal(u, v)}

    def first_fundamental_form(self, u, v):
        a, b, c = self.a, self.b, self.c
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        E = a**2 * su**2 * cv**2 + b**2 * cu**2 * cv**2
        F = (a**2 - b**2) * su * cu * sv * cv
        G = a**2 * cu**2 * sv**2 + b**2 * su**2 * sv**2 + c**2 * cv**2
        return E, F, G

    def second_fundamental_form(self, u, v):
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        denom = np.sqrt(
            (cu * cv / self.a)**2 +
            (su * cv / self.b)**2 +
            (sv / self.c)**2
        )
        L = -cv**2 / denom
        M = 0.0
        N = -1.0 / denom
        return L, M, N

    def metric_derivatives(self, u, v):
        a, b, c = self.a, self.b, self.c
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        Eu = 2 * (a**2 - b**2) * su * cu * cv**2
        Ev = -2 * cv * sv * (a**2 * su**2 + b**2 * cu**2)
        Fu = (a**2 - b**2) * np.cos(2*u) * sv * cv
        Fv = (a**2 - b**2) * su * cu * np.cos(2*v)
        Gu = 2 * (b**2 - a**2) * su * cu * sv**2
        Gv = 2 * sv * cv * (a**2 * cu**2 + b**2 * su**2 - c**2)
        return Eu, Ev, Fu, Fv, Gu, Gv

    def christoffel_symbols(self, u, v):
        E, F, G = self.first_fundamental_form(u, v)
        Eu, Ev, Fu, Fv, Gu, Gv = self.metric_derivatives(u, v)
        det = E * G - F**2
        if abs(det) < 1e-14:
            raise ValueError("Вырожденная метрика")
        inv = 1.0 / det
        g11, g12, g22 = G * inv, -F * inv, E * inv
        Gamma = np.zeros((2, 2, 2))
        Gamma[0, 0, 0] = 0.5 * (g11 * Eu + g12 * (2.0 * Fu - Ev))
        Gamma[0, 0, 1] = 0.5 * (g11 * Ev + g12 * Gu)
        Gamma[0, 1, 0] = Gamma[0, 0, 1]
        Gamma[0, 1, 1] = 0.5 * (g11 * (2.0 * Fv - Gu) + g12 * Gv)
        Gamma[1, 0, 0] = 0.5 * (g12 * Eu + g22 * (2.0 * Fu - Ev))
        Gamma[1, 0, 1] = 0.5 * (g12 * Ev + g22 * Gu)
        Gamma[1, 1, 0] = Gamma[1, 0, 1]
        Gamma[1, 1, 1] = 0.5 * (g12 * (2.0 * Fv - Gu) + g22 * Gv)
        return Gamma


@dataclass(frozen=True)
class SurfaceGeometryPack:
    """
    Полная геометрия точки (u,v) на поверхности.
    Собирается из любого объекта с интерфейсом:
      derivatives(), first_fundamental_form(), second_fundamental_form()
    """

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
