#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dae_predictor.py
================

Предиктор для DAE-схемы обратной задачи намотки.

Реализует шаг предиктора на основе аналитического дифференцирования
алгебраической связи Φ(u, z) = ⟨R(z) − r(u), m(u)⟩ = 0
с последующей экстраполяцией методом Эйлера.

Все формулы соответствуют отчёту «Обратная задача намотки нити…», 5 июня 2026 г.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol
from core.trajectory import Trajectory

from dae_helper.surface_geometry_pack import SurfaceGeometryPack


# # ----------------------------------------------------------------------
# # Интерфейс траектории раскладчика
# # ----------------------------------------------------------------------
# class Trajectory(Protocol):
#     """Траектория точки схода нити R(z)."""

#     def R(self, z: float) -> np.ndarray: ...

#     def R_deriv(self, z: float) -> np.ndarray: ...


# ----------------------------------------------------------------------
# Результат работы предиктора
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PredictorResult:
    """
    Полная диагностика одного шага предиктора.

    Attributes
    ----------
    u_pred, v_pred : float
        Предсказанные координаты после экстраполяции Эйлера.
    du_dz, dv_dz : float
        Полная скорость точки укладки (контравариантные компоненты).
    R_prime : np.ndarray
        Скорость раскладчика dR/dz в текущей точке.
    V_thread : np.ndarray
        Вектор нити R − r (не нормированный, формула 33 отчёта).
    mu : float
        Скалярный множитель коррекции (формула 26).
    grad_u : np.ndarray
        Ковариантный градиент связи ∇_u Φ.
    grad_s : np.ndarray
        Поверхностный (контравариантный) градиент ∇_s Φ.
    Ng : float
        Квадрат нормы |∇_s Φ|²_G — нормирующий множитель.
    dPhi_dz : float
        Частная производная ∂Φ/∂z (формула 12).
    Rp_parallel : np.ndarray
        Базовая скорость раскладчика R′_∥ в координатах поверхности.
    speed : float
        Длина скорости на поверхности √(u′ᵀ G u′).
    optical_fallback : bool
        True, если |∇_s Φ|²_G слишком мал. Сигнал для переключения
        на оптический корректор (трассировку лучей).
    """

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

    Вычисляет полную скорость точки укладки u′ = du/dz через геометрическую
    декомпозицию (формула 27) и экстраполирует координаты методом Эйлера
    (формула 28).
    """

    def __init__(self, max_speed: Optional[float] = 50.0):
        """
        Parameters
        ----------
        max_speed : float | None
            Максимально допустимая скорость перемещения по поверхности
            (в единицах длины на единицу z). Если итоговая скорость
            превышает порог, она пропорционально уменьшается.
            None — без ограничения.
        """
        self.max_speed = max_speed

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------
    def predict_step(
        self,
        surface_geom: SurfaceGeometryPack,
        traj: Trajectory,
        u: float,
        v: float,
        z: float,
        dz: float,
    ) -> PredictorResult:
        """
        Выполнить один шаг предиктора от (u, v) при параметре z на шаг dz.

        Алгоритм (псевдокод раздела 7.1 отчёта, пункты a–k):

        1.  Вектор нити: V_thread = R(z) − r(u, v).
            Работаем с не нормированным вектором (раздел 6.4, формула 33).

        2.  Скорость раскладчика: R′ = dR/dz.

        3.  Базовая скорость (проекция R′ на касательную плоскость,
            пересчитанная в координаты поверхности):
                R′_∥ = G⁻¹ · P_R,  где P_R = (⟨R′, ru⟩, ⟨R′, rv⟩)ᵀ
            (формула 24 / 37).

        4.  Частная производная связи по z:
                ∂Φ/∂z = ⟨R′, m⟩  (формула 12).

        5.  Координатный градиент связи:
                ∇_u Φ = −B · G⁻¹ · P,
            где P = project_on_basis(V_thread) (формула 36).

        6.  Поверхностный градиент:
                ∇_s Φ = G⁻¹ · ∇_u Φ  (формула 18).

        7.  Квадрат нормы:
                N_g = |∇_s Φ|²_G = (∇_u Φ)ᵀ · ∇_s Φ  (формула 19).

        8.  Скалярный множитель:
                μ = −[∂Φ/∂z + (∇_u Φ)ᵀ · R′_∥] / N_g  (формула 26).
            При N_g → 0 (асимптотическое направление, κ_n → 0) градиент
            связи обращается в нуль — корректор Ньютона теряет сходимость.
            Полагаем μ = 0 и поднимаем флаг optical_fallback.

        9.  Полная скорость:
                u′ = R′_∥ + μ · ∇_s Φ  (формула 27).

        10. Ограничение скорости (опционально).

        11. Экстраполяция Эйлера:
                u_{pred} = u + u′ · dz
                v_{pred} = v + v′ · dz  (формула 28).
        """

        # --- (b) Вектор нити: V_thread = R(z) − r(u, v) ----------------
        R = traj.R(z)
        V_thread = R - surface_geom.r

        # --- (c) Скорость раскладчика ----------------------------------
        R_prime = traj.R_deriv(z)

        # --- (d) Координатный градиент связи ∇_u Φ --------------------
        # Формула (36): ∇_u Φ = −B · G⁻¹ · P
        grad_u = surface_geom.grad_Phi(V_thread)

        # --- (e) Поверхностный градиент ∇_s Φ ------------------------
        # Формула (18): поднятие индексов
        grad_s = surface_geom.surface_gradient(grad_u)

        # --- (f) Квадрат нормы |∇_s Φ|²_G -----------------------------
        # Формула (19)
        Ng = surface_geom.norm_grad_sq(grad_u)

        # --- (g) Частная производная ∂Φ/∂z -----------------------------
        # Формула (12): ∂Φ/∂z = ⟨R′(z), m⟩
        dphi_dz = surface_geom.dPhi_dz(R_prime)

        # --- (h) Базовая скорость раскладчика R′_∥ ---------------------
        # Формула (24) / (37)
        Rp_parallel = surface_geom.base_velocity(R_prime)

        # --- (i) Скалярный множитель μ ---------------------------------
        # Формула (26):
        #     μ = − [ ∂Φ/∂z + (∇_u Φ)ᵀ · R′_∥ ] / |∇_s Φ|²_G
        #
        # При Ng → 0 (асимптотическое направление) корректор Ньютона
        # теряет сходимость — это фундаментальное ограничение метода
        # (раздел 6.2 отчёта). В таком случае μ = 0, и предиктор
        # работает только на базовой скорости раскладчика.
        optical_fallback = False
        if Ng < 1e-14:
            mu = 0.0
            optical_fallback = True
        else:
            residual = dphi_dz + float(grad_u @ Rp_parallel)
            mu = -residual / Ng

        # --- (j) Полная скорость точки укладки u′ -----------------------
        # Формула (27): u′ = R′_∥ + μ · ∇_s Φ
        u_prime = Rp_parallel + mu * grad_s
        du_dz, dv_dz = float(u_prime[0]), float(u_prime[1])

        # --- Ограничение скорости (защита от катастрофических шагов) ---
        speed = surface_geom.metric_speed(u_prime)
        if self.max_speed is not None and speed > self.max_speed:
            scale = self.max_speed / speed
            du_dz *= scale
            dv_dz *= scale
            u_prime = np.array([du_dz, dv_dz])
            speed = self.max_speed

        # --- (k) Экстраполяция методом Эйлера ---------------------------
        # Формула (28)
        u_pred = u + du_dz * dz
        v_pred = v + dv_dz * dz

        return PredictorResult(
            u_pred=u_pred,
            v_pred=v_pred,
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

    # ------------------------------------------------------------------
    # Удобная обёртка: сам создаёт SurfaceGeometryPack
    # ------------------------------------------------------------------
    def predict(
        self,
        surface,
        traj: Trajectory,
        u: float,
        v: float,
        z: float,
        dz: float,
    ) -> PredictorResult:
        """
        Удобная обёртка: самостоятельно создаёт SurfaceGeometryPack
        из поверхности и вызывает predict_step.
        """
        geom = SurfaceGeometryPack.from_surface(surface, u, v)
        return self.predict_step(geom, traj, u, v, z, dz)