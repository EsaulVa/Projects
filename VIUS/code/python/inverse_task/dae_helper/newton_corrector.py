#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newton_corrector.py
===================
Корректор Ньютона для DAE-схемы обратной задачи намотки.

Проецирует предсказанную точку на многообразие алгебраической связи
Φ(u, z) = ⟨R(z) − r(u), m(u)⟩ = 0, сдвигая строго в координатах
поверхности (u, v).

Все формулы соответствуют отчёту «Обратная задача намотки нити…», 5 июня 2026 г.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from dae_helper.surface_geometry_pack import SurfaceGeometryPack


# ----------------------------------------------------------------------
# Результат работы корректора
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CorrectorResult:
    """
    Результат одного шага корректора Ньютона.

    Attributes
    ----------
    u_corr, v_corr : float
        Скорректированные координаты на поверхности.
    Phi : float
        Итоговая невязка связи (должна быть |Φ| < eps).
    iterations : int
        Число выполненных итераций Ньютона (0, если сразу попали в eps).
    converged : bool
        True, если |Φ| < eps после корректора.
    """

    u_corr: float
    v_corr: float
    Phi: float
    iterations: int
    converged: bool


# ----------------------------------------------------------------------
# NewtonCorrector
# ----------------------------------------------------------------------
class NewtonCorrector:
    """
    Корректор Ньютона для проекции на многообразие связи Φ = 0.

    На каждой итерации вычисляет геометрию в текущем кандидате,
    строит линеаризацию зазора и делает шаг вдоль поверхностного
    градиента связи до обнуления невязки.
    """

    def __init__(
        self,
        eps_Phi: float = 1e-10,
        max_iter: int = 7,
        u_bounds: Optional[Tuple[float, float]] = None,
        v_bounds: Optional[Tuple[float, float]] = None,
        v_periodic: bool = False,
        v_period: float = 2.0 * np.pi,
    ):
        """
        Parameters
        ----------
        eps_Phi : float
            Допуск по невязке связи |Φ| < eps_Phi считается сходимостью.
        max_iter : int
            Максимальное число итераций Ньютона на одном шаге.
        u_bounds, v_bounds : tuple(float, float) | None
            Жёсткие границы для координат (u_min, u_max) и (v_min, v_max).
            Если заданы, после каждого шага корректора координаты
            ограничиваются этими пределами (clamping).
        v_periodic : bool
            Если True, координата v считается периодической с периодом
            v_period (например, для эллипсоида u — долгота, период 2π).
            Пока применяется только к v; для u периодичность добавляется
            аналогично при необходимости.
        v_period : float
            Период для v (по умолчанию 2π).
        """
        self.eps_Phi = eps_Phi
        self.max_iter = max_iter
        self.u_bounds = u_bounds
        self.v_bounds = v_bounds
        self.v_periodic = v_periodic
        self.v_period = v_period

    # ==================================================================
    # Основной метод
    # ==================================================================
    def correct(
        self,
        surface,
        traj,
        u_pred: float,
        v_pred: float,
        z_target: float,
    ) -> CorrectorResult:
        """
        Спроецировать предсказанную точку (u_pred, v_pred) на многообразие
        связи Φ(u, z_target) = 0.

        Алгоритм (раздел 5.3 и псевдокод 7.1(l) отчёта):

        1.  Инициализация: u^(0) = u_pred, v^(0) = v_pred.
        2.  Цикл по n = 0 … max_iter−1:
            a.  Собрать геометрию в кандидате (u^(n), v^(n)).
            b.  Вектор нити: V_thread = R(z_target) − r(u^(n)).
            c.  Невязка: Φ^(n) = ⟨V_thread, m^(n)⟩.
            d.  Если |Φ^(n)| < eps_Phi — сходимость, возврат.
            e.  Координатный градиент: ∇_u Φ^(n) = −B·G⁻¹·P
                (формула 36, через SurfaceGeometryPack.grad_Phi).
            f.  Поверхностный градиент: ∇_s Φ^(n) = G⁻¹·∇_u Φ^(n)
                (формула 18).
            g.  Квадрат нормы: N_g = |∇_s Φ^(n)|²_G
                (формула 19).
            h.  Если N_g → 0 — корректор теряет сходимость
                (асимптотическое направление). Возврат с converged=False.
            i.  Шаг Ньютона (формула 31):
                    Δu = − Φ^(n) / N_g · ∇_s Φ^(n).
            j.  Обновление: u^(n+1) = u^(n) + Δu.
            k.  Clamping к границам (если заданы).
        3.  После цикла: вычислить Φ в последнем кандидате и вернуть
            результат с флагом converged = |Φ| < eps_Phi.

        Parameters
        ----------
        surface
            Поверхность с интерфейсом, совместимым с SurfaceGeometryPack.
        traj
            Траектория раскладчика с методами R(z) и R_deriv(z).
        u_pred, v_pred : float
            Предсказанные координаты (от предиктора Эйлера).
        z_target : float
            Значение параметра z на следующем шаге (z_{k+1}).

        Returns
        -------
        CorrectorResult
        """
        u_c = float(u_pred)
        v_c = float(v_pred)

        for nit in range(self.max_iter):
            # --- (a) Геометрия в кандидате ---------------------------
            try:
                geom = SurfaceGeometryPack.from_surface(surface, u_c, v_c)
            except ValueError as exc:
                # Вырожденная метрика или точка вне допустимой области
                return CorrectorResult(
                    u_corr=u_c, v_corr=v_c,
                    Phi=np.nan, iterations=nit, converged=False
                )

            # --- (b) Вектор нити --------------------------------------
            R_target = traj.R(z_target)
            V_thread = R_target - geom.r

            # --- (c) Невязка связи Φ ---------------------------------
            Phi = SurfaceGeometryPack.compute_Phi(R_target, geom.r, geom.normal)

            # --- (d) Критерий сходимости -----------------------------
            if abs(Phi) < self.eps_Phi:
                return CorrectorResult(
                    u_corr=u_c, v_corr=v_c,
                    Phi=Phi, iterations=nit, converged=True
                )

            # --- (e)(f) Градиенты связи -----------------------------
            grad_u = geom.grad_Phi(V_thread)      # формула 36
            grad_s = geom.surface_gradient(grad_u)  # формула 18

            # --- (g) Квадрат нормы -----------------------------------
            Ng = geom.norm_grad_sq(grad_u)          # формула 19

            # --- (h) Проверка на вырождение --------------------------
            if Ng < 1e-14:
                # Асимптотическое направление: κ_n → 0, градиент связи
                # обращается в нуль. Корректор Ньютона теряет сходимость.
                # Это фундаментальное ограничение метода (раздел 6.2 отчёта).
                return CorrectorResult(
                    u_corr=u_c, v_corr=v_c,
                    Phi=Phi, iterations=nit, converged=False
                )

            # --- (i) Шаг Ньютона (формула 31) ------------------------
            # Δu = − Φ / |∇_s Φ|²_G · ∇_s Φ
            delta = -Phi / Ng * grad_s

            # --- (j) Обновление кандидата ----------------------------
            u_c += float(delta[0])
            v_c += float(delta[1])

            # --- (k) Clamping и периодичность ------------------------
            u_c, v_c = self._apply_bounds(u_c, v_c)

        # --- После max_iter: финальная проверка ----------------------
        try:
            geom = SurfaceGeometryPack.from_surface(surface, u_c, v_c)
        except ValueError:
            return CorrectorResult(
                u_corr=u_c, v_corr=v_c,
                Phi=np.nan, iterations=self.max_iter, converged=False
            )

        R_target = traj.R(z_target)
        Phi_final = SurfaceGeometryPack.compute_Phi(
            R_target, geom.r, geom.normal
        )
        converged = abs(Phi_final) < self.eps_Phi

        return CorrectorResult(
            u_corr=u_c, v_corr=v_c,
            Phi=Phi_final, iterations=self.max_iter, converged=converged
        )

    # ==================================================================
    # Вспомогательные методы
    # ==================================================================
    def _apply_bounds(self, u: float, v: float) -> Tuple[float, float]:
        """
        Применить жёсткие границы и периодичность к координатам.
        """
        if self.u_bounds is not None:
            u_min, u_max = self.u_bounds
            u = max(u_min, min(u_max, u))

        if self.v_bounds is not None:
            v_min, v_max = self.v_bounds
            v = max(v_min, min(v_max, v))

        if self.v_periodic:
            # Приводим v к диапазону [−period/2, +period/2]
            half = self.v_period / 2.0
            v = ((v + half) % self.v_period) - half

        return u, v


# ----------------------------------------------------------------------
# Самотестирование (минимальное)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     # --- Поверхность: эллипсоид ---------------------------------------
#     a, b, c = 2.4, 2.0, 1.6

#     class DummyEllipsoid:
#         def position(self, u, v):
#             return np.array([
#                 a * np.cos(u) * np.cos(v),
#                 b * np.sin(u) * np.cos(v),
#                 c * np.sin(v)
#             ])

#         def derivatives(self, u, v):
#             cos_u, sin_u = np.cos(u), np.sin(u)
#             cos_v, sin_v = np.cos(v), np.sin(v)
#             r = self.position(u, v)
#             ru = np.array([-a * sin_u * cos_v, b * cos_u * cos_v, 0.0])
#             rv = np.array([-a * cos_u * sin_v, -b * sin_u * sin_v, c * cos_v])
#             n = np.array([cos_u * cos_v / a, sin_u * cos_v / b, sin_v / c])
#             n = n / np.linalg.norm(n)
#             return {"r": r, "ru": ru, "rv": rv, "normal": n}

#         def first_fundamental_form(self, u, v):
#             cos_u, sin_u = np.cos(u), np.sin(u)
#             cos_v, sin_v = np.cos(v), np.sin(v)
#             E = a**2 * sin_u**2 * cos_v**2 + b**2 * cos_u**2 * cos_v**2
#             F = (a**2 - b**2) * sin_u * cos_u * sin_v * cos_v
#             G = a**2 * cos_u**2 * sin_v**2 + b**2 * sin_u**2 * sin_v**2 + c**2 * cos_v**2
#             return E, F, G

#         def second_fundamental_form(self, u, v):
#             cos_u, sin_u = np.cos(u), np.sin(u)
#             cos_v, sin_v = np.cos(v), np.sin(v)
#             denom = np.sqrt(
#                 (cos_u * cos_v / a)**2 +
#                 (sin_u * cos_v / b)**2 +
#                 (sin_v / c)**2
#             )
#             L = a * b * c * cos_v / (a**2 * denom**3)
#             M = 0.0
#             N = a * b * c / (c**2 * denom**3)
#             return L, M, N

    # # --- Траектория: точка на внешнем эллипсоиде ----------------------
    # a1, b1, c1 = 3.0, 2.5, 2.0

    # class DummyTrajectory:
    #     def __init__(self, z0):
    #         self.z0 = z0

    #     def R(self, z):
    #         # Точка на внешнем эллипсоиде, смещённая по z
    #         u = np.pi / 3.0
    #         v = np.pi / 6.0 + 0.01 * (z - self.z0)
    #         return np.array([
    #             a1 * np.cos(u) * np.cos(v),
    #             b1 * np.sin(u) * np.cos(v),
    #             c1 * np.sin(v)
    #         ])

    #     def R_deriv(self, z):
    #         # Приблизительно
    #         return np.array([0.0, 0.0, 1.0])

    # # --- Тест ----------------------------------------------------------
    # print("=" * 60)
    # print("Самотестирование NewtonCorrector")
    # print("=" * 60)

    # surf = DummyEllipsoid()
    # traj = DummyTrajectory(z0=0.6)
    # corrector = NewtonCorrector(eps_Phi=1e-10, max_iter=7)

    # # Начальная точка: чуть смещённая от истинной
    # u0 = np.pi / 3.0 + 0.01
    # v0 = np.pi / 6.0 + 0.01
    # z_target = 0.0

    # result = corrector.correct(surf, traj, u0, v0, z_target)

    # print(f"Начальная точка: u={u0:.6f}, v={v0:.6f}")
    # print(f"Скорректировано: u={result.u_corr:.6f}, v={result.v_corr:.6f}")
    # print(f"Итераций: {result.iterations}")
    # print(f"Итоговая Φ: {result.Phi:.2e}")
    # print(f"Сошлось: {result.converged}")

    # Проверка: Phi должно быть < eps
    # assert result.converged, "Корректор не сошёлся"
    # assert abs(result.Phi) < 1e-10, f"Phi слишком велико: {result.Phi}"
    # print("✓ Все проверки пройдены.")