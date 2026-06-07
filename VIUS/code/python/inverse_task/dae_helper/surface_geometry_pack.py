#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surface_geometry_pack.py
========================
Геометрический пакет для DAE-предиктора–корректора обратной задачи намотки.

Принимает любую поверхность, реализующую интерфейс AnalyticalSurface:
  • position(u, v) → np.ndarray[3]
  • derivatives(u, v) → dict с ключами 'r', 'ru', 'rv', 'normal'
  • first_fundamental_form(u, v) → (E, F, G)
  • second_fundamental_form(u, v) → (L, M, N)

Вычисляет матрицы G, B, G⁻¹, проекции 3D-векторов на касательный базис,
градиент связи Φ, базовую скорость раскладчика и нормальную кривизну.

Все формулы соответствуют отчёту «Обратная задача намотки…», 5 июня 2026 г.
"""

import numpy as np
from dataclasses import dataclass
from typing import Protocol, Tuple


# ----------------------------------------------------------------------
# Интерфейс поверхности (минимальный)
# ----------------------------------------------------------------------
class Surface(Protocol):
    """Минимальный интерфейс, достаточный для SurfaceGeometryPack."""

    def position(self, u: float, v: float) -> np.ndarray: ...

    def derivatives(self, u: float, v: float) -> dict: ...

    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]: ...

    def second_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]: ...


# ----------------------------------------------------------------------
# Результат геометрического пакета
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SurfaceGeometryPack:
    """
    Полная геометрия точки (u, v) на поверхности.

    Поля
    ----
    r, ru, rv, normal : np.ndarray
        Радиус-вектор, касательные базисные векторы, единичная нормаль.
    E, F, G : float
        Компоненты первой фундаментальной формы (метрическая матрица).
    L, M, N : float
        Компоненты второй фундаментальной формы (матрица кривизны).
    G_inv : np.ndarray, shape (2, 2)
        Обратная метрическая матрица g^{αβ}.
    B : np.ndarray, shape (2, 2)
        Матрица второй фундаментальной формы b_{αβ}.
    det_G : float
        Определитель метрической матрицы EG − F².
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

    # ------------------------------------------------------------------
    # Фабричный метод
    # ------------------------------------------------------------------
    @classmethod
    def from_surface(cls, surface: Surface, u: float, v: float) -> "SurfaceGeometryPack":
        """Собрать геометрию из поверхности в точке (u, v)."""
        d = surface.derivatives(u, v)
        E, F, G = surface.first_fundamental_form(u, v)
        L, M, N_ff = surface.second_fundamental_form(u, v)

        det_G = E * G - F * F
        if abs(det_G) < 1e-14:
            raise ValueError(
                f"Вырожденная метрика в точке (u={u}, v={v}): det={det_G}"
            )

        inv_det = 1.0 / det_G
        G_inv = np.array([[G * inv_det, -F * inv_det],
                          [-F * inv_det,  E * inv_det]])

        B = np.array([[L, M],
                      [M, N_ff]])

        return cls(
            r=d["r"], ru=d["ru"], rv=d["rv"], normal=d["normal"],
            E=E, F=F, G=G,
            L=L, M=M, N=N_ff,
            G_inv=G_inv, B=B, det_G=det_G
        )

    # ==================================================================
    # 1. Проекции 3D-вектора на касательный базис
    # ==================================================================
    def project_on_basis(self, vec_3d: np.ndarray) -> np.ndarray:
        """
        Проекции вектора vec_3d на базис {ru, rv}.

        Формула (34) отчёта:
            P = ( ⟨vec, e₁⟩, ⟨vec, e₂⟩ )ᵀ

        Returns
        -------
        P : np.ndarray, shape (2,)
        """
        return np.array([
            np.dot(vec_3d, self.ru),
            np.dot(vec_3d, self.rv)
        ])

    # ==================================================================
    # 2. Координатный градиент связи ∇_u Φ
    # ==================================================================
    def grad_Phi(self, V_thread: np.ndarray) -> np.ndarray:
        """
        Координатный градиент зазора Φ по параметрам поверхности.

        Формула (36) отчёта:
            ∇_u Φ = − B · G⁻¹ · P,
        где P = project_on_basis(V_thread) — проекции вектора нити на базис.

        Parameters
        ----------
        V_thread : np.ndarray, shape (3,)
            Вектор нити **без нормировки**: V_thread = R(z) − r(u, v).
            (В отчёте исключены λ и τ, работаем напрямую с V_thread.)

        Returns
        -------
        grad_u : np.ndarray, shape (2,)
            Ковариантные компоненты (∂Φ/∂u, ∂Φ/∂v).
        """
        P = self.project_on_basis(V_thread)          # формула (34)
        # B @ G_inv @ P  — матричное произведение
        return -self.B @ self.G_inv @ P               # формула (36)

    # ==================================================================
    # 3. Поверхностный градиент ∇_S Φ  (контравариантный)
    # ==================================================================
    def surface_gradient(self, grad_u: np.ndarray) -> np.ndarray:
        """
        Поднятие индексов: контравариантный градиент на поверхности.

        Формула (18) отчёта:
            ∇_S Φ = G⁻¹ · ∇_u Φ

        Parameters
        ----------
        grad_u : np.ndarray, shape (2,)
            Ковариантный градиент (результат grad_Phi).

        Returns
        -------
        grad_s : np.ndarray, shape (2,)
            Контравариантные компоненты.
        """
        return self.G_inv @ grad_u                    # формула (18)

    # ==================================================================
    # 4. Квадрат нормы поверхностного градиента |∇_S Φ|²_G
    # ==================================================================
    def norm_grad_sq(self, grad_u: np.ndarray) -> float:
        """
        Квадрат нормы поверхностного градиента в метрике G.

        Формула (19) отчёта:
            |∇_S Φ|²_G = (∇_u Φ)ᵀ · G⁻¹ · (∇_u Φ)
                      = (∇_u Φ)ᵀ · (∇_S Φ)

        Используется как нормирующий множитель в корректоре Ньютона
        и при вычислении μ в предикторе.

        Parameters
        ----------
        grad_u : np.ndarray, shape (2,)

        Returns
        -------
        float
        """
        grad_s = self.surface_gradient(grad_u)
        return float(grad_u @ grad_s)                 # формула (19)

    # ==================================================================
    # 5. Базовая скорость раскладчика R'_∥
    # ==================================================================
    def base_velocity(self, R_prime: np.ndarray) -> np.ndarray:
        """
        Проекция скорости раскладчика на касательную плоскость,
        пересчитанная в координаты поверхности.

        Формула (24) / (37) отчёта:
            P_R = ( ⟨R′, e₁⟩, ⟨R′, e₂⟩ )ᵀ
            R′_∥ = G⁻¹ · P_R

        Parameters
        ----------
        R_prime : np.ndarray, shape (3,)
            Скорость раскладчика dR/dz в 3D.

        Returns
        -------
        Rp_parallel : np.ndarray, shape (2,)
            Контравариантные компоненты базовой скорости (du/dz, dv/dz).
        """
        P_R = self.project_on_basis(R_prime)          # формула (37)
        return self.G_inv @ P_R                       # формула (24)

    # ==================================================================
    # 6. Частная производная связи по z
    # ==================================================================
    def dPhi_dz(self, R_prime: np.ndarray) -> float:
        """
        Производная зазора по z при зафиксированных координатах поверхности.

        Формула (12) отчёта:
            ∂Φ/∂z = ⟨R′(z), m⟩

        Parameters
        ----------
        R_prime : np.ndarray, shape (3,)

        Returns
        -------
        float
        """
        return float(np.dot(R_prime, self.normal))    # формула (12)

    # ==================================================================
    # 7. Скалярный множитель μ
    # ==================================================================
    def compute_mu(self, R_prime: np.ndarray, V_thread: np.ndarray) -> float:
        """
        Скалярный множитель коррекции скорости.

        Формула (26) отчёта:
            μ = − [ ∂Φ/∂z + (∇_u Φ)ᵀ · R′_∥ ] / |∇_S Φ|²_G

        Parameters
        ----------
        R_prime : np.ndarray, shape (3,)
            Скорость раскладчика.
        V_thread : np.ndarray, shape (3,)
            Вектор нити R − r (не нормированный).

        Returns
        -------
        float
            Множитель μ. При |∇_S Φ|²_G → 0 возвращает 0.0 (сигнал
            для переключения на оптический корректор).
        """
        dphi_dz = self.dPhi_dz(R_prime)
        grad_u = self.grad_Phi(V_thread)
        Rp_par = self.base_velocity(R_prime)

        Ng = self.norm_grad_sq(grad_u)
        if Ng < 1e-14:
            return 0.0

        residual = dphi_dz + float(grad_u @ Rp_par)
        return -residual / Ng                         # формула (26)

    # ==================================================================
    # 8. Полная скорость точки укладки u′ = du/dz
    # ==================================================================
    def winding_velocity(self, R_prime: np.ndarray, V_thread: np.ndarray) -> np.ndarray:
        """
        Полная скорость изменения координат точки укладки.

        Формула (27) отчёта:
            u′ = R′_∥ + μ · ∇_S Φ

        Parameters
        ----------
        R_prime : np.ndarray, shape (3,)
        V_thread : np.ndarray, shape (3,)

        Returns
        -------
        u_prime : np.ndarray, shape (2,)
            (du/dz, dv/dz).
        """
        Rp_par = self.base_velocity(R_prime)
        grad_u = self.grad_Phi(V_thread)
        grad_s = self.surface_gradient(grad_u)
        mu = self.compute_mu(R_prime, V_thread)
        return Rp_par + mu * grad_s                   # формула (27)

    # ==================================================================
    # 9. Нормальная кривизна κ_n
    # ==================================================================
    def normal_curvature(self, direction: np.ndarray) -> float:
        """
        Нормальная кривизна поверхности в направлении direction.

        direction — контравариантные компоненты (u′, v′).
        κ_n = II(dir, dir) / I(dir, dir).

        Используется для диагностики: при κ_n → 0 градиент связи
        обращается в нуль и корректор теряет сходимость.

        Parameters
        ----------
        direction : np.ndarray, shape (2,)
            Контравариантные компоненты направления.

        Returns
        -------
        float
        """
        du, dv = direction[0], direction[1]
        II_val = self.L * du**2 + 2.0 * self.M * du * dv + self.N * dv**2
        I_val  = self.E * du**2 + 2.0 * self.F * du * dv + self.G * dv**2
        if abs(I_val) < 1e-15:
            return 0.0
        return II_val / I_val

    # ==================================================================
    # 10. Вспомогательные: длина вектора на поверхности, скорость
    # ==================================================================
    def metric_norm_sq(self, vec_uv: np.ndarray) -> float:
        """
        Квадрат длины вектора vec_uv = (du, dv) в метрике поверхности.

        Формула (4) отчёта:
            ‖V‖² = vᵀ G v = E·du² + 2F·du·dv + G·dv²
        """
        du, dv = vec_uv[0], vec_uv[1]
        return self.E * du**2 + 2.0 * self.F * du * dv + self.G * dv**2

    def metric_speed(self, vec_uv: np.ndarray) -> float:
        """Длина вектора скорости на поверхности: √(vᵀ G v)."""
        return np.sqrt(max(self.metric_norm_sq(vec_uv), 0.0))

    # ==================================================================
    # 11. Невязка связи Φ
    # ==================================================================
    @staticmethod
    def compute_Phi(R: np.ndarray, r: np.ndarray, normal: np.ndarray) -> float:
        """
        Зазор (невязка связи).

        Формула (7) отчёта:
            Φ = ⟨R − r, m⟩
        """
        return float(np.dot(R - r, normal))


