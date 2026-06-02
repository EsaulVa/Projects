#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_corridor_homothetic.py
==============================
Обобщение client_corridor_3.

Поверхность безопасности E1 строится как гомотетичная (масштабированная
в поперечном сечении относительно оси Z) к оправке E2.

Логика трассировки полностью повторяет corridor_3:
  – проекция касательной к ЛУ на касательную плоскость E2;
  – выпуск луча по направлению проекции наружу;
  – пересечение с E1 и проверка невязки Φ = ⟨R − r, m⟩.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from core.trajectory import Trajectory
from helpers.intersection import RayTracer, PiecewisePolynomialIntersection, RobustRevolutionIntersection
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution


# ------------------------------------------------------------------
# 1. Класс гомотетичной поверхности вращения
# ------------------------------------------------------------------
class ScaledPiecewisePolynomialRevolution(PiecewisePolynomialRevolution):
    """
    Поверхность вращения, гомотетичная базовой в поперечном сечении.

    Для каждой осевой координаты z = u радиус образующей умножается
    на коэффициент k:
        r_scaled(u) = k * r_base(u)

    Осевые границы (u_min, u_max) и азимут сохраняются, что соответствует
    гомотетии относительно оси Z в плоскостях z = const.
    """
    def __init__(self, base_surface: PiecewisePolynomialRevolution, scale_factor: float):
        # Инициализируем базовой логикой — полиномы те же,
        # но радиус цилиндра уже масштабирован.
        super().__init__(
            phi_coeffs=base_surface.phi_coeffs,
            R_coeffs=base_surface.R_coeffs,
            segment_bounds=[base_surface.a, base_surface.b, base_surface.c, base_surface.d],
            cylinder_radius=base_surface.cylinder_radius * scale_factor
        )
        self.base_surface = base_surface
        self.scale_factor = scale_factor

        # Копируем вспомогательные атрибуты, которые могут использовать solvers
        self.bounds = getattr(base_surface, 'bounds', None)
        self.axial_min = getattr(base_surface, 'axial_min', self.u_min)
        self.axial_max = getattr(base_surface, 'axial_max', self.u_max)

    def _compute_r_and_derivs(self, u):
        """Масштабируем радиус и его производные на коэффициент k."""
        r, rp, rpp = self.base_surface._compute_r_and_derivs(u)
        k = self.scale_factor
        return k * r, k * rp, k * rpp


# ------------------------------------------------------------------
# 2. Структуры данных и калькулятор (аналог CorridorShadowCalculator)
# ------------------------------------------------------------------
@dataclass
class CorridorShadowResult:
    """Результаты расчёта ТСН в модели тени."""
    s_array: np.ndarray       # параметр длины дуги ЛУ
    lu_points: np.ndarray     # точки на ЛУ (E2)
    safety_points: np.ndarray # точки ТСН на E1
    lambda_max: np.ndarray    # длины лучей
    valid_mask: np.ndarray    # маска успешных лучей
    phi_values: np.ndarray    # невязка Φ = <R - r, m>


class CorridorShadowCalculator:
    """
    Вычисляет ТСН в модели тени для пары E2 (оправка) и E1 (безопасность).
    E1 может быть любой поверхностью вращения, в частности гомотетичной E2.
    """
    def __init__(self,
                 lu_trajectory: Trajectory,
                 mandrel_surface,
                 safety_surface,
                 ray_tracer: RayTracer,
                 safe_distance: float = 10.0):
        self.traj = lu_trajectory
        self.mandrel = mandrel_surface
        self.safety = safety_surface
        self.tracer = ray_tracer
        self.safe_dist = safe_distance

    def _get_surface_normal(self, point_3d):
        """Возвращает нормаль к оправке E2 в точке point_3d."""
        if hasattr(self.mandrel, 'uv_from_point'):
            try:
                u, v = self.mandrel.uv_from_point(point_3d)
                return self.mandrel.normal(u, v)
            except (ValueError, AttributeError):
                pass
        # Fallback: радиальная нормаль для поверхности вращения
        r_xy = np.hypot(point_3d[0], point_3d[1])
        if r_xy > 1e-6:
            return np.array([point_3d[0] / r_xy, point_3d[1] / r_xy, 0.0])
        return np.array([1.0, 0.0, 0.0])

    def _project_to_tangent_plane(self, vec, normal):
        """Проекция вектора на касательную плоскость: v_tan = v - (v·n)n."""
        dot = np.dot(vec, normal)
        return vec - dot * normal

    def _compute_phi(self, r_point, R_point):
        """Невязка связи Φ = ⟨R - r, m⟩."""
        m = self._get_surface_normal(r_point)
        return np.dot(R_point - r_point, m)

    def calculate(self, num_points: int = 200, t_max: float = 1500.0) -> CorridorShadowResult:
        s_array = np.linspace(0, self.traj.total_length, num_points)
        lu_points = np.zeros((num_points, 3))
        safety_points = np.zeros((num_points, 3))
        lambda_max = np.zeros(num_points)
        valid_mask = np.zeros(num_points, dtype=bool)
        phi_values = np.zeros(num_points)

        print(f"Расчёт ТСН в модели тени: {num_points} точек")
        k = getattr(self.safety, 'scale_factor', None)
        if k is not None:
            print(f"  E1 гомотетична E2 с k = {k:.3f}")

        for i, s in enumerate(s_array):
            r = self.traj.R(s)
            lu_points[i] = r
            tau_lu = self.traj.R_deriv(s)

            # Нормаль к оправке в текущей точке
            m = self._get_surface_normal(r)
            # Проекция касательной ЛУ на касательную плоскость E2
            tau_proj = self._project_to_tangent_plane(tau_lu, m)

            # Нормализация направления луча
            norm_proj = np.linalg.norm(tau_proj)
            if norm_proj < 1e-6:
                radial = np.array([-r[0], -r[1], 0.0])
                tau_proj = self._project_to_tangent_plane(radial, m)
                norm_proj = np.linalg.norm(tau_proj)
                if norm_proj < 1e-6:
                    tau_proj = np.array([1.0, 0.0, 0.0])
                else:
                    tau_proj /= norm_proj
            else:
                tau_proj /= norm_proj

            # Трассировка луча от точки на оправке наружу к E1
            try:
                t, pt = self.tracer.trace(
                    self.safety,
                    r,
                    tau_proj,
                    t_min=self.safe_dist,
                    t_max=t_max
                )
                if t is not None:
                    safety_points[i] = pt
                    lambda_max[i] = t
                    valid_mask[i] = True
                    phi_values[i] = self._compute_phi(r, pt)
                    if i % 50 == 0:
                        print(f"  [{i:3d}/{num_points}] Φ = {phi_values[i]:.2e}, λ = {lambda_max[i]:.1f}")
                else:
                    safety_points[i] = r + t_max * tau_proj
                    lambda_max[i] = np.inf
                    phi_values[i] = np.nan
                    if i % 20 == 0:
                        print(f"  Точка {i}: луч не попал в E1")
            except Exception as e:
                safety_points[i] = r + t_max * tau_proj
                lambda_max[i] = np.inf
                phi_values[i] = np.nan
                if i % 20 == 0:
                    print(f"  Точка {i}: ошибка трассировки: {e}")

        valid_phi = phi_values[valid_mask]
        if len(valid_phi) > 0:
            print(f"\nСтатистика Φ (невязка связи):")
            print(f"  |Φ| mean: {np.mean(np.abs(valid_phi)):.2e}")
            print(f"  |Φ| max:  {np.max(np.abs(valid_phi)):.2e}")
            print(f"  Φ ≈ 0 (|Φ| < 1e-6): {np.sum(np.abs(valid_phi) < 1e-6)}/{len(valid_phi)}")

        return CorridorShadowResult(
            s_array=s_array,
            lu_points=lu_points,
            safety_points=safety_points,
            lambda_max=lambda_max,
            valid_mask=valid_mask,
            phi_values=phi_values
        )


# ------------------------------------------------------------------
# 3. Вспомогательные функции (сохранение, визуализация)
# ------------------------------------------------------------------
def save_results_csv(result: CorridorShadowResult, filename: str = "tsn_shadow_homothetic.csv"):
    """Сохраняет результаты в CSV (s, X, Y, Z, lambda, valid, phi)."""
    df = pd.DataFrame({
        's': result.s_array,
        'X': result.safety_points[:, 0],
        'Y': result.safety_points[:, 1],
        'Z': result.safety_points[:, 2],
        'lambda': result.lambda_max,
        'valid': result.valid_mask,
        'phi': result.phi_values
    })
    df.to_csv(filename, index=False)
    print(f"Результаты сохранены в {filename}")


def plot_phi_vs_s(result: CorridorShadowResult):
    """Строит график зависимости Φ(s)."""
    s_valid = result.s_array[result.valid_mask]
    phi_valid = result.phi_values[result.valid_mask]
    plt.figure(figsize=(10, 5))
    plt.plot(s_valid, phi_valid, 'b.-', markersize=3, linewidth=0.8, label='Φ(s)')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Параметр s (длина дуги ЛУ)')
    plt.ylabel('Невязка Φ')
    plt.title('Зависимость Φ(s) – модель тени (гомотетичная E1)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phi_shadow_homothetic.png', dpi=150)
    plt.show()
    print("График Φ(s) сохранён как phi_shadow_homothetic.png")


def create_3d_visualization(result: CorridorShadowResult,
                            mandrel_surface,
                            safety_surface,
                            z_offset: float = 0.0,
                            filename: str = 'corridor_shadow_homothetic_3d.html'):
    """Создаёт 3D-сцену с оправкой, безопасностью, ЛУ, ТСН и лучами."""
    fig = go.Figure()

    # --- Оправка E2 ---
    u_mandrel = np.linspace(mandrel_surface.u_min, mandrel_surface.u_max, 60)
    v_mandrel = np.linspace(0, 2 * np.pi, 40)
    Um, Vm = np.meshgrid(u_mandrel, v_mandrel)
    Xm = np.zeros_like(Um)
    Ym = np.zeros_like(Um)
    Zm = np.zeros_like(Um)
    for i in range(Um.shape[0]):
        for j in range(Um.shape[1]):
            p = mandrel_surface.position(Um[i, j], Vm[i, j])
            Xm[i, j], Ym[i, j], Zm[i, j] = p[0], p[1], p[2]
    fig.add_trace(go.Surface(
        x=Xm, y=Ym, z=Zm + z_offset,
        opacity=0.4, colorscale='Blues',
        showscale=False, name='Оправка (E2)'
    ))

    # --- Поверхность безопасности E1 ---
    u_safe = np.linspace(safety_surface.u_min, safety_surface.u_max, 80)
    v_safe = np.linspace(0, 2 * np.pi, 40)
    Us, Vs = np.meshgrid(u_safe, v_safe)
    Xs = np.zeros_like(Us)
    Ys = np.zeros_like(Us)
    Zs = np.zeros_like(Us)
    for i in range(Us.shape[0]):
        for j in range(Us.shape[1]):
            p = safety_surface.position(Us[i, j], Vs[i, j])
            Xs[i, j], Ys[i, j], Zs[i, j] = p[0], p[1], p[2]
    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs + z_offset,
        opacity=0.2, colorscale='Reds',
        showscale=False, name='Безопасность (E1, гомотетичная)'
    ))

    # --- Линия укладки ---
    lu_global = result.lu_points.copy()
    lu_global[:, 2] += z_offset
    fig.add_trace(go.Scatter3d(
        x=lu_global[:, 0], y=lu_global[:, 1], z=lu_global[:, 2],
        mode='lines+markers', line=dict(color='blue', width=4),
        marker=dict(size=3), name='Линия укладки (E2)'
    ))

    # --- ТСН (только валидные) ---
    valid = result.valid_mask
    tsn_global = result.safety_points[valid].copy()
    tsn_global[:, 2] += z_offset
    fig.add_trace(go.Scatter3d(
        x=tsn_global[:, 0], y=tsn_global[:, 1], z=tsn_global[:, 2],
        mode='lines+markers', line=dict(color='red', width=3),
        marker=dict(size=3), name='ТСН (E1)'
    ))

    # --- Лучи ---
    for i in np.where(valid)[0]:
        p1 = result.lu_points[i].copy()
        p1[2] += z_offset
        p2 = result.safety_points[i].copy()
        p2[2] += z_offset
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', line=dict(color='green', width=1.5),
            showlegend=False
        ))

    fig.update_layout(
        title='Модель тени: ТСН на гомотетичной поверхности безопасности',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        width=1200, height=900
    )
    fig.write_html(filename)
    print(f"3D-график сохранён как {filename}")
    fig.show()


# ------------------------------------------------------------------
# 4. main
# ------------------------------------------------------------------
def main():
    # ===== Параметры оправки E2 (как в corridor_3) =====
    phi_c_opravka = [
        0.0000000005642, -0.0000003012748, 0.0000605882383,
        -0.0099656628535, 2.9503573330764
    ]
    R_c_opravka = [
        -344.1468891010463, 3932.5139101580062, -17756.7012553763525,
        39582.6812110246392, -43518.6731429065403, 19122.1758646943599
    ]
    bound_opravka = [0, 234.27, 534.27, 768.54]
    cyl_r_opravka = 251.705

    E2 = PiecewisePolynomialRevolution(
        phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka
    )

        # В fnc_ballon_hybrid.py:
    k_scale = 1.1 # Коэффициент раздувания баллона

    # Оставляем полиномы от оправки, но множим на масштаб!
    R_c_safe_scaled = [c * k_scale for c in R_c_opravka]
    # Оставляем полиномы от оправки, но множим на масштаб!
    bound_safe_extended= [c * k_scale for c in bound_opravka]

    # Границы по Z берем с запасом!
    # bound_safe_extended = [0, 234.27, 534.27, 768.54, 900.0] # Добавили хвост до 900

    # E1 = PiecewisePolynomialRevolution(phi_c_opravka, R_c_safe_scaled, bound_safe_extended, cyl_r_opravka * k_scale)

    # ===== Поверхность безопасности E1 как гомотетичная к E2 =====
    # Коэффициент масштабирования (пример: 1.4, как отношение радиусов в corridor_3)
    SCALE_FACTOR = 1.5
    E1 = ScaledPiecewisePolynomialRevolution(E2, SCALE_FACTOR)
    print(f"E2: cyl_r = {E2.cylinder_radius:.3f}, bounds = {bound_opravka}")
    print(f"E1: cyl_r = {E1.cylinder_radius:.3f}, bounds = [{E1.a}, {E1.b}, {E1.c}, {E1.d}]")
    print(f"Масштабный коэффициент k = {SCALE_FACTOR}")

    # ===== Загрузка эталонной линии укладки =====
    import scipy.io
    data = scipy.io.loadmat('LU_data.mat')
    r_etalon = data['r']
    lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')
    print(f"Загружена ЛУ: {r_etalon.shape[0]} точек, длина = {lu_trajectory.total_length:.2f}")

    # ===== Трассировщик лучей =====
    tracer = RayTracer()
    # Регистрируем алгоритмы и для базового класса, и для масштабированного
    tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
    tracer.register(ScaledPiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
    # Альтернатива: RobustRevolutionIntersection (раскомментируйте при необходимости)
    # tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
    # tracer.register(ScaledPiecewisePolynomialRevolution, RobustRevolutionIntersection())

    # ===== Расчёт ТСН =====
    calculator = CorridorShadowCalculator(
        lu_trajectory=lu_trajectory,
        mandrel_surface=E2,
        safety_surface=E1,
        ray_tracer=tracer,
        safe_distance=10.0
    )
    result = calculator.calculate(num_points=200, t_max=1500.0)

    # ===== Сохранение и визуализация =====
    save_results_csv(result, 'tsn_shadow_homothetic.csv')
    plot_phi_vs_s(result)
    create_3d_visualization(result, E2, E1, z_offset=0.0)


if __name__ == "__main__":
    main()
