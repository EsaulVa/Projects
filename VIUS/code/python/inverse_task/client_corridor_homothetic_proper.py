#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_corridor_homothetic_proper.py
=====================================
Обобщение corridor_3 с истинной гомотетией поверхности безопасности E1
относительно оправки E2.

Гомотетия с центром в нижнем донце (z = z_center):
    z_new = z_center + k * (z_old - z_center)
    r_new(z_new) = k * r_old(z_old)

Донца совпадают, профиль меридиана сохраняет форму,
E1 и E2 строго концентрически.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass

from core.trajectory import Trajectory
from helpers.intersection import RayTracer, PiecewisePolynomialIntersection
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution


# ------------------------------------------------------------------
# 1. Класс истинной гомотетичной поверхности вращения
# ------------------------------------------------------------------
class HomotheticRevolutionSurface(PiecewisePolynomialRevolution):
    """
    Поверхность вращения, гомотетичная базовой относительно точки на оси Z.

    Параметризация:
        X(u,v) = ( r_new(u) * cos v,  r_new(u) * sin v,  u )
    где
        w      = z_center + (u - z_center) / k          (параметр базовой)
        r_new  = k * r_base(w)
        r'_new = r'_base(w)          (т.к. dr_new/du = k * r'_base * (1/k))
        r''_new= r''_base(w) / k

    Геометрический смысл: меридиан E1 — это меридиан E2,
    масштабированный относительно точки (0,0,z_center) в плоскости (r,z).
    """

    def __init__(self, base_surface: PiecewisePolynomialRevolution,
                 scale_factor: float, z_center: float = None):
        self.base = base_surface
        self.k = scale_factor
        self.zc = z_center if z_center is not None else base_surface.u_min

        # Новые осевые границы (масштабируем относительно zc)
        new_a = self.zc + self.k * (base_surface.a - self.zc)
        new_b = self.zc + self.k * (base_surface.b - self.zc)
        new_c = self.zc + self.k * (base_surface.c - self.zc)
        new_d = self.zc + self.k * (base_surface.d - self.zc)
        new_cyl_r = base_surface.cylinder_radius * self.k

        # Инициализация родителя — полиномы не используются напрямую,
        # т.к. _compute_r_and_derivs переопределён, но нужны для совместимости
        super().__init__(
            phi_coeffs=base_surface.phi_coeffs,
            R_coeffs=base_surface.R_coeffs,
            segment_bounds=[new_a, new_b, new_c, new_d],
            cylinder_radius=new_cyl_r
        )
        # Переопределяем границы, т.к. родитель мог их изменить
        self.u_min = new_a
        self.u_max = new_d
        self.v_min = 0.0
        self.v_max = 2.0 * np.pi
        self.a, self.b, self.c, self.d = new_a, new_b, new_c, new_d
        self.cylinder_radius = new_cyl_r

    def _base_param(self, u_new: float) -> float:
        """Преобразование параметра u_new -> w (параметр базовой поверхности)."""
        return self.zc + (u_new - self.zc) / self.k

    def _compute_r_and_derivs(self, u_new):
        """
        r_new(u)  = k * r_base(w)
        r'_new(u) = r'_base(w)          (цепное правило: *k и *1/k сокращаются)
        r''_new(u)= r''_base(w) / k
        """
        w = self._base_param(u_new)
        r_base, rp_base, rpp_base = self.base._compute_r_and_derivs(w)
        r_new = self.k * r_base
        rp_new = rp_base
        rpp_new = rpp_base / self.k
        return r_new, rp_new, rpp_new

    def second_fundamental_form(self, u, v):
        """
        Правильная II форма для поверхности вращения.
        L = r'' / sqrt(1+r'^2)   (меридиан)
        N = -r / sqrt(1+r'^2)    (параллель)
        """
        r, rp, rpp = self._compute_r_and_derivs(u)
        denom = np.sqrt(1.0 + rp * rp)
        L = rpp / denom
        M = 0.0
        N = -r / denom
        return L, M, N

    def first_fundamental_form(self, u, v):
        """I форма: E = r^2, F = 0, G = 1 + r'^2."""
        r, rp, _ = self._compute_r_and_derivs(u)
        E = r * r
        F = 0.0
        G = 1.0 + rp * rp
        return E, F, G

    def position(self, u, v):
        r, _, _ = self._compute_r_and_derivs(u)
        return np.array([r * np.cos(v), r * np.sin(v), u])

    def derivatives(self, u, v):
        r, rp, _ = self._compute_r_and_derivs(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        point = np.array([r * cos_v, r * sin_v, u])
        ru = np.array([rp * cos_v, rp * sin_v, 1.0])
        rv = np.array([-r * sin_v, r * cos_v, 0.0])
        # Внешняя нормаль: ru x rv = (r*cos_v, r*sin_v, -r*rp)
        cross = np.array([r * cos_v, r * sin_v, -r * rp])
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-12:
            normal = cross / norm_cross
        else:
            normal = np.array([0.0, 0.0, 1.0])
        return {'r': point, 'ru': ru, 'rv': rv, 'normal': normal}

    def normal(self, u, v):
        return self.derivatives(u, v)['normal']

    def partial_u(self, u, v):
        return self.derivatives(u, v)['ru']

    def partial_v(self, u, v):
        return self.derivatives(u, v)['rv']

    def uv_from_point(self, point):
        x, y, z = point
        v = np.arctan2(y, x)
        u = z
        return u, v

    def radius(self, u):
        r, _, _ = self._compute_r_and_derivs(u)
        return r


# ------------------------------------------------------------------
# 2. Структуры данных и калькулятор (аналог CorridorShadowCalculator)
# ------------------------------------------------------------------
@dataclass
class CorridorShadowResult:
    s_array: np.ndarray
    lu_points: np.ndarray
    safety_points: np.ndarray
    lambda_max: np.ndarray
    valid_mask: np.ndarray
    phi_values: np.ndarray


class CorridorShadowCalculator:
    """
    Вычисляет ТСН в модели тени для пары E2 (оправка) и E1 (безопасность).
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
        r_xy = np.hypot(point_3d[0], point_3d[1])
        if r_xy > 1e-6:
            return np.array([point_3d[0] / r_xy, point_3d[1] / r_xy, 0.0])
        return np.array([1.0, 0.0, 0.0])

    def _project_to_tangent_plane(self, vec, normal):
        dot = np.dot(vec, normal)
        return vec - dot * normal

    def _compute_phi(self, r_point, R_point):
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
        print(f"  E1 гомотетична E2: k = {self.safety.k:.3f}, центр zc = {self.safety.zc:.3f}")

        for i, s in enumerate(s_array):
            r = self.traj.R(s)
            lu_points[i] = r
            tau_lu = self.traj.R_deriv(s)

            m = self._get_surface_normal(r)
            tau_proj = self._project_to_tangent_plane(tau_lu, m)

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

            try:
                t, pt = self.tracer.trace(
                    self.safety, r, tau_proj,
                    t_min=self.safe_dist, t_max=t_max
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
            s_array=s_array, lu_points=lu_points,
            safety_points=safety_points, lambda_max=lambda_max,
            valid_mask=valid_mask, phi_values=phi_values
        )


# ------------------------------------------------------------------
# 3. Вспомогательные функции
# ------------------------------------------------------------------
def save_results_csv(result: CorridorShadowResult, filename: str = "tsn_shadow_homothetic_proper.csv"):
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
    s_valid = result.s_array[result.valid_mask]
    phi_valid = result.phi_values[result.valid_mask]
    plt.figure(figsize=(10, 5))
    plt.plot(s_valid, phi_valid, 'b.-', markersize=3, linewidth=0.8, label='Φ(s)')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Параметр s (длина дуги ЛУ)')
    plt.ylabel('Невязка Φ')
    plt.title('Зависимость Φ(s) – модель тени (истинная гомотетия)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phi_shadow_homothetic_proper.png', dpi=150)
    plt.show()


def create_3d_visualization(result, mandrel_surface, safety_surface,
                            z_offset=0.0,
                            filename='corridor_shadow_homothetic_proper_3d.html'):
    fig = go.Figure()

    # Оправка E2
    u_m = np.linspace(mandrel_surface.u_min, mandrel_surface.u_max, 60)
    v_m = np.linspace(0, 2*np.pi, 40)
    Um, Vm = np.meshgrid(u_m, v_m)
    Xm, Ym, Zm = np.zeros_like(Um), np.zeros_like(Um), np.zeros_like(Um)
    for i in range(Um.shape[0]):
        for j in range(Um.shape[1]):
            p = mandrel_surface.position(Um[i,j], Vm[i,j])
            Xm[i,j], Ym[i,j], Zm[i,j] = p[0], p[1], p[2]
    fig.add_trace(go.Surface(
        x=Xm, y=Ym, z=Zm + z_offset,
        opacity=0.4, colorscale='Blues',
        showscale=False, name='Оправка (E2)'
    ))

    # Безопасность E1
    u_s = np.linspace(safety_surface.u_min, safety_surface.u_max, 80)
    v_s = np.linspace(0, 2*np.pi, 40)
    Us, Vs = np.meshgrid(u_s, v_s)
    Xs, Ys, Zs = np.zeros_like(Us), np.zeros_like(Us), np.zeros_like(Us)
    for i in range(Us.shape[0]):
        for j in range(Us.shape[1]):
            p = safety_surface.position(Us[i,j], Vs[i,j])
            Xs[i,j], Ys[i,j], Zs[i,j] = p[0], p[1], p[2]
    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs + z_offset,
        opacity=0.2, colorscale='Reds',
        showscale=False, name='Безопасность (E1, гомотетичная)'
    ))

    # ЛУ
    lu_g = result.lu_points.copy()
    lu_g[:,2] += z_offset
    fig.add_trace(go.Scatter3d(
        x=lu_g[:,0], y=lu_g[:,1], z=lu_g[:,2],
        mode='lines+markers', line=dict(color='blue', width=4),
        marker=dict(size=3), name='Линия укладки (E2)'
    ))

    # ТСН
    valid = result.valid_mask
    tsn_g = result.safety_points[valid].copy()
    tsn_g[:,2] += z_offset
    fig.add_trace(go.Scatter3d(
        x=tsn_g[:,0], y=tsn_g[:,1], z=tsn_g[:,2],
        mode='lines+markers', line=dict(color='red', width=3),
        marker=dict(size=3), name='ТСН (E1)'
    ))

    # Лучи
    for i in np.where(valid)[0]:
        p1 = result.lu_points[i].copy(); p1[2] += z_offset
        p2 = result.safety_points[i].copy(); p2[2] += z_offset
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', line=dict(color='green', width=1.5),
            showlegend=False
        ))

    fig.update_layout(
        title='Модель тени: ТСН на истинно гомотетичной поверхности безопасности',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1200, height=900
    )
    fig.write_html(filename)
    print(f"3D-график сохранён как {filename}")
    fig.show()


# ------------------------------------------------------------------
# 4. main
# ------------------------------------------------------------------
def main():
    # ===== Параметры оправки E2 =====
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

    # ===== E1 как истинно гомотетичная к E2 =====
    SCALE_FACTOR = 1.4
    Z_CENTER = bound_opravka[0]   # нижнее донце, z = 0
    E1 = HomotheticRevolutionSurface(E2, SCALE_FACTOR, z_center=Z_CENTER)

    print(f"E2: bounds = {bound_opravka}, cyl_r = {E2.cylinder_radius:.3f}")
    print(f"E1: bounds = [{E1.a:.3f}, {E1.b:.3f}, {E1.c:.3f}, {E1.d:.3f}], cyl_r = {E1.cylinder_radius:.3f}")
    print(f"Масштаб k = {SCALE_FACTOR}, центр гомотетии zc = {Z_CENTER}")

    # Проверка: радиусы на характерных высотах
    for z in [0, 234.27, 534.27, 768.54]:
        if E2.u_min <= z <= E2.u_max:
            r2 = E2.radius(z)
            z1 = Z_CENTER + SCALE_FACTOR * (z - Z_CENTER)
            r1 = E1.radius(z1)
            print(f"  z_E2={z:.2f} -> r={r2:.3f} | z_E1={z1:.2f} -> r={r1:.3f} | ratio={r1/r2:.3f}")

    # ===== Загрузка эталонной ЛУ =====
    import scipy.io
    data = scipy.io.loadmat('LU_data.mat')
    r_etalon = data['r']
    lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')
    print(f"\nЗагружена ЛУ: {r_etalon.shape[0]} точек, длина = {lu_trajectory.total_length:.2f}")

    # ===== Трассировщик =====
    tracer = RayTracer()
    tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
    tracer.register(HomotheticRevolutionSurface, PiecewisePolynomialIntersection())

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
    save_results_csv(result, 'tsn_shadow_homothetic_proper.csv')
    plot_phi_vs_s(result)
    create_3d_visualization(result, E2, E1, z_offset=0.0)


if __name__ == "__main__":
    main()
