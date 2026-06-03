#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_corridor_ellipsoid.py
============================
Обобщение corridor_3 с поверхностью безопасности E1 = Ellipsoid.

Поверхность безопасности — строгий трёхосный эллипсоид, заданный аналитически.
Пересечение луча с эллипсоидом — решение квадратного уравнения (без шума).

Рекомендуемые оси (по концепции сплющивания):
    a = b = R_max(E2) + ΔR   (ΔR ≈ 50–100 мм)
    c = H(E2) * 0.6           (сплюснутый по высоте)

Центр эллипсоида размещается в центре оправки (средняя точка по Z).
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass

from core.trajectory import Trajectory
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution


# ------------------------------------------------------------------
# 1. Класс эллипсоида (чистый NumPy, совместимый с AnalyticalSurface)
# ------------------------------------------------------------------
class EllipsoidSurface:
    """
    Трёхосный эллипсоид: x²/a² + y²/b² + z²/c² = 1.
    Параметризация (u, v):
        x = a * cos(u) * cos(v)
        y = b * sin(u) * cos(v)
        z = c * sin(v) + z_center
    где u ∈ [0, 2π), v ∈ [-π/2, π/2].
    """
    def __init__(self, a: float, b: float, c: float, z_center: float = 0.0):
        self.a = a
        self.b = b
        self.c = c
        self.z_center = z_center
        # Для совместимости с интерфейсом
        self.u_min = 0.0
        self.u_max = 2.0 * np.pi
        self.v_min = -np.pi / 2.0
        self.v_max = np.pi / 2.0

    def position(self, u, v):
        """Точка на поверхности."""
        x = self.a * np.cos(u) * np.cos(v)
        y = self.b * np.sin(u) * np.cos(v)
        z = self.c * np.sin(v) + self.z_center
        return np.array([x, y, z])

    def derivatives(self, u, v):
        """ru, rv, normal."""
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)

        r = np.array([
            a * cos_u * cos_v,
            b * sin_u * cos_v,
            c * sin_v + self.z_center
        ])
        ru = np.array([-a * sin_u * cos_v, b * cos_u * cos_v, 0.0])
        rv = np.array([-a * cos_u * sin_v, -b * sin_u * sin_v, c * cos_v])

        # Ненормированная нормаль (градиент неявной функции)
        nx = cos_u * cos_v / a
        ny = sin_u * cos_v / b
        nz = sin_v / c
        n_vec = np.array([nx, ny, nz])
        n_norm = np.linalg.norm(n_vec)
        if n_norm > 1e-12:
            normal = n_vec / n_norm
        else:
            normal = np.array([0.0, 0.0, 1.0])

        return {'r': r, 'ru': ru, 'rv': rv, 'normal': normal}

    def normal(self, u, v):
        return self.derivatives(u, v)['normal']

    def partial_u(self, u, v):
        return self.derivatives(u, v)['ru']

    def partial_v(self, u, v):
        return self.derivatives(u, v)['rv']

    def first_fundamental_form(self, u, v):
        """I форма (E, F, G)."""
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        E = a**2 * sin_u**2 * cos_v**2 + b**2 * cos_u**2 * cos_v**2
        F = (a**2 - b**2) * sin_u * cos_u * sin_v * cos_v
        G = a**2 * cos_u**2 * sin_v**2 + b**2 * sin_u**2 * sin_v**2 + c**2 * cos_v**2
        return E, F, G

    def second_fundamental_form(self, u, v):
        """II форма (L, M, N)."""
        a, b, c = self.a, self.b, self.c
        cos_v, sin_v = np.cos(v), np.sin(v)
        E, F, G = self.first_fundamental_form(u, v)
        Delta = E * G - F**2
        sqrt_Delta = np.sqrt(max(Delta, 1e-16))
        L = -a * b * c * cos_v**3 / sqrt_Delta
        M = 0.0
        N = -a * b * c * cos_v / sqrt_Delta
        return L, M, N

    def uv_from_point(self, point):
        """Декартовы -> параметрические (приближённо)."""
        x, y, z = point
        z_local = z - self.z_center
        u = np.arctan2(y / self.b, x / self.a)
        # Для v используем arcsin с ограничением
        sin_v = np.clip(z_local / self.c, -1.0, 1.0)
        v = np.arcsin(sin_v)
        return u, v

    def radius(self, z):
        """Радиус поперечного сечения на высоте z."""
        z_local = z - self.z_center
        if abs(z_local) > self.c:
            return 0.0
        cos_v = np.sqrt(1.0 - (z_local / self.c)**2)
        # Средний радиус (для a != b возвращаем геометрическое среднее)
        return np.sqrt(self.a * self.b) * cos_v


# ------------------------------------------------------------------
# 2. Аналитическая трассировка луча с эллипсоидом
# ------------------------------------------------------------------
class EllipsoidRayTracer:
    """
    Аналитический трассировщик лучей для эллипсоида.
    Решает квадратное уравнение |O + t*D|_ellipsoid = 1.
    """
    def __init__(self, ellipsoid: EllipsoidSurface):
        self.ellipsoid = ellipsoid

    def trace(self, origin, direction, t_min=1e-6, t_max=1e6):
        """
        Находит пересечение луча O + t*D с эллипсоидом.
        Возвращает (t, point) или (None, None).
        """
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        a, b, c = self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c
        zc = self.ellipsoid.z_center

        # Сдвигаем начало координат в центр эллипсоида
        ox, oy, oz = o[0], o[1], o[2] - zc
        dx, dy, dz = d[0], d[1], d[2]

        # Коэффициенты квадратного уравнения A*t² + B*t + C = 0
        A = (dx**2) / (a**2) + (dy**2) / (b**2) + (dz**2) / (c**2)
        B = 2.0 * ((ox * dx) / (a**2) + (oy * dy) / (b**2) + (oz * dz) / (c**2))
        C = (ox**2) / (a**2) + (oy**2) / (b**2) + (oz**2) / (c**2) - 1.0

        if abs(A) < 1e-14:
            # Луч почти параллелен поверхности — линейное уравнение
            if abs(B) < 1e-14:
                return None, None
            t = -C / B
            if t_min <= t <= t_max:
                pt = o + t * d
                return t, pt
            return None, None

        discr = B**2 - 4.0 * A * C
        if discr < 0:
            return None, None

        sqrt_discr = np.sqrt(discr)
        t1 = (-B - sqrt_discr) / (2.0 * A)
        t2 = (-B + sqrt_discr) / (2.0 * A)

        # Выбираем ближайший положительный корень, больший t_min
        best_t = None
        for t in sorted([t1, t2]):
            if t >= t_min:
                best_t = t
                break

        if best_t is None or best_t > t_max:
            return None, None

        pt = o + best_t * d
        return best_t, pt


# ------------------------------------------------------------------
# 3. Структуры данных и калькулятор
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
    Вычисляет ТСН в модели тени: E2 (оправка) -> E1 (эллипсоид).
    """
    def __init__(self,
                 lu_trajectory: Trajectory,
                 mandrel_surface,
                 safety_surface: EllipsoidSurface,
                 safe_distance: float = 10.0):
        self.traj = lu_trajectory
        self.mandrel = mandrel_surface
        self.safety = safety_surface
        self.tracer = EllipsoidRayTracer(safety_surface)
        self.safe_dist = safe_distance

    def _get_surface_normal(self, point_3d):
        """Нормаль к оправке E2 в точке point_3d."""
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

        print(f"Расчёт ТСН на эллипсоиде: {num_points} точек")
        print(f"  Эллипсоид: a={self.safety.a:.2f}, b={self.safety.b:.2f}, c={self.safety.c:.2f}, zc={self.safety.z_center:.2f}")

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

            # Аналитическая трассировка
            t, pt = self.tracer.trace(r, tau_proj, t_min=self.safe_dist, t_max=t_max)
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
                    print(f"  Точка {i}: луч не попал в эллипсоид")

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
# 4. Вспомогательные функции
# ------------------------------------------------------------------
def save_results_csv(result: CorridorShadowResult, filename: str = "tsn_shadow_ellipsoid.csv"):
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
    plt.title('Зависимость Φ(s) – модель тени (эллипсоид)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('phi_shadow_ellipsoid.png', dpi=150)
    plt.show()


def create_3d_visualization(result, mandrel_surface, safety_surface,
                            z_offset=0.0,
                            filename='corridor_shadow_ellipsoid_3d.html'):
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

    # Эллипсоид E1
    u_s = np.linspace(0, 2*np.pi, 80)
    v_s = np.linspace(-np.pi/2, np.pi/2, 40)
    Us, Vs = np.meshgrid(u_s, v_s)
    Xs, Ys, Zs = np.zeros_like(Us), np.zeros_like(Us), np.zeros_like(Us)
    for i in range(Us.shape[0]):
        for j in range(Us.shape[1]):
            p = safety_surface.position(Us[i,j], Vs[i,j])
            Xs[i,j], Ys[i,j], Zs[i,j] = p[0], p[1], p[2]
    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs + z_offset,
        opacity=0.2, colorscale='Reds',
        showscale=False, name='Безопасность (E1, эллипсоид)'
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
        title='Модель тени: ТСН на эллипсоиде',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1200, height=900
    )
    fig.write_html(filename)
    print(f"3D-график сохранён как {filename}")
    fig.show()


# ------------------------------------------------------------------
# 5. main
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

    # ===== Параметры эллипсоида E1 =====
    R_max = E2.cylinder_radius          # ~251.7 мм
    H = bound_opravka[3] - bound_opravka[0]  # ~768.5 мм
    z_center = (bound_opravka[0] + bound_opravka[3]) / 2.0  # центр оправки

    # По рекомендации: a = b = R_max + ΔR, c = H * 0.6
    DELTA_R = 80.0   # запас (50–100 мм)
    a = b = R_max + DELTA_R
    c = H * 0.6

    E1 = EllipsoidSurface(a, b, c, z_center=z_center)
    print(f"E2: R_max={R_max:.2f}, H={H:.2f}, bounds={bound_opravka}")
    print(f"E1: a={a:.2f}, b={b:.2f}, c={c:.2f}, zc={z_center:.2f}")

    # Проверка охвата: радиусы эллипсоида на границах оправки
    for z in bound_opravka:
        r_ell = E1.radius(z)
        r_opr = E2.radius(z) if z <= E2.u_max else 0.0
        print(f"  z={z:.2f}: r_ell={r_ell:.2f}, r_opr={r_opr:.2f}, delta={r_ell - r_opr:.2f}")

    # ===== Загрузка эталонной ЛУ =====
    import scipy.io
    data = scipy.io.loadmat('LU_data.mat')
    r_etalon = data['r']
    lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')
    print(f"\nЗагружена ЛУ: {r_etalon.shape[0]} точек, длина = {lu_trajectory.total_length:.2f}")

    # ===== Расчёт ТСН =====
    calculator = CorridorShadowCalculator(
        lu_trajectory=lu_trajectory,
        mandrel_surface=E2,
        safety_surface=E1,
        safe_distance=0
    )
    result = calculator.calculate(num_points=4000, t_max=1500.0)

    # ===== Сохранение и визуализация =====
    save_results_csv(result, 'tsn_shadow_ellipsoid.csv')
    plot_phi_vs_s(result)
    create_3d_visualization(result, E2, E1, z_offset=0.0)


if __name__ == "__main__":
    main()
