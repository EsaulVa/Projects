#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part1_ellipsoid_forward_shadow.py
==================================
Часть 1:
1. Прямая задача (геодезическая) на внутреннем эллипсоиде E2.
2. Трассировка ТСН на внешний эллипсоид E1 по касательным к ЛУ.
3. Сохранение CSV + 3D-визуализация (Plotly HTML) + график Φ(s).

Параметры взяты из client_ellipsoid_compare_methods.py:
  E1: a=3.0, b=2.5, c=2.0
  E2: scale=0.8 от E1
  Начальные условия: u0=π/3, v0=π/6, alpha=π/6, s_end=30.0
"""

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# 1. КЛАСС ЭЛЛИПСОИДА (аналитические формулы)
# =====================================================================
class Ellipsoid:
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
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        n = np.array([
            cos_u * cos_v / a,
            sin_u * cos_v / b,
            sin_v / c
        ])
        return n / np.linalg.norm(n)

    def derivatives(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        r = self.position(u, v)
        ru = np.array([-a * sin_u * cos_v, b * cos_u * cos_v, 0.0])
        rv = np.array([-a * cos_u * sin_v, -b * sin_u * sin_v, c * cos_v])
        normal = self.normal(u, v)
        return {'r': r, 'ru': ru, 'rv': rv, 'normal': normal}

    def first_fundamental_form(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        E = a**2 * sin_u**2 * cos_v**2 + b**2 * cos_u**2 * cos_v**2
        F = (a**2 - b**2) * sin_u * cos_u * sin_v * cos_v
        G = a**2 * cos_u**2 * sin_v**2 + b**2 * sin_u**2 * sin_v**2 + c**2 * cos_v**2
        return E, F, G

    def metric_derivatives(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_v, sin_v = np.cos(v), np.sin(v)
        E_u = 2 * (a**2 - b**2) * sin_u * cos_u * cos_v**2
        E_v = -2 * cos_v * sin_v * (a**2 * sin_u**2 + b**2 * cos_u**2)
        F_u = (a**2 - b**2) * np.cos(2*u) * sin_v * cos_v
        F_v = (a**2 - b**2) * sin_u * cos_u * np.cos(2*v)
        G_u = 2 * (b**2 - a**2) * sin_u * cos_u * sin_v**2
        G_v = 2 * sin_v * cos_v * (a**2 * cos_u**2 + b**2 * sin_u**2 - c**2)
        return E_u, E_v, F_u, F_v, G_u, G_v

    def christoffel_symbols(self, u, v):
        E, F, G = self.first_fundamental_form(u, v)
        E_u, E_v, F_u, F_v, G_u, G_v = self.metric_derivatives(u, v)
        det = E * G - F**2
        if abs(det) < 1e-14:
            raise ValueError("Вырожденная метрика")
        inv_det = 1.0 / det
        g11 = G * inv_det
        g12 = -F * inv_det
        g22 = E * inv_det

        Gamma = np.zeros((2, 2, 2))
        # k=0 (u)
        Gamma[0, 0, 0] = 0.5 * (g11 * E_u + g12 * (2.0 * F_u - E_v))
        Gamma[0, 0, 1] = 0.5 * (g11 * E_v + g12 * G_u)
        Gamma[0, 1, 0] = Gamma[0, 0, 1]
        Gamma[0, 1, 1] = 0.5 * (g11 * (2.0 * F_v - G_u) + g12 * G_v)
        # k=1 (v)
        Gamma[1, 0, 0] = 0.5 * (g12 * E_u + g22 * (2.0 * F_u - E_v))
        Gamma[1, 0, 1] = 0.5 * (g12 * E_v + g22 * G_u)
        Gamma[1, 1, 0] = Gamma[1, 0, 1]
        Gamma[1, 1, 1] = 0.5 * (g12 * (2.0 * F_v - G_u) + g22 * G_v)
        return Gamma


# =====================================================================
# 2. ПРЯМАЯ ЗАДАЧА: геодезическая на эллипсоиде
# =====================================================================
def solve_geodesic(surface, u0, v0, alpha, s_end, num_points=300):
    """
    Строит геодезическую на поверхности.
    alpha — угол в касательной плоскости относительно направления ru.
    """
    E0, F0, G0 = surface.first_fundamental_form(u0, v0)
    det0 = E0 * G0 - F0**2
    if det0 <= 0:
        raise ValueError("Вырожденная метрика в начальной точке")

    # Начальные скорости (p=u', q=v'), |t|=1
    p0 = np.cos(alpha) / np.sqrt(E0) - F0 * np.sin(alpha) / np.sqrt(E0 * det0)
    q0 = np.sin(alpha) * np.sqrt(E0 / det0)

    # Проверка единичности
    ru0 = surface.derivatives(u0, v0)['ru']
    rv0 = surface.derivatives(u0, v0)['rv']
    t0 = ru0 * p0 + rv0 * q0
    assert abs(np.linalg.norm(t0) - 1.0) < 1e-10

    def rhs(s, y):
        u, v, p, q = y
        try:
            Gamma = surface.christoffel_symbols(u, v)
        except ValueError:
            return [0.0, 0.0, 0.0, 0.0]
        dp = -Gamma[0, 0, 0] * p**2 - 2 * Gamma[0, 0, 1] * p * q - Gamma[0, 1, 1] * q**2
        dq = -Gamma[1, 0, 0] * p**2 - 2 * Gamma[1, 0, 1] * p * q - Gamma[1, 1, 1] * q**2
        return [p, q, dp, dq]

    s_eval = np.linspace(0, s_end, num_points)
    sol = solve_ivp(rhs, [0, s_end], [u0, v0, p0, q0],
                    method='DOP853', t_eval=s_eval, rtol=1e-8, atol=1e-10)

    if sol.status != 0:
        raise RuntimeError(f"Интегрирование завершилось с ошибкой: {sol.message}")

    s_vals = sol.t
    uv = sol.y.T
    points = np.array([surface.position(u, v) for u, v in uv[:, :2]])
    return s_vals, uv, points


# =====================================================================
# 3. ТРАССИРОВКА ЛУЧА К ЭЛЛИПСОИДУ
# =====================================================================
def trace_ray_to_ellipsoid(ellipsoid, origin, direction):
    """
    Находит пересечение луча P(t) = origin + t*direction (t>0) с эллипсоидом.
    """
    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
    ox, oy, oz = origin
    dx, dy, dz = direction

    A = (dx/a)**2 + (dy/b)**2 + (dz/c)**2
    B = 2 * (ox*dx/a**2 + oy*dy/b**2 + oz*dz/c**2)
    C = (ox/a)**2 + (oy/b)**2 + (oz/c)**2 - 1.0

    if abs(A) < 1e-14:
        return None, None

    D = B**2 - 4 * A * C
    if D < 0:
        return None, None

    sqrtD = np.sqrt(D)
    t1 = (-B + sqrtD) / (2 * A)
    t2 = (-B - sqrtD) / (2 * A)

    t_candidates = [t for t in (t1, t2) if t > 1e-9]
    if not t_candidates:
        return None, None

    t = min(t_candidates)
    point = origin + t * direction
    return t, point


# =====================================================================
# 4. ОСНОВНОЙ БЛОК
# =====================================================================
def main():
    # --- Геометрия ---
    a1, b1, c1 = 3.0, 2.5, 2.0
    E1 = Ellipsoid(a1, b1, c1)
    scale = 0.8
    a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
    E2 = Ellipsoid(a2, b2, c2)

    print("=" * 60)
    print("ЧАСТЬ 1: Прямая задача на E2 + трассировка ТСН на E1")
    print(f"E1 (внешний): a={a1}, b={b1}, c={c1}")
    print(f"E2 (внутренний): a={a2}, b={b2}, c={c2} (scale={scale})")

    # --- Прямая задача на E2 ---
    u0, v0 = np.pi / 3.0, np.pi / 6.0
    alpha = np.pi / 6.0
    s_end = 30.0
    num_points = 3000

    print(f"\nПрямая задача на E2: u0={u0:.4f}, v0={v0:.4f}, alpha={alpha:.4f}, s_end={s_end}")
    s_vals, uv, lu_points = solve_geodesic(E2, u0, v0, alpha, s_end, num_points)
    print(f"Построено {len(s_vals)} точек ЛУ на E2")

    # --- Трассировка ТСН на E1 ---
    print(f"\nТрассировка ТСН на E1...")
    tsn_points = np.zeros((num_points, 3))
    lambda_vals = np.zeros(num_points)
    phi_vals = np.zeros(num_points)
    valid_mask = np.zeros(num_points, dtype=bool)

    for i in range(num_points):
        u, v = uv[i, 0], uv[i, 1]
        r = lu_points[i]
        p, q = uv[i, 2], uv[i, 3]
        geom = E2.derivatives(u, v)
        tau = geom['ru'] * p + geom['rv'] * q
        tau_norm = np.linalg.norm(tau)
        if tau_norm > 1e-12:
            tau = tau / tau_norm

        t, R_pt = trace_ray_to_ellipsoid(E1, r, tau)
        if t is not None:
            tsn_points[i] = R_pt
            lambda_vals[i] = t
            valid_mask[i] = True
            m = E2.normal(u, v)
            phi_vals[i] = np.dot(R_pt - r, m)
        else:
            tsn_points[i] = r + 100.0 * tau
            lambda_vals[i] = np.inf
            phi_vals[i] = np.nan

    valid_count = np.sum(valid_mask)
    print(f"Валидных точек ТСН: {valid_count}/{num_points}")
    if valid_count > 0:
        print(f"  Max |Φ| = {np.max(np.abs(phi_vals[valid_mask])):.2e}")
        print(f"  Mean |Φ| = {np.mean(np.abs(phi_vals[valid_mask])):.2e}")
        print(f"  Φ ≈ 0 (|Φ|<1e-6): {np.sum(np.abs(phi_vals[valid_mask]) < 1e-6)}/{valid_count}")

    # --- Сохранение CSV ---
    df = pd.DataFrame({
        's': s_vals,
        'X': tsn_points[:, 0],
        'Y': tsn_points[:, 1],
        'Z': tsn_points[:, 2],
        'lambda': lambda_vals,
        'valid': valid_mask,
        'phi': phi_vals
    })
    df.to_csv('part1_tsn_shadow.csv', index=False)
    print(f"\nCSV сохранён: part1_tsn_shadow.csv")

    # --- 3D Визуализация ---
    print("\nПостроение 3D-сцены...")
    fig = go.Figure()

    u_e = np.linspace(0, 2*np.pi, 60)
    v_e = np.linspace(-np.pi/2, np.pi/2, 40)
    Ue, Ve = np.meshgrid(u_e, v_e)

    X1 = a1 * np.cos(Ue) * np.cos(Ve)
    Y1 = b1 * np.sin(Ue) * np.cos(Ve)
    Z1 = c1 * np.sin(Ve)
    fig.add_trace(go.Surface(
        x=X1, y=Y1, z=Z1, opacity=0.15, colorscale='Blues',
        showscale=False, name='E1 (внешний, ТСН)'
    ))

    X2 = a2 * np.cos(Ue) * np.cos(Ve)
    Y2 = b2 * np.sin(Ue) * np.cos(Ve)
    Z2 = c2 * np.sin(Ve)
    fig.add_trace(go.Surface(
        x=X2, y=Y2, z=Z2, opacity=0.25, colorscale='Reds',
        showscale=False, name='E2 (внутренний, ЛУ)'
    ))

    fig.add_trace(go.Scatter3d(
        x=lu_points[:, 0], y=lu_points[:, 1], z=lu_points[:, 2],
        mode='lines', line=dict(color='red', width=4),
        name='ЛУ (геодезическая на E2)'
    ))

    valid = valid_mask
    fig.add_trace(go.Scatter3d(
        x=tsn_points[valid, 0], y=tsn_points[valid, 1], z=tsn_points[valid, 2],
        mode='lines', line=dict(color='green', width=4),
        name='ТСН (трассировка на E1)'
    ))

    for i in range(0, num_points, 10):
        if valid_mask[i]:
            fig.add_trace(go.Scatter3d(
                x=[lu_points[i, 0], tsn_points[i, 0]],
                y=[lu_points[i, 1], tsn_points[i, 1]],
                z=[lu_points[i, 2], tsn_points[i, 2]],
                mode='lines', line=dict(color='black', width=1),
                showlegend=False
            ))

    fig.update_layout(
        title='Часть 1: ЛУ на E2 → ТСН на E1 (трассировка лучей)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1200, height=900
    )
    fig.write_html('part1_corridor_shadow_3d.html')
    print(f"3D-сцена сохранена: part1_corridor_shadow_3d.html")

    # --- График Φ(s) ---
    plt.figure(figsize=(10, 5))
    s_valid = s_vals[valid]
    phi_valid = phi_vals[valid]
    plt.plot(s_valid, phi_valid, 'b.-', markersize=3, linewidth=0.8, label='Φ(s)')
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('s (длина дуги ЛУ)')
    plt.ylabel('Невязка Φ')
    plt.title('Зависимость Φ(s) — трассировка ТСН на E1')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('part1_phi_shadow.png', dpi=150)
    print(f"График Φ(s) сохранён: part1_phi_shadow.png")
    plt.show()

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
