#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_visualize_diffusion.py
=============================
Визуализация поверхности оправки до и после диффузионной фильтрации.

Строит:
  • 3D-сцену: оригинальная и сглаженная поверхности (разные цвета, прозрачность)
  • 2D-график профиля меридиана (r vs z) — до и после
  • 2D-график кривизны профиля
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from filters.surface_diffusion_filter import DiffusedRevolutionSurface


def plot_meridian_profile(base_surface, diffused_surface, N=500):
    """2D-график профиля меридиана: r(u) и z(u)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Оригинал
    u_vals = np.linspace(base_surface.u_min, base_surface.u_max, N)
    pts_orig = np.array([base_surface.position(u, 0.0) for u in u_vals])
    r_orig = np.sqrt(pts_orig[:, 0]**2 + pts_orig[:, 1]**2)
    z_orig = pts_orig[:, 2]

    # Сглаженный
    s_vals = np.linspace(diffused_surface.u_min, diffused_surface.u_max, N)
    pts_diff = np.array([diffused_surface.position(s, 0.0) for s in s_vals])
    r_diff = np.sqrt(pts_diff[:, 0]**2 + pts_diff[:, 1]**2)
    z_diff = pts_diff[:, 2]

    ax = axes[0]
    ax.plot(z_orig, r_orig, 'b-', lw=2, label='Оригинал')
    ax.plot(z_diff, r_diff, 'r-', lw=2, label=f'Диффузия (τ={diffused_surface.tau}, n={diffused_surface.n_steps})')
    ax.set_xlabel('Z, мм')
    ax.set_ylabel('R, мм')
    ax.set_title('Профиль меридиана (r vs z)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # Разница
    ax = axes[1]
    # Интерполируем на общую сетку z для сравнения
    z_common = np.linspace(max(z_orig.min(), z_diff.min()),
                           min(z_orig.max(), z_diff.max()), 500)
    r_orig_i = np.interp(z_common, z_orig, r_orig)
    r_diff_i = np.interp(z_common, z_diff, r_diff)
    ax.plot(z_common, (r_diff_i - r_orig_i), 'g-', lw=1.5)
    ax.axhline(0, color='k', ls='--', lw=0.5)
    ax.set_xlabel('Z, мм')
    ax.set_ylabel('ΔR, мм')
    ax.set_title('Разница радиусов (сглаженный − оригинал)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diffusion_meridian_profile.png', dpi=150)
    print("График профиля: diffusion_meridian_profile.png")
    plt.show()


def plot_curvature_profile(base_surface, diffused_surface, N=500):
    """2D-график кривизны профиля меридиана."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Оригинал
    u_vals = np.linspace(base_surface.u_min, base_surface.u_max, N)
    pts = np.array([base_surface.position(u, 0.0) for u in u_vals])
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    z = pts[:, 2]
    # Численная кривизна плоской кривой (r(z))
    dr = np.gradient(r, z)
    d2r = np.gradient(dr, z)
    k_orig = np.abs(d2r) / (1 + dr**2)**1.5

    # Сглаженный
    s_vals = np.linspace(diffused_surface.u_min, diffused_surface.u_max, N)
    pts_d = np.array([diffused_surface.position(s, 0.0) for s in s_vals])
    r_d = np.sqrt(pts_d[:, 0]**2 + pts_d[:, 1]**2)
    z_d = pts_d[:, 2]
    dr_d = np.gradient(r_d, z_d)
    d2r_d = np.gradient(dr_d, z_d)
    k_diff = np.abs(d2r_d) / (1 + dr_d**2)**1.5

    ax.semilogy(z, k_orig + 1e-12, 'b-', lw=1.5, label='Оригинал')
    ax.semilogy(z_d, k_diff + 1e-12, 'r-', lw=1.5, label='Сглаженный')
    ax.set_xlabel('Z, мм')
    ax.set_ylabel('|κ|, 1/мм')
    ax.set_title('Кривизна профиля меридиана')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diffusion_curvature.png', dpi=150)
    print("График кривизны: diffusion_curvature.png")
    plt.show()


def create_3d_visualization(base_surface, diffused_surface, N_u=80, N_v=40):
    """3D-сцена с обеими поверхностями."""
    fig = go.Figure()

    # --- Оригинальная поверхность ---
    u = np.linspace(base_surface.u_min, base_surface.u_max, N_u)
    v = np.linspace(0, 2 * np.pi, N_v)
    U, V = np.meshgrid(u, v)

    X_orig = np.zeros_like(U)
    Y_orig = np.zeros_like(U)
    Z_orig = np.zeros_like(U)
    for i in range(N_u):
        for j in range(N_v):
            p = base_surface.position(U[j, i], V[j, i])
            X_orig[j, i] = p[0]
            Y_orig[j, i] = p[1]
            Z_orig[j, i] = p[2]

    fig.add_trace(go.Surface(
        x=X_orig, y=Y_orig, z=Z_orig,
        opacity=0.3, colorscale='Blues', showscale=False,
        name='Оригинал (E2)'
    ))

    # --- Сглаженная поверхность ---
    s = np.linspace(diffused_surface.u_min, diffused_surface.u_max, N_u)
    S, V2 = np.meshgrid(s, v)

    X_diff = np.zeros_like(S)
    Y_diff = np.zeros_like(S)
    Z_diff = np.zeros_like(S)
    for i in range(N_u):
        for j in range(N_v):
            p = diffused_surface.position(S[j, i], V2[j, i])
            X_diff[j, i] = p[0]
            Y_diff[j, i] = p[1]
            Z_diff[j, i] = p[2]

    fig.add_trace(go.Surface(
        x=X_diff, y=Y_diff, z=Z_diff,
        opacity=0.5, colorscale='Reds', showscale=False,
        name='Сглаженный (E2 diffused)'
    ))

    # --- Меридианы (профили) ---
    u_mer = np.linspace(base_surface.u_min, base_surface.u_max, 200)
    pts_mer_orig = np.array([base_surface.position(u, 0.0) for u in u_mer])
    pts_mer_orig_pi = np.array([base_surface.position(u, np.pi) for u in u_mer])

    s_mer = np.linspace(diffused_surface.u_min, diffused_surface.u_max, 200)
    pts_mer_diff = np.array([diffused_surface.position(s, 0.0) for s in s_mer])
    pts_mer_diff_pi = np.array([diffused_surface.position(s, np.pi) for s in s_mer])

    fig.add_trace(go.Scatter3d(
        x=pts_mer_orig[:, 0], y=pts_mer_orig[:, 1], z=pts_mer_orig[:, 2],
        mode='lines', line=dict(color='blue', width=6),
        name='Меридиан оригинал'
    ))
    fig.add_trace(go.Scatter3d(
        x=pts_mer_diff[:, 0], y=pts_mer_diff[:, 1], z=pts_mer_diff[:, 2],
        mode='lines', line=dict(color='red', width=6),
        name='Меридиан сглаженный'
    ))
    # Вторая сторона (v=π) для наглядности
    fig.add_trace(go.Scatter3d(
        x=pts_mer_orig_pi[:, 0], y=pts_mer_orig_pi[:, 1], z=pts_mer_orig_pi[:, 2],
        mode='lines', line=dict(color='blue', width=3, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=pts_mer_diff_pi[:, 0], y=pts_mer_diff_pi[:, 1], z=pts_mer_diff_pi[:, 2],
        mode='lines', line=dict(color='red', width=3, dash='dash'),
        showlegend=False
    ))

    fig.update_layout(
        title='Диффузионная фильтрация поверхности оправки',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1200, height=900
    )
    fig.write_html('diffusion_surface_3d.html')
    print("3D-сцена: diffusion_surface_3d.html")
    fig.show()


def main():
    # --- Поверхность оправки (из corridor_3) ---
    phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                     -0.0099656628535, 2.9503573330764]
    R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
                   39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
    bound_opravka = [0, 234.27, 534.27, 768.54]
    cyl_r_opravka = 251.705

    E2 = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)
    print(f"Оригинальная поверхность: u ∈ [{E2.u_min:.1f}, {E2.u_max:.1f}]")

    # --- Сглаженная поверхность ---
    # tau = 5.0 мм², n_steps = 20  →  полное время t = 100 мм²
    # Для сильного эффекта можно увеличить tau или n_steps
    E2_smooth = DiffusedRevolutionSurface(E2, N=800, tau=5.0, n_steps=20)
    print(f"Сглаженная поверхность: s ∈ [{E2_smooth.u_min:.1f}, {E2_smooth.u_max:.1f}]")

    # --- Визуализация ---
    print("Построение 2D-графиков...")
    plot_meridian_profile(E2, E2_smooth)
    plot_curvature_profile(E2, E2_smooth)

    print("Построение 3D-сцены...")
    create_3d_visualization(E2, E2_smooth)

    print("ГОТОВО")


if __name__ == "__main__":
    main()
