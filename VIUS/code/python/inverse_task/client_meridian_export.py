#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_meridian_export.py
=========================
Экспорт меридиана оправки E2_raw как зависимости r(z) и демонстрация
DiffusedRevolutionSurface с preserve_z_parameter=True.

Выход:
  • meridian_E2_raw_rz.csv      — исходный меридиан (z, r, dr/dz, d²r/dz², κ)
  • meridian_E2_diffused_s.csv    — внутренний дамп фильтра (s, z, r, …)
  • meridian_E2_smooth_rz.csv   — сглаженный меридиан в координатах (z, r)
  • meridian_rz_comparison.png  — графики r(z) и кривизны
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from filters.surface_diffusion_filter import DiffusedRevolutionSurface, BoundaryCondition


def compute_curvature_rz(z, r):
    """Численная кривизна плоской кривой r(z)."""
    dr = np.gradient(r, z)
    d2r = np.gradient(dr, z)
    return np.abs(d2r) / (1.0 + dr**2)**1.5


def main():
    # =====================================================================
    # 1. Исходная поверхность E2 (corridor_3)
    # =====================================================================
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

    E2_raw = PiecewisePolynomialRevolution(
        phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka
    )
    print(f"E2_raw создана: z ∈ [{E2_raw.u_min:.3f}, {E2_raw.u_max:.3f}] мм")

    # =====================================================================
    # 2. Экспорт исходного меридиана r(z) в CSV
    # =====================================================================
    N = 2000
    z_grid = np.linspace(E2_raw.u_min, E2_raw.u_max, N)
    r_grid = np.array([E2_raw.radius(z) for z in z_grid])

    dr_dz = np.gradient(r_grid, z_grid)
    d2r_dz2 = np.gradient(dr_dz, z_grid)
    kappa = compute_curvature_rz(z_grid, r_grid)

    df_raw = pd.DataFrame({
        'z': z_grid,
        'r': r_grid,
        'dr_dz': dr_dz,
        'd2r_dz2': d2r_dz2,
        'curvature': kappa,
    })
    df_raw.to_csv('meridian_E2_raw_rz.csv', index=False, float_format='%.6f')
    print("[OK] CSV сохранён: meridian_E2_raw_rz.csv")

    # =====================================================================
    # 3. Диффузия с сохранением z-параметризации
    # =====================================================================
    print("\nЗапуск диффузии с preserve_z_parameter=True …")
    E2_smooth = DiffusedRevolutionSurface(
        E2_raw,
        N=800,
        tau=5.0,
        n_steps=20,
        bc_left=BoundaryCondition.dirichlet(0.0),
        bc_right=BoundaryCondition.dirichlet(0.0),
        preserve_z_parameter=True,
        save_meridian_path='meridian_E2_diffused_s.csv'
    )
    print(f"[OK] E2_smooth: z-параметризация активна, z ∈ [{E2_smooth._z_min:.3f}, {E2_smooth._z_max:.3f}] мм")

    # =====================================================================
    # 4. Демонстрация доступа по z (preserve_z_parameter)
    # =====================================================================
    print("\n--- Демонстрация методов preserve_z_parameter ---")
    z_demo = np.linspace(E2_raw.u_min, E2_raw.u_max, 10)
    for z in z_demo:
        r_z = E2_smooth.radius(z)
        s_z = E2_smooth.s_from_z(z)
        pt = E2_smooth.position_by_z(z, 0.0)
        print(f"  z = {z:7.2f} мм  →  r = {r_z:8.4f} мм,  s = {s_z:8.4f} мм,  "
              f"pos = ({pt[0]:8.3f}, {pt[1]:8.3f}, {pt[2]:8.3f})")

    # =====================================================================
    # 5. Экспорт сглаженного меридиана r(z) в CSV
    # =====================================================================
    # z_smooth_grid = np.linspace(E2_smooth._z_min, E2_smooth._z_max, N)
    # r_smooth_grid = np.array([E2_smooth.radius(z) for z in z_smooth_grid])

    # dr_smooth = np.gradient(r_smooth_grid, z_smooth_grid)
    # d2r_smooth = np.gradient(dr_smooth, z_smooth_grid)
    # kappa_smooth = compute_curvature_rz(z_smooth_grid, r_smooth_grid)

    # df_smooth = pd.DataFrame({
    #     'z': z_smooth_grid,
    #     'r': r_smooth_grid,
    #     'dr_dz': dr_smooth,
    #     'd2r_dz2': d2r_smooth,
    #     'curvature': kappa_smooth,
    # })
    # df_smooth.to_csv('meridian_E2_smooth_rz.csv', index=False, float_format='%.6f')
    # print("\n[OK] CSV сохранён: meridian_E2_smooth_rz.csv")
        # =====================================================================
    # # 5. Экспорт сглаженного меридиана r(z) — КОРРЕКТНАЯ кривизна
    # # =====================================================================
    # z_smooth_grid = np.linspace(E2_smooth._z_min, E2_smooth._z_max, N)
    # r_smooth_grid = np.array([E2_smooth.radius(z) for z in z_smooth_grid])
    # # Аналитические производные через сплайн по s (инвариантная кривизна)
    # s_for_z = np.array([E2_smooth.s_from_z(z) for z in z_smooth_grid])
    # r_s = E2_smooth._cs_r(s_for_z)
    # z_s = E2_smooth._cs_z(s_for_z)
    # dr_ds = E2_smooth._cs_r_deriv(s_for_z)
    # dz_ds = E2_smooth._cs_z_deriv(s_for_z)
    # d2r_ds2 = E2_smooth._cs_r_deriv2(s_for_z)
    # d2z_ds2 = E2_smooth._cs_z_deriv2(s_for_z)
    
    # # Производные по z через цепочку (для справки)
    # dr_dz = dr_ds / dz_ds
    # d2r_dz2 = (d2r_ds2 * dz_ds - dr_ds * d2z_ds2) / (dz_ds**3)
    
    # # Кривизна — инвариант параметризации, вычисляем по s (точно)
    # kappa_smooth = np.abs(d2r_ds2 * dz_ds - dr_ds * d2z_ds2) / \
    #                (dr_ds**2 + dz_ds**2 + 1e-12)**1.5

    # df_smooth = pd.DataFrame({
    #     'z': z_smooth_grid,
    #     'r': r_s,
    #     'dr_dz': dr_dz,
    #     'd2r_dz2': d2r_dz2,
    #     'curvature': kappa_smooth,
    # })
    # df_smooth.to_csv('meridian_E2_smooth_rz.csv', index=False, float_format='%.6f')

        # =====================================================================
    # 5. Экспорт сглаженного меридиана r(z) — КОРРЕКТНАЯ кривизна
    # =====================================================================
    z_smooth_grid = np.linspace(E2_smooth._z_min, E2_smooth._z_max, N)
    r_smooth_grid = np.array([E2_smooth.radius(z) for z in z_smooth_grid])

    # --- Кривизна: параметрический подход через s (как в client_visualize_diffusion) ---
    s_for_curv = np.linspace(E2_smooth.u_min, E2_smooth.u_max, N)
    pts_curv = np.array([E2_smooth.position(s, 0.0) for s in s_for_curv])
    r_s = np.sqrt(pts_curv[:, 0]**2 + pts_curv[:, 1]**2)
    z_s = pts_curv[:, 2]
    
    # Производные по z через параметрическое дифференцирование
    dr_dz = np.gradient(r_s, z_s)
    d2r_dz2 = np.gradient(dr_dz, z_s)
    kappa_smooth = np.abs(d2r_dz2) / (1.0 + dr_dz**2)**1.5

    # Для справки: аналитические производные по s (альтернатива)
    # dr_ds = E2_smooth._cs_r_deriv(s_for_curv)
    # dz_ds = E2_smooth._cs_z_deriv(s_for_curv)
    # d2r_ds2 = E2_smooth._cs_r_deriv2(s_for_curv)
    # d2z_ds2 = E2_smooth._cs_z_deriv2(s_for_curv)
    # kappa_analytic = np.abs(d2r_ds2 * dz_ds - dr_ds * d2z_ds2) / \
    #                  (dr_ds**2 + dz_ds**2 + 1e-12)**1.5

    df_smooth = pd.DataFrame({
        'z': z_smooth_grid,
        'r': r_smooth_grid,
        'dr_dz': dr_dz,          # теперь гладкие
        'd2r_dz2': d2r_dz2,      # теперь гладкие
        'curvature': kappa_smooth,
    })
    df_smooth.to_csv('meridian_E2_smooth_rz.csv', index=False, float_format='%.6f')

    # =====================================================================
    # 6. Графики сравнения
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(z_grid, r_grid, 'b-', lw=2, label='E2_raw  r(z)')
    ax.plot(z_smooth_grid, r_smooth_grid, 'r-', lw=2, label='E2_smooth  r(z)')
    ax.set_xlabel('z, мм')
    ax.set_ylabel('r, мм')
    ax.set_title('Меридиан оправки: r(z)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(z_grid, kappa + 1e-12, 'b-', lw=1.2, label='E2_raw  κ(z)')
    ax.semilogy(z_smooth_grid, kappa_smooth + 1e-12, 'r-', lw=1.2, label='E2_smooth  κ(z)')
    ax.set_xlabel('z, мм')
    ax.set_ylabel('κ, 1/мм')
    ax.set_title('Кривизна меридиана')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('meridian_rz_comparison.png', dpi=150)
    print("[OK] График сохранён: meridian_rz_comparison.png")
    plt.show()

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
