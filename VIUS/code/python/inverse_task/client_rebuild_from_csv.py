#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_rebuild_from_csv.py
===========================
Читает меридиан из CSV (z, r), строит DiscreteRevolutionSurface,
пропускает через диффузионную фильтрацию и визуализирует результат.

Вход:  meridian_E2_raw_rz.csv (колонки z, r)
Выход: meridian_E2_smooth_from_csv_rz.csv
       meridian_rebuild_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from geometry.discrete_revolution_surface import DiscreteRevolutionSurface
from filters.surface_diffusion_filter import DiffusedRevolutionSurface, BoundaryCondition


def main():
    csv_path = Path("meridian_E2_raw_rz.csv")
    if not csv_path.exists():
        print(f"Файл {csv_path} не найден! Сначала запустите client_meridian_export.py.")
        return

    # =====================================================================
    # 1. Читаем CSV
    # =====================================================================
    df = pd.read_csv(csv_path)
    z = df["z"].values.astype(float)
    r = df["r"].values.astype(float)
    print(f"Загружено {len(z)} точек меридиана, z ∈ [{z.min():.3f}, {z.max():.3f}] мм")

    # =====================================================================
    # 2. Строим дискретную поверхность вращения
    # =====================================================================
    E2_raw = DiscreteRevolutionSurface(z, r)
    print(f"DiscreteRevolutionSurface создана: u ∈ [{E2_raw.u_min:.3f}, {E2_raw.u_max:.3f}]")

    # Проверка: точка на цилиндрическом участке
    z_test = 400.0
    r_test = E2_raw.radius(z_test)
    pt_test = E2_raw.position(z_test, 0.0)
    print(f"Проверка: z={z_test:.1f} → r={r_test:.4f} мм, pos=({pt_test[0]:.4f}, {pt_test[1]:.4f}, {pt_test[2]:.4f})")

    # =====================================================================
    # 3. Диффузия с сохранением z-параметризации
    # =====================================================================
    print("\nЗапуск диффузии …")
    E2_smooth = DiffusedRevolutionSurface(
        E2_raw,
        N=800,
        tau=2.0,
        n_steps=5,
        bc_left=BoundaryCondition.dirichlet(0.0),
        bc_right=BoundaryCondition.dirichlet(0.0),
        preserve_z_parameter=True,
        save_meridian_path="meridian_E2_diffused_from_csv_s.csv"
    )
    print(f"[OK] E2_smooth: z ∈ [{E2_smooth._z_min:.3f}, {E2_smooth._z_max:.3f}] мм")

    # =====================================================================
    # 4. Демонстрация доступа по z
    # =====================================================================
    print("\n--- Демонстрация preserve_z_parameter ---")
    z_demo = np.linspace(E2_raw.u_min, E2_raw.u_max, 10)
    for z_val in z_demo:
        r_z = E2_smooth.radius(z_val)
        s_z = E2_smooth.s_from_z(z_val)
        pt = E2_smooth.position_by_z(z_val, 0.0)
        print(f"  z={z_val:7.2f} → r={r_z:8.4f}, s={s_z:8.4f}, pos=({pt[0]:8.3f},{pt[1]:8.3f},{pt[2]:8.3f})")

    # =====================================================================
    # 5. Экспорт сглаженного меридиана r(z) в CSV
    # =====================================================================
    N_out = 2000
    z_out = np.linspace(E2_smooth._z_min, E2_smooth._z_max, N_out)
    r_out = np.array([E2_smooth.radius(z) for z in z_out])

    # Кривизна: параметрический подход через s (гладкая, без осцилляций)
    s_curv = np.linspace(E2_smooth.u_min, E2_smooth.u_max, N_out)
    pts_curv = np.array([E2_smooth.position(s, 0.0) for s in s_curv])
    r_s = np.sqrt(pts_curv[:, 0]**2 + pts_curv[:, 1]**2)
    z_s = pts_curv[:, 2]
    dr_dz = np.gradient(r_s, z_s)
    d2r_dz2 = np.gradient(dr_dz, z_s)
    kappa = np.abs(d2r_dz2) / (1.0 + dr_dz**2)**1.5

    df_smooth = pd.DataFrame({
        "z": z_out,
        "r": r_out,
        "dr_dz": dr_dz,
        "d2r_dz2": d2r_dz2,
        "curvature": kappa,
    })
    df_smooth.to_csv("meridian_E2_smooth_from_csv_rz.csv", index=False, float_format="%.6f")
    print("\n[OK] CSV сохранён: meridian_E2_smooth_from_csv_rz.csv")

    # =====================================================================
    # 6. Графики сравнения
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(z, r, "b-", lw=2, label="E2_raw  r(z) (из CSV)")
    ax.plot(z_out, r_out, "r-", lw=2, label="E2_smooth  r(z) (из CSV → diffused)")
    ax.set_xlabel("z, мм")
    ax.set_ylabel("r, мм")
    ax.set_title("Меридиан: восстановление из CSV")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Кривизна оригинала (из CSV) — численная, может быть шумной
    dr_raw = np.gradient(r, z)
    d2r_raw = np.gradient(dr_raw, z)
    kappa_raw = np.abs(d2r_raw) / (1.0 + dr_raw**2)**1.5
    ax.semilogy(z, kappa_raw + 1e-12, "b-", lw=1.2, label="E2_raw  κ(z)")
    ax.semilogy(z_out, kappa + 1e-12, "r-", lw=1.2, label="E2_smooth  κ(z)")
    ax.set_xlabel("z, мм")
    ax.set_ylabel("κ, 1/мм")
    ax.set_title("Кривизна меридиана (параметрический расчёт)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("meridian_rebuild_comparison.png", dpi=150)
    print("[OK] График сохранён: meridian_rebuild_comparison.png")
    plt.show()

    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)


if __name__ == "__main__":
    main()
