#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fnc_ellipsoid_hybrid.py
=======================
Клиент обратной задачи для ТСН, построенной на эллипсоиде (client_corridor_ellipsoid).

Отличия от fnc_ballon_hybrid:
  – нет z_offset (эллипсоид и оправка в одной системе координат);
  – начальная точка ищется через safe_initial_point + коррекцию Ньютона;
  – используется только DAE-предиктор (оптика не нужна, т.к. ТСН гладкая).
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# from geometry.piecewise_polynomial_revolution_fixed_v2 import PiecewisePolynomialRevolution, FixedPiecewisePolynomialRevolution
from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from geometry.fixed_surfaces_fixed import safe_initial_point
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method_fixed import compute_grad_Phi, compute_tangent_components, newton_corrector, compute_dr_dz
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor


# ------------------------------------------------------------------
# 1. Параметры оправки E2
# ------------------------------------------------------------------
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

E2 = FixedPiecewisePolynomialRevolution(
    phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka
)
E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

# На цилиндре (u = 400, rp = 0):
u_test = 400.0
L, M, N = E2.second_fundamental_form(u_test, 0.0)
print(f"Цилиндр: L={L:.4f}, M={M:.4f}, N={N:.4f}")
# Ожидается: L ≈ 0, N ≈ -251.7

# На днище (u = 0):
u_test = 0.0
L, M, N = E2.second_fundamental_form(u_test, 0.0)
print(f"Днище: L={L:.4f}, M={M:.4f}, N={N:.4f}")
# Ожидается: L = r''/√(1+r'²), N = -r/√(1+r'²)

u_test = 0.0
v_test = 1.2665
R_t = np.array([54.924, 174.894, -0.037])  # R0 из вашего вывода
r0 = E2.position(u_test, v_test)
m0 = E2.normal(u_test, v_test)
delta = R_t - r0
lam = np.linalg.norm(delta)
tau = delta / lam

up, vp = compute_tangent_components(E2, u_test, v_test, tau)
dPhidu_ana, dPhidv_ana = compute_grad_Phi(E2, u_test, v_test, up, vp, lam)

# Численный градиент
eps = 1e-5
Phi0 = np.dot(R_t - r0, m0)
Phi_u = np.dot(R_t - E2.position(u_test+eps, v_test), E2.normal(u_test+eps, v_test))
Phi_v = np.dot(R_t - E2.position(u_test, v_test+eps), E2.normal(u_test, v_test+eps))

dPhidu_num = (Phi_u - Phi0) / eps
dPhidv_num = (Phi_v - Phi0) / eps

print(f"Аналит: dPhi/du={dPhidu_ana:.4e}, dPhi/dv={dPhidv_ana:.4e}")
print(f"Числен: dPhi/du={dPhidu_num:.4e}, dPhi/dv={dPhidv_num:.4e}")
print(f"Отношение: du={dPhidu_num/dPhidu_ana:.2f}, dv={dPhidv_num/dPhidv_ana:.2f}")
# ------------------------------------------------------------------
# 2. Загрузка ТСН из CSV (без z_offset!)
# ------------------------------------------------------------------
df = pd.read_csv('tsn_shadow_ellipsoid.csv')
df_valid = df[df['valid'] == True].copy()
points_tsn = df_valid[['X', 'Y', 'Z']].values
print(f"ТСН: {len(points_tsn)} валидных точек")

# Траектория ТСН
traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"Z начала: {points_tsn[0,2]:.3f}, Z конца: {points_tsn[-1,2]:.3f}")
print(f"R'(0) = {traj.R_deriv(0.0)}")

# Инвертируем, если ТСН шла сверху вниз
if points_tsn[0, 2] > points_tsn[-1, 2]:
    points_tsn = points_tsn[::-1].copy()
    traj = Trajectory.from_points(points_tsn, method='cubic')
    print("Инвертировано: ТСН шла сверху вниз")

# ------------------------------------------------------------------
# 3. Начальная точка: safe_initial_point + коррекция Ньютона
# ------------------------------------------------------------------
R0 = traj.R(0.0)
print(f"R0 (ТСН в z=0): {R0}")

u0, v0 = safe_initial_point(E2, R0)
print(f"safe_initial_point: u0={u0:.4f}, v0={v0:.4f}")

# Корректируем до Φ = 0
u0, v0, Phi0, _, conv = newton_corrector(
    E2, traj, u0, v0, 0.0, eps_Phi=1e-12, max_iter=50
)
print(f"После коррекции: u0={u0:.4f}, v0={v0:.4f}, Φ0={Phi0:.2e}, conv={conv}")

if not conv or abs(Phi0) > 1e-8:
    print("ВНИМАНИЕ: Начальная точка не скорректировалась!")
    print("Пробуем fallback: поиск в окрестности...")
    # Fallback: перебор начальных приближений по сетке
    best_Phi = 1e9
    best_u, best_v = u0, v0
    for du in np.linspace(-20, 20, 21):
        for dv in np.linspace(-0.5, 0.5, 21):
            try:
                u_try = np.clip(u0 + du, E2.u_min, E2.u_max)
                v_try = np.mod(v0 + dv, 2*np.pi)
                u_c, v_c, Phi_c, _, conv_c = newton_corrector(
                    E2, traj, u_try, v_try, 0.0, eps_Phi=1e-10, max_iter=20
                )
                if conv_c and abs(Phi_c) < best_Phi:
                    best_Phi = abs(Phi_c)
                    best_u, best_v = u_c, v_c
            except Exception:
                pass
    u0, v0 = best_u, best_v
    print(f"Fallback: u0={u0:.4f}, v0={v0:.4f}, |Φ|={best_Phi:.2e}")

# ------------------------------------------------------------------
# 4. Загрузка эталонной ЛУ (для сравнения, если есть)
# ------------------------------------------------------------------
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']
    print(f"Эталонная ЛУ загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    r_etalon = None
    print("Эталонная ЛУ не найдена")

# ------------------------------------------------------------------
# 5. Настройка DAE-предиктора
# ------------------------------------------------------------------
solver_dae = SciPySolver(method='BDF', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)

# ------------------------------------------------------------------
# 6. Гибридный решатель (только DAE, без оптики)
# ------------------------------------------------------------------
print("\n===== Обратная задача: восстановление ЛУ на E2 (эллипсоид) =====")

result = inverse_winding_hybrid(
    E2, traj, u0, v0,
    count_points=5000,
    eps_Phi=1e-10,
    max_newton=20,
    max_bisect=4,
    jump_threshold=3.0,
    predictor_dae=dae_predictor,
    predictor_optical=None,      # ← оптика не нужна для гладкой ТСН
    eps_kappa=1e-2,
    u_margin=20.0,
    force_optical_after_fail=False
)

# ------------------------------------------------------------------
# 7. Извлечение и диагностика
# ------------------------------------------------------------------
z_vals = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
flags = result['flags']
line_E2 = result['points_3d']

n_bisected = np.sum(flags == 1)
print(f"\n=== Результаты ===")
print(f"Шагов с бисекцией: {n_bisected} из {len(z_vals)-1}")
print(f"Максимальная |Φ|: {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя |Φ|:      {np.mean(np.abs(Phi_hist)):.2e}")
print(f"Среднее итераций Ньютона: {np.mean(newton_iters_hist[1:]):.2f}")
print(f"Максимум итераций: {np.max(newton_iters_hist[1:])}")

# ------------------------------------------------------------------
# 8. Визуализация
# ------------------------------------------------------------------
fig = go.Figure()

# Сетка оправки
v_grid = np.linspace(0, 2*np.pi, 80)
u_grid_E2 = np.linspace(E2.u_min, E2.u_max, 60)
X2 = np.zeros((len(v_grid), len(u_grid_E2)))
Y2, Z2 = np.zeros_like(X2), np.zeros_like(X2)
for i, v in enumerate(v_grid):
    for j, u in enumerate(u_grid_E2):
        p = E2.position(u, v)
        X2[i,j], Y2[i,j], Z2[i,j] = p[0], p[1], p[2]
fig.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds',
    showscale=False, name='Оправка (E2)'
))

# ТСН
fig.add_trace(go.Scatter3d(
    x=points_tsn[:,0], y=points_tsn[:,1], z=points_tsn[:,2],
    mode='lines', line=dict(color='blue', width=4), name='ТСН (эллипсоид)'
))

# Восстановленная ЛУ
fig.add_trace(go.Scatter3d(
    x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
    mode='lines', line=dict(color='red', width=4), name='Восстановленная ЛУ'
))

# Эталонная ЛУ
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(
        x=r_etalon[:,0], y=r_etalon[:,1], z=r_etalon[:,2],
        mode='lines', line=dict(color='orange', width=2, dash='dot'),
        name='Эталонная ЛУ'
    ))

fig.update_layout(
    title='Обратная задача: эллипсоид -> оправка',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=1000, height=800
)
fig.write_html('fnc_ellipsoid_hybrid.html')
print("\nГрафик сохранён в fnc_ellipsoid_hybrid.html")

# Диагностические графики
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.semilogy(z_vals[1:], np.abs(Phi_hist[1:])+1e-16)
plt.title('Невязка |Φ|')
plt.xlabel('z')
plt.ylabel('|Φ|')
plt.grid(True, alpha=0.3)

plt.subplot(2,2,2)
plt.plot(z_vals[1:], newton_iters_hist[1:], '.')
plt.title('Итерации Ньютона')
plt.xlabel('z')
plt.ylabel('итерации')
plt.grid(True, alpha=0.3)

plt.subplot(2,2,3)
plt.plot(z_vals, kappa_n_hist)
plt.title('κ_n')
plt.xlabel('z')
plt.ylabel('κ_n')
plt.grid(True, alpha=0.3)

plt.subplot(2,2,4)
plt.plot(z_vals, u_hist, label='u(z)')
plt.plot(z_vals, v_hist, label='v(z)')
plt.legend()
plt.title('Параметры поверхности')
plt.xlabel('z')
plt.ylabel('u, v')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fnc_ellipsoid_diagnostics.png', dpi=150)
plt.show()
print("Диагностика сохранена в fnc_ellipsoid_diagnostics.png")
