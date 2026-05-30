"""
Клиент для тестирования обратной задачи с исправленными параметрами.
Ключевые исправления:
1. max_newton=20 (было 5)
2. eps_Phi=1e-6 (было 1e-10)
3. jump_threshold=5.0 (было 1.0)
"""
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import sys
from pathlib import Path

# Добавляем путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.piecewise_polynomial_revolution_fixed_v2 import PiecewisePolynomialRevolution
from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method_fixed import newton_corrector
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor, RayTracer
from helpers.fixed_intersections import FixedRobustRevolutionIntersection
from helpers.inverse_winding_robust import (
    inverse_winding_robust,
    newton_corrector_stable,
    compute_dr_dz
)


# ======================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ======================================================================

print("===== Загрузка данных =====")

# --- Оправка E2 ---
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                 -0.0099656628535, 2.9503573330764]
R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
               39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705
E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka,
                                          bound_opravka, cyl_r_opravka)

# --- Безопасность E1 ---
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076,
              -0.0066486075257, 2.9473869159379]
R_c_safe = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463,
            27152.4105364360366, -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387
E1 = FixedPiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)

z_offset = (bound_safe[3] - bound_opravka[3]) / 2  # ≈ 93.708

# --- ТСН ---
df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True]
points_tsn = df_valid[['X', 'Y', 'Z']].values
print(f"ТСН загружена: {points_tsn.shape[0]} точек")

# Траектория ТСН
traj_tsn = Trajectory.from_points(points_tsn, method='cubic')
print(f"Траектория ТСН: total_length = {traj_tsn.total_length:.2f}")

# --- Эталонная ЛУ ---
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']
    print(f"Эталонная ЛУ: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    print("Файл LU_data.mat не найден")
    r_etalon = None


# ======================================================================
# 2. АНАЛИЗ Φ ДЛЯ ТСН
# ======================================================================

print("\n===== Проверка Φ для ТСН =====")

# Вычислим Φ в нескольких точках ТСН
s_check = np.linspace(0, traj_tsn.total_length, 20)
phi_vals = []

for s in s_check:
    R_z = traj_tsn.R(s)
    # Найдём ближайшую точку на E2
    try:
        u_approx = R_z[2]  # высота
        v_approx = np.arctan2(R_z[1], R_z[0])

        r = E2.position(u_approx, v_approx)
        m = E2.normal(u_approx, v_approx)

        Phi = np.dot(R_z - r, m)
        phi_vals.append(Phi)

        if len(phi_vals) % 5 == 0:
            print(f"  s={s:.1f}: Φ = {Phi:.2e}")
    except:
        phi_vals.append(np.nan)

phi_vals = np.array(phi_vals)
valid_phi = phi_vals[~np.isnan(phi_vals)]

if len(valid_phi) > 0:
    print(f"\nСтатистика Φ для ТСН:")
    print(f"  |Φ| mean = {np.mean(np.abs(valid_phi)):.2e}")
    print(f"  |Φ| max  = {np.max(np.abs(valid_phi)):.2e}")
    print(f"  Φ ≈ 0 (|Φ| < 1e-6): {np.sum(np.abs(valid_phi) < 1e-6)}/{len(valid_phi)}")


# ======================================================================
# 3. ПОИСК НАЧАЛЬНОЙ ТОЧКИ
# ======================================================================

print("\n===== Поиск начальной точки =====")

from helpers.inverse_winding_robust import find_valid_initial_point

u0, v0, Phi0, found = find_valid_initial_point(
    E2, traj_tsn,
    z_start=0.0,
    num_attempts=20
)

if not found:
    print("WARNING: Не удалось найти точку с Φ ≈ 0!")
    # Используем безопасную начальную точку
    R0 = traj_tsn.R(0.0)
    u0, v0 = safe_initial_point(E2, R0)
    print(f"Использована безопасная точка: u={u0:.4f}, v={v0:.4f}")


# ======================================================================
# 4. ОБРАТНАЯ ЗАДАЧА (ИСПРАВЛЕННЫЕ ПАРАМЕТРЫ)
# ======================================================================

print("\n===== Обратная задача (исправленные параметры) =====")

result = inverse_winding_robust(
    E2, traj_tsn, u0, v0,
    count_points=4000,          # 100 точек для теста
    eps_Phi=1e-6,             # ослабленный критерий
    max_newton=20,            # ИСПРАВЛЕНО: было 5
    max_bisect=5,
    jump_threshold=5.0,       # ИСПРАВЛЕНО: было 1.0
    verbose=True
)


# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ======================================================================

print("\n===== Визуализация =====")

fig = go.Figure()

# --- Поверхность E2 ---
u_grid = np.linspace(E2.u_min, E2.u_max, 50)
v_grid = np.linspace(0, 2*np.pi, 30)
Um, Vm = np.meshgrid(u_grid, v_grid)
X2 = np.zeros_like(Um)
Y2 = np.zeros_like(Um)
Z2 = np.zeros_like(Um)
for i in range(Um.shape[0]):
    for j in range(Um.shape[1]):
        p = E2.position(Um[i,j], Vm[i,j])
        X2[i,j], Y2[i,j], Z2[i,j] = p

fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4,
                         colorscale='Reds', showscale=False, name='Оправка E2'))

# --- ТСН ---
fig.add_trace(go.Scatter3d(
    x=points_tsn[:,0], y=points_tsn[:,1], z=points_tsn[:,2],
    mode='lines', line=dict(color='blue', width=4),
    name='ТСН'
))

# --- Восстановленная ЛУ ---
line_E2 = result['points_3d']
fig.add_trace(go.Scatter3d(
    x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
    mode='lines+markers', line=dict(color='green', width=3),
    marker=dict(size=3), name='Восстановленная ЛУ'
))

# --- Эталон ---
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(
        x=r_etalon[:,0], y=r_etalon[:,1], z=r_etalon[:,2],
        mode='lines', line=dict(color='orange', width=2, dash='dot'),
        name='Эталонная ЛУ'
    ))

fig.update_layout(
    title='Обратная задача: восстановление ЛУ (исправленные параметры)',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
               aspectmode='data'),
    width=1200, height=900
)
fig.write_html('inverse_winding_test.html')
print("График сохранён: inverse_winding_test.html")


# ======================================================================
# 6. ДИАГНОСТИЧЕСКИЕ ГРАФИКИ
# ======================================================================

fig_d, axes = plt.subplots(2, 2, figsize=(12, 10))

# Φ(z)
ax = axes[0, 0]
z_eval = result['z_eval']
Phi = result['Phi']
valid = np.abs(Phi) < 1e10
ax.semilogy(z_eval[valid], np.abs(Phi[valid]) + 1e-16, 'b.-')
ax.set_xlabel('z')
ax.set_ylabel('|Φ|')
ax.set_title('Невязка связи')
ax.grid(True, alpha=0.3)

# Итерации Ньютона
ax = axes[0, 1]
ax.plot(z_eval[1:], result['newton_iters'][1:], 'r.-')
ax.set_xlabel('z')
ax.set_ylabel('Итерации Ньютона')
ax.set_title('Итерации корректора')
ax.grid(True, alpha=0.3)

# κ_n(z)
ax = axes[1, 0]
ax.plot(z_eval, result['kappa_n'], 'g.-')
ax.set_xlabel('z')
ax.set_ylabel('κ_n')
ax.set_title('Нормальная кривизна')
ax.grid(True, alpha=0.3)

# u(z), v(z)
ax = axes[1, 1]
ax.plot(z_eval, result['u'], 'b-', label='u(z)')
ax.plot(z_eval, result['v'], 'r-', label='v(z)')
ax.set_xlabel('z')
ax.set_ylabel('Координаты')
ax.set_title('Криволинейные координаты')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics_test.png', dpi=150)
plt.show()
print("Диагностика сохранена: diagnostics_test.png")


# ======================================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ======================================================================

df_result = pd.DataFrame({
    'z': z_eval,
    'u': result['u'],
    'v': result['v'],
    'Phi': result['Phi'],
    'kappa_n': result['kappa_n'],
    'newton_iters': result['newton_iters'],
    'flag': result['flags']
})
df_result.to_csv('inverse_winding_result.csv', index=False)
print("Результат сохранён: inverse_winding_result.csv")


# ======================================================================
# 8. ИТОГОВАЯ СТАТИСТИКА
# ======================================================================

print("\n" + "="*50)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*50)
print(f"Всего шагов: {len(z_eval) - 1}")
print(f"Успешных (flag=0): {np.sum(result['flags'] == 0)}")
print(f"С бисекцией (flag=1): {np.sum(result['flags'] == 1)}")
print(f"Макс |Φ|: {np.max(np.abs(Phi)): .2e}")
print(f"Сред |Φ|: {np.mean(np.abs(Phi)): .2e}")
print(f"Макс итераций Ньютона: {np.max(result['newton_iters'][1:])}")
print(f"Сред итераций Ньютона: {np.mean(result['newton_iters'][1:]): .1f}")