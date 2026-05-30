import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io
import sys
from pathlib import Path

# Добавляем корневую директорию проекта (родительскую по отношению к папке gui)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from geometry.piecewise_polynomial_revolution_fixed_v2 import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method_fixed import newton_corrector
from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor, RayTracer
from helpers.intersection import RevolutionIntersection,RobustRevolutionIntersection
from geometry.tsurfaces import FixedPointTrajectory

import sys
from pathlib import Path

# Добавляем корневую директорию проекта (родительскую по отношению к папке gui)
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# ---------- Коэффициенты из surface_r.m (оправка) ----------
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392,
                 -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

# E2 = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)
# пишем:
from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)


# ---------- Коэффициенты из surface_r_b.m (безопасность) ----------
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366,
              -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387

# E1 = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)
E1 = FixedPiecewisePolynomialRevolution(phi_c_safe,   R_c_safe,   bound_safe,   cyl_r_safe)

# 1. Загружаем ТСН из .mat
data = scipy.io.loadmat('winding_trajectory_result.mat')
z_offset = (bound_safe[3] - bound_opravka[3]) / 2  # (955.956 - 768.54)/2 ≈ 93.708
X, Y, Z = data['X_tsn'].flatten(), data['Y_tsn'].flatten(), data['Z_tsn'].flatten()
Z_local = Z - z_offset
points_tsn = np.column_stack([X, Y, Z_local])
count_points = len(X)



print(f"z_offset = {z_offset:.3f} мм")
# points_tsn = points_tsn[::-1].copy()
# traj = Trajectory.from_points(points_tsn, method='cubic')
traj = Trajectory.from_points(points_tsn, method='cubic')  # cubic для стабильности
# traj = Trajectory.from_points(points_tsn, method='nurbs', degree=7)
print(f"Z начала: {points_tsn[0,2]:.3f}, Z конца: {points_tsn[-1,2]:.3f}")
print(f"R'(0) = {traj.R_deriv(0.0)}")
if points_tsn[0, 2] > points_tsn[-1, 2]:
    points_tsn = points_tsn[::-1].copy()
    print("Инвертировано: ТСН шла сверху вниз")
# Загружаем эталонную линию укладки из LU_data.mat
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']  # массив 545x3
    print(f"Эталонная линия укладки загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    print("Файл LU_data.mat не найден – эталонная линия не будет показана.")
    r_etalon = None

# if r_etalon is not None:
#     u0 = r_etalon[0, 2]
#     v0 = np.arctan2(r_etalon[0, 1], r_etalon[0, 0])
# else:
#     R0 = traj.R(0.0)
#     u0, v0 = safe_initial_point(E2, R0)
# R0 = traj.R(0.0)
# u_guess, v_guess = safe_initial_point(E2, R0)
# # или точнее: u_guess = R0[2]; v_guess = np.arctan2(R0[1], R0[0])

# dummy = FixedPointTrajectory(R0)
# u0, v0, Phi0, _, conv = newton_corrector(
#     E2, dummy, u_guess, v_guess, 0.0,
#     eps_Phi=1e-12, max_iter=50
# )
if r_etalon is not None:
    u0 = r_etalon[0, 2]
    v0 = np.arctan2(r_etalon[0, 1], r_etalon[0, 0])
    # Небольшая коррекция на случай, если эталон не идеален
    u0, v0, Phi0, _, conv = newton_corrector(
        E2, traj, u0, v0, 0.0, eps_Phi=1e-10, max_iter=20
    )
    print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}, conv={conv}")
else:
    R0 = traj.R(0.0)
    u0, v0 = safe_initial_point(E2, R0)
# print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}, conv={conv}")

# # Коррекция начальной точки до касания Φ = 0
# r0 = E2.position(u0, v0)
# R0 = traj.R(0.0)
# m0 = E2.normal(u0, v0)
# Phi0 = np.dot(R0 - r0, m0)
# if abs(Phi0) > 1e-8:
#     print("Корректировка начальной точки...")
#     u0, v0, Phi0_corr, _, conv0 = newton_corrector(
#         E2, traj, u0, v0, 0.0, eps_Phi=1e-12, max_iter=2000
#     )
#     print(f"После коррекции: Φ₀ = {Phi0_corr:.6e}, сошёлся: {conv0}")

# # --- Диагностика нормали и корректора на одной точке ---
# z_test = 0.0
# R_test = traj.R(z_test)
# u_test, v_test = u0, v0

# # Проверим нормаль
# r_test = E2.position(u_test, v_test)
# m_test = E2.normal(u_test, v_test)
# Phi_test = np.dot(R_test - r_test, m_test)
# print(f"Φ на старте: {Phi_test:.4f}")
# print(f"Нормаль: {m_test}")
# print(f"R-r (направление нити): {R_test - r_test}")

# # Запустим корректор вручную
# u_corr, v_corr, Phi_corr, nit, conv = newton_corrector(
#     E2, traj, u_test, v_test, z_test, eps_Phi=1e-10, max_iter=20
# )
# print(f"Корректор: сошёлся={conv}, итераций={nit}, Φ={Phi_corr:.2e}")
# print(f"Исправленная точка: u={u_corr:.3f}, v={v_corr:.3f}")

# from helpers.inverse_method import compute_dr_dz

# # --- Тест 1: Проверка compute_dr_dz в стартовой точке ---
# z0 = 0.0
dz = traj.total_length / 300  # шаг, соответствующий count_points=300
# du, dv = compute_dr_dz(E2, traj, u0, v0, z0)

# u1_euler = u0 + du * dz
# v1_euler = v0 + dv * dz

# r1 = E2.position(u1_euler, v1_euler)
# R1 = traj.R(dz)
# m1 = E2.normal(u1_euler, v1_euler)
# Phi_euler = np.dot(R1 - r1, m1)

# print(f"\n=== Тест предиктора (явный Эйлер) ===")
# print(f"du/dz={du:.6f}, dv/dz={dv:.6f}")
# print(f"u1_euler={u1_euler:.3f}, v1_euler={v1_euler:.3f}")
# print(f"Φ после шага Эйлера: {Phi_euler:.4e}")

# # --- Тест 2: Корректор на предсказании ---
# u1_c, v1_c, Phi_c, nit_c, conv_c = newton_corrector(
#     E2, traj, u1_euler, v1_euler, dz, eps_Phi=1e-10, max_iter=50
# )
# print(f"Корректор: сошёлся={conv_c}, итераций={nit_c}, Φ={Phi_c:.4e}")
# print(f"Исправлено: u={u1_c:.3f}, v={v1_c:.3f}")

# # --- Численное du/dz из эталонной линии укладки ---
# u_etalon = r_etalon[:, 2]  # z = u для поверхности вращения
# v_etalon = np.arctan2(r_etalon[:, 1], r_etalon[:, 0])
# s_etalon = np.cumsum([0] + [np.linalg.norm(r_etalon[i+1]-r_etalon[i]) 
#                               for i in range(len(r_etalon)-1)])
# # Производные в начале
# du_ds_etalon = (u_etalon[1] - u_etalon[0]) / (s_etalon[1] - s_etalon[0])
# dv_ds_etalon = np.mod(v_etalon[1] - v_etalon[0] + np.pi, 2*np.pi) - np.pi
# dv_ds_etalon /= (s_etalon[1] - s_etalon[0])
# print(f"Эталон: du/ds={du_ds_etalon:.6f}, dv/ds={dv_ds_etalon:.6f}")

# Сравним с аналитическим compute_dr_dz
# (нужно знать ds/dz — скорость точки укладки относительно параметра ТСН)
# Приближённо: ds/dz ≈ |dr/dz| = 1 (если z — длина дуги ТСН)
# print(f"Аналит: du/dz={du:.6f}, dv/dz={dv:.6f}")
# Настройка предикторов
solver_dae = SciPySolver(method='BDF', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)
# ray_tracer = RayTracer()
# # ray_tracer.register(PiecewisePolynomialRevolution, RevolutionIntersection())
# ray_tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
# from helpers.fixed_intersections import FixedPiecewisePolynomialIntersection, FixedRobustRevolutionIntersection

# ray_tracer = RayTracer()
# ray_tracer.register(FixedPiecewisePolynomialRevolution, FixedRobustRevolutionIntersection())
# # или:
# # ray_tracer.register(PiecewisePolynomialRevolution, FixedPiecewisePolynomialIntersection())
# optical_predictor = OpticalPredictor(ray_tracer)
# --- Тест 3: Предиктор DAE (интегратор) ---
# pred = dae_predictor.predict(0.0, dz, u0, v0, E2, traj)
# if pred:
#     u1_dae, v1_dae = pred
#     r1_dae = E2.position(u1_dae, v1_dae)
#     m1_dae = E2.normal(u1_dae, v1_dae)
#     Phi_dae_pred = np.dot(R1 - r1_dae, m1_dae)
#     print(f"DAE-предиктор: u={u1_dae:.3f}, v={v1_dae:.3f}, Φ предсказания={Phi_dae_pred:.4e}")
    
#     u1_dae_c, v1_dae_c, Phi_dae_c, nit_d, conv_d = newton_corrector(
#         E2, traj, u1_dae, v1_dae, dz, eps_Phi=1e-10, max_iter=50
#     )
#     print(f"DAE+корректор: сошёлся={conv_d}, итераций={nit_d}, Φ={Phi_dae_c:.4e}")
# else:
#     print("DAE-предиктор вернул None!")

print("\n===== Обратная задача: восстановление линии укладки на E2 (гибрид) =====")



# # ray_tracer = RayTracer()
# # # ray_tracer.register(PiecewisePolynomialRevolution, RevolutionIntersection())
# # ray_tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())
# from helpers.fixed_intersections import FixedPiecewisePolynomialIntersection, FixedRobustRevolutionIntersection

# ray_tracer = RayTracer()
# ray_tracer.register(FixedPiecewisePolynomialRevolution, FixedRobustRevolutionIntersection())
# # или:
# # ray_tracer.register(PiecewisePolynomialRevolution, FixedPiecewisePolynomialIntersection())
# optical_predictor = OpticalPredictor(ray_tracer)

# Гибридный решатель
# result = inverse_winding_hybrid(
#     E2, traj, u0, v0,
#     count_points=300,
#     eps_Phi=1e-10, max_newton=7, max_bisect=10, jump_threshold=3.0,
#     predictor_dae=dae_predictor,
#     predictor_optical=optical_predictor,
#     eps_kappa=1e-2,
#     u_margin=0.05,
#     force_optical_after_fail=True
# )
from helpers.fixed_intersections import FixedRobustRevolutionIntersection

ray_tracer = RayTracer()
ray_tracer.register(FixedPiecewisePolynomialRevolution, FixedRobustRevolutionIntersection())
optical_predictor = OpticalPredictor(ray_tracer)

# result = inverse_winding_hybrid(
#     E2, traj, u0, v0,
#     count_points=200,    # достаточно, чтобы увидеть проблему
#     max_newton=20,
#     eps_Phi=1e-10,
#     # остальное как есть
#     max_bisect=4,
#     jump_threshold=3.0,
#     predictor_dae=dae_predictor,
#     predictor_optical=optical_predictor,
#     eps_kappa=1e-2,      # ← включить кривизну
#     u_margin=20.0,       # ← сузить оптику до реального днища
#     force_optical_after_fail=False
# )
# ============================================
# ТЕСТОВЫЙ БЛОК: проверка знака градиента
# ============================================
from helpers.inverse_method_fixed import *
print("\n" + "="*60)
print("ЧИСЛЕННАЯ ПРОВЕРКА ЗНАКА ГРАДИЕНТА (тестовая точка из лога)")
print("="*60)

# Точка из вашего лога, где Phi взрывается (шаг ~255, z≈243)
# Если хотите проверить на старте — замените u_test/v_test на u0, v0, а z_test на 0.0
z_test  = 243.03
u_test  = 36.088
v_test  = 1.6256

# --- Численный градиент (конечные разности) ---
r0   = E2.position(u_test, v_test)
m0   = E2.normal(u_test, v_test)
R_t  = traj.R(z_test)
Phi0 = float(np.dot(R_t - r0, m0))
print(f"Точка: z={z_test}, u={u_test}, v={v_test}")
print(f"Phi0 = {Phi0:.4e}")

eps = 1e-5
r_u   = E2.position(u_test + eps, v_test)
m_u   = E2.normal(u_test + eps, v_test)
Phi_u = float(np.dot(R_t - r_u, m_u))

r_v   = E2.position(u_test, v_test + eps)
m_v   = E2.normal(u_test, v_test + eps)
Phi_v = float(np.dot(R_t - r_v, m_v))

dPhi_du_num = (Phi_u - Phi0) / eps
dPhi_dv_num = (Phi_v - Phi0) / eps
print(f"\nЧИСЛЕННЫЙ:  dPhi/du = {dPhi_du_num: .4e}")
print(f"ЧИСЛЕННЫЙ:  dPhi/dv = {dPhi_dv_num: .4e}")

# --- Аналитический градиент ---
delta = R_t - r0
lam   = np.linalg.norm(delta)
tau   = delta / lam
up, vp = compute_tangent_components(E2, u_test, v_test, tau)
dPhidu_ana, dPhidv_ana = compute_grad_Phi(E2, u_test, v_test, up, vp, lam)

print(f"\nАНАЛИТИЧ.:  dPhi/du = {dPhidu_ana: .4e}")
print(f"АНАЛИТИЧ.:  dPhi/dv = {dPhidv_ana: .4e}")

# --- Сравнение ---
ratio_u = dPhi_du_num / dPhidu_ana if abs(dPhidu_ana) > 1e-14 else float('inf')
ratio_v = dPhi_dv_num / dPhidv_ana if abs(dPhidv_ana) > 1e-14 else float('inf')
print(f"\nОТНОШЕНИЕ:  du = {ratio_u: .2f}   dv = {ratio_v: .2f}")

if abs(ratio_u + 1.0) < 0.1 and abs(ratio_v + 1.0) < 0.1:
    print("\n>>> ВНИМАНИЕ: АНАЛИТИЧЕСКИЙ ГРАДИЕНТ ИНВЕРТИРОВАН ПО ЗНАКУ!")
    print(">>> Нужно убрать минус в compute_grad_Phi (inverse_method_fixed.py)")
    print(">>> dPhidu =  lam * (...)  вместо  -lam * (...)\n")
elif abs(ratio_u - 1.0) < 0.1 and abs(ratio_v - 1.0) < 0.1:
    print("\n>>> ЗНАК ГРАДИЕНТА ВЕРНЫЙ.")
    print(">>> Проблема в другом: вырожденность, max_bisect, или геометрия траектории.\n")
else:
    print(f"\n>>> НЕОДНОЗНАЧНО: отношения {ratio_u:.2f}, {ratio_v:.2f}")
    print(">>> Возможно, точка на границе сегмента или метрика вырождена.\n")

# --- Дополнительно: проверим, что max_bisect точно 4 ---
print(f"Параметры вызова inverse_winding_hybrid:")
print(f"  max_bisect = 4  (в коде явно задано)")
print("="*60 + "\n")
# --- Диагностика: расстояние от ТСН до оси Z ---
z_test = 243.03
R_t = traj.R(z_test)
dist_to_axis = np.hypot(R_t[0], R_t[1])
print(f"\n>>> Диагностика на z={z_test}:")
print(f"    ТСН: R = {R_t}")
print(f"    Расстояние ТСН до оси Z: {dist_to_axis:.3f}")
print(f"    Радиус оправки E2 в u={u_test}: {E2.radius(u_test):.3f}")
if dist_to_axis < E2.radius(u_test):
    print("    !!! ТСН ПРОХОДИТ ВНУТРИ ОПРАВКИ — решения нет!")
else:
    print("    ТСН снаружи — решение должно существовать.")
# ============================================
# --- Диагностика: куда ведет compute_dr_dz ---
from helpers.inverse_method_fixed import compute_dr_dz
z_test = 243.03
u_test = 36.088
v_test = 1.6256

du, dv = compute_dr_dz(E2, traj, u_test, v_test, z_test)
print(f"\n>>> compute_dr_dz на проблемной точке:")
print(f"    du/dz = {du:.6f}, dv/dz = {dv:.6f}")

# Шаг Эйлера вперёд
dz = 0.5
u1 = u_test + du * dz
v1 = v_test + dv * dz
r1 = E2.position(u1, v1)
m1 = E2.normal(u1, v1)
Phi1 = float(np.dot(traj.R(z_test + dz) - r1, m1))
print(f"    После шага dz={dz}: u={u1:.3f}, v={v1:.3f}, Phi={Phi1:.4e}")

# Шаг Эйлера назад (проверка симметрии)
u2 = u_test - du * dz
v2 = v_test - dv * dz
r2 = E2.position(u2, v2)
m2 = E2.normal(u2, v2)
Phi2 = float(np.dot(traj.R(z_test - dz) - r2, m2))
print(f"    Шаг назад dz=-{dz}: u={u2:.3f}, v={v2:.3f}, Phi={Phi2:.4e}")

print("=== Атрибуты поверхности E2 ===")
print(f"  u_min = {getattr(E2, 'u_min', 'НЕТ')}")
print(f"  u_max = {getattr(E2, 'u_max', 'НЕТ')}")
print(f"  v_min = {getattr(E2, 'v_min', 'НЕТ')}")
print(f"  v_max = {getattr(E2, 'v_max', 'НЕТ')}")
print(f"  bounds = {E2.bounds if hasattr(E2, 'bounds') else 'НЕТ'}")
print(f"  bound_opravka = {bound_opravka}")
print("================================")
result = inverse_winding_hybrid(
    E2, traj, u0, v0,
    count_points=5800,
    eps_Phi=1e-10,
    max_newton=20,
    max_bisect=4,
    jump_threshold=3.0,       # ← вернём строгость
    predictor_dae=dae_predictor,
    predictor_optical=optical_predictor,
    eps_kappa=1e-2,         # ← оптика включается при |κ_n| < 0.1 (почти плоские участки)
    u_margin=20.0,          # ← оптика включается у днища/крышки
    force_optical_after_fail=True  # ← после фейла пробуем DAE снова, не застреваем на оптике
)
# Извлечение результатов
z_vals = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
lam_hist = result['lam']
flags = result['flags']
line_E2 = result['points_3d']

n_bisected = np.sum(flags == 1)
print(f"Шагов с бисекцией: {n_bisected} из {len(z_vals)-1}")
print(f"Максимальная невязка |Φ| = {np.max(np.abs(Phi_hist)):.2e}")
print(f"Средняя невязка |Φ|      = {np.mean(np.abs(Phi_hist)):.2e}")
print(f"Среднее итераций Ньютона  = {np.mean(newton_iters_hist[1:]):.2f}")
print(f"Максимум итераций Ньютона = {np.max(newton_iters_hist[1:])}")
print(f"Минимальная невязка Phi: {np.min(result['Phi']):.4f}")
print(f"Максимальная невязка Phi: {np.max(result['Phi']):.4f}")
print(f"Средняя невязка Phi: {np.mean(np.abs(result['Phi'])):.4f}")

# Визуализация (базовый вариант)
fig = go.Figure()
v_grid = np.linspace(0, 2*np.pi, 80)              # азимут
u_grid_E2 = np.linspace(E2.u_min, E2.u_max, 60)  # высота (аксиальная)
X2 = np.zeros((len(v_grid), len(u_grid_E2)))
Y2, Z2 = np.zeros_like(X2), np.zeros_like(X2)
for i, v in enumerate(v_grid):
    for j, u in enumerate(u_grid_E2):
        p = E2.position(u, v)
        X2[i,j], Y2[i,j], Z2[i,j] = p
# # Сетки поверхностей
# u_grid = np.linspace(0, 2*np.pi, 80)
# v_grid_E2 = np.linspace(E2.v_min, E2.v_max, 60)
# X2, Y2, Z2 = np.zeros((len(v_grid_E2), len(u_grid))), np.zeros_like(X2), np.zeros_like(X2)
# for i, v in enumerate(v_grid_E2):
#     for j, u in enumerate(u_grid):
#         p = E2.position(u, v)
#         X2[i,j], Y2[i,j], Z2[i,j] = p
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds', showscale=False, name='Оправка'))

# Траектория R(z)
fig.add_trace(go.Scatter3d(x=points_tsn[:,0], y=points_tsn[:,1], z=points_tsn[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='ТСН'))

# Восстановленная линия укладки
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                           mode='lines', line=dict(color='red', width=4), name='Восстановленная ЛУ'))

# Эталонная линия
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(x=r_etalon[:,0], y=r_etalon[:,1], z=r_etalon[:,2],
                               mode='lines', line=dict(color='orange', width=2, dash='dot'),
                               name='Эталонная ЛУ'))

fig.update_layout(title='Гибридная обратная задача (полиномиальный баллон)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
                  width=1000, height=800)
fig.write_html('fnc_ballon_hybrid.html')
print("График сохранён в fnc_ballon_hybrid.html")

# Диагностические графики
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.semilogy(z_vals[1:], np.abs(Phi_hist[1:])+1e-16)
plt.title('Невязка |Φ|')
plt.subplot(2,2,2)
plt.plot(z_vals[1:], newton_iters_hist[1:], '.')
plt.title('Итерации Ньютона')
plt.subplot(2,2,3)
plt.plot(z_vals, kappa_n_hist)
plt.title('κ_n')
plt.subplot(2,2,4)
plt.plot(z_vals, u_hist, label='u(z)')
plt.plot(z_vals, v_hist, label='v(z)')
plt.legend()
plt.tight_layout()
plt.show()