"""
Клиент обратной задачи с ГИБРИДНЫМ предиктором и ПРАВИЛЬНОЙ параметризацией.
Ключевое исправление: z = s_LU (длина дуги эталонной линии укладки).
"""
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import sys
from pathlib import Path
from scipy.interpolate import CubicSpline

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from helpers.inverse_winding_robust import (
    inverse_winding_robust,
    newton_corrector_stable,
    # find_valid_initial_point
)
# Импорт для гибридного предиктора
from helpers.optical_predictor import OpticalPredictor
from helpers.intersection import RayTracer
from helpers.fixed_intersections import FixedRobustRevolutionIntersection

# ======================================================================
# 1. КЛАСС TRAJECTORY WITH S (САМОДОСТАТОЧНЫЙ)
# ======================================================================
class TrajectoryWithS:
    def __init__(self, s_array, points_array):
        sort_idx = np.argsort(s_array)
        self._s = s_array[sort_idx]
        self._points = points_array[sort_idx]
        self._total_length = float(self._s[-1] - self._s[0])
        
        # Natural bc_type обеспечивает гладкость 2-й производной
        self._cs_x = CubicSpline(self._s, self._points[:, 0], bc_type='natural')
        self._cs_y = CubicSpline(self._s, self._points[:, 1], bc_type='natural')
        self._cs_z = CubicSpline(self._s, self._points[:, 2], bc_type='natural')

    @property
    def total_length(self) -> float:
        return self._total_length

    def R(self, z):
        z_clamped = np.clip(z, self._s[0], self._s[-1])
        return np.array([self._cs_x(z_clamped), self._cs_y(z_clamped), self._cs_z(z_clamped)])

    def R_deriv(self, z):
        z_clamped = np.clip(z, self._s[0], self._s[-1])
        return np.array([self._cs_x(z_clamped, 1), self._cs_y(z_clamped, 1), self._cs_z(z_clamped, 1)])

# ======================================================================
# 2. ЗАГРУЗКА И СИНХРОНИЗАЦИЯ
# ======================================================================
print("===== Загрузка данных =====")

# ТСН
df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True].copy()
points_xyz = df_valid[['X', 'Y', 'Z']].values
original_indices = df_valid.index.values
print(f"ТСН: {len(points_xyz)} валидных точек")

# Эталонная ЛУ для параметра s
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']
    
    # Вычисляем длину дуги эталонной ЛУ
    diffs = np.diff(r_etalon, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    s_lu_full = np.zeros(len(r_etalon))
    s_lu_full[1:] = np.cumsum(dists)
    
    print(f"Эталонная ЛУ: {len(r_etalon)} точек")
    
    # Синхронизация: берем s_lu по индексам валидных точек ТСН
    if np.max(original_indices) >= len(s_lu_full):
        raise ValueError(f"Индекс {np.max(original_indices)} выходит за пределы эталона ({len(s_lu_full)})")
        
    s_values_for_tsn = s_lu_full[original_indices]
    print("Параметр s успешно синхронизирован с эталонной ЛУ.")
    
except Exception as e:
    print(f"Warning: {e}. Используем длину дуги самой ТСН (менее точно).")
    diffs_tsn = np.diff(points_xyz, axis=0)
    dists_tsn = np.sqrt(np.sum(diffs_tsn**2, axis=1))
    s_values_for_tsn = np.zeros(len(points_xyz))
    s_values_for_tsn[1:] = np.cumsum(dists_tsn)

if len(s_values_for_tsn) != len(points_xyz):
    raise ValueError("Критическая ошибка: длины массивов s и points не совпадают!")

traj_tsn = TrajectoryWithS(s_values_for_tsn, points_xyz)
print(f"Траектория создана: total_length = {traj_tsn.total_length:.2f}")

# ======================================================================
# 3. ПОВЕРХНОСТЬ И ГИБРИДНЫЙ ПРЕДИКТОР
# ======================================================================
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705
E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

# Настройка оптического предиктора
tracer = RayTracer()
tracer.register(type(E2), FixedRobustRevolutionIntersection())
opt_pred = OpticalPredictor(tracer)

# ======================================================================
# 4. ЗАПУСК
# ======================================================================
print("\n===== Поиск начальной точки =====")
# u0, v0, Phi0, found = find_valid_initial_point(E2, traj_tsn, z_start=s_values_for_tsn[0])
found=False
if not found:
    R0 = traj_tsn.R(s_values_for_tsn[0])
    u0, v0 = safe_initial_point(E2, R0)
    u0, v0, Phi0, _, _ = newton_corrector_stable(E2, traj_tsn, u0, v0, s_values_for_tsn[0])
print(f"Старт: u={u0:.4f}, v={v0:.4f}, Φ={Phi0:.2e}")

print("\n===== Расчет (Гибрид) =====")
result = inverse_winding_robust(
    E2, traj_tsn, u0, v0,
    optical_predictor=opt_pred,  # <-- ВКЛЮЧЕНИЕ ГИБРИДА
    count_points=3000,
    eps_Phi=1e-6,
    max_newton=20,
    max_bisect=8,
    jump_threshold=5.0,
    grad_threshold=1e-4,
    verbose=True
)

# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ======================================================================
fig = go.Figure()
# Поверхность
u_grid = np.linspace(E2.u_min, E2.u_max, 50)
v_grid = np.linspace(0, 2*np.pi, 30)
Um, Vm = np.meshgrid(u_grid, v_grid)
X2, Y2, Z2 = np.zeros_like(Um), np.zeros_like(Um), np.zeros_like(Um)
for i in range(Um.shape[0]):
    for j in range(Um.shape[1]):
        p = E2.position(Um[i,j], Vm[i,j])
        X2[i,j], Y2[i,j], Z2[i,j] = p
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds', showscale=False))

# Траектории
fig.add_trace(go.Scatter3d(x=points_xyz[:,0], y=points_xyz[:,1], z=points_xyz[:,2],
                           mode='lines', line=dict(color='blue', width=4), name='ТСН'))
line_E2 = result['points_3d']
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                           mode='lines+markers', line=dict(color='green', width=3), name='ЛУ'))

# Эталон
try:
    fig.add_trace(go.Scatter3d(x=r_etalon[:,0], y=r_etalon[:,1], z=r_etalon[:,2],
                               mode='lines', line=dict(color='orange', width=2, dash='dot'), name='Эталон'))
except: pass

fig.update_layout(title='Обратная задача: Гибрид + Правильная параметризация', scene_aspectmode='data')
fig.write_html('inverse_hybrid_final.html')

# Сохранение
pd.DataFrame({
    'z': result['z_eval'], 'u': result['u'], 'v': result['v'],
    'Phi': result['Phi'], 'kappa_n': result['kappa_n'],
    'iters': result['newton_iters'], 'flag': result['flags'],
    'pred_type': result['predictor_type']
}).to_csv('inverse_winding_hybrid_final.csv', index=False)

print("\nГотово! Проверьте inverse_winding_hybrid_final.csv")