# clients/client_collocation_ballon_scaled.py
"""
Клиент коллокации с масштабированием, автобалансировкой весов
и каскадной сеткой (mesh refinement).
"""
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy.io
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from core.trajectory import Trajectory
from helpers.inverse_collocation_scaled_snapshot import solve_collocation_scaled
from helpers.inverse_method_fixed import newton_corrector


# ---------- 1. Поверхность E2 ----------
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                 -0.0099656628535, 2.9503573330764]
R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
               39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka,
                                          bound_opravka, cyl_r_opravka)
print(f"E2: u∈[{E2.u_min}, {E2.u_max}], R_cyl={cyl_r_opravka}")


# ---------- 2. ТСН ----------
df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True].copy()
points_tsn = df_valid[['X', 'Y', 'Z']].values
print(f"ТСН: {len(points_tsn)} валидных точек")

traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"Траектория: длина = {traj.total_length:.2f}")


# ---------- 3. Начальная точка (перебор v) ----------
R0 = traj.R(0.0)

class DummyTraj:
    def __init__(self, R_fix):
        self._R = np.asarray(R_fix, dtype=float)
        self.total_length = 1.0
    def R(self, z):
        return self._R
    def R_deriv(self, z):
        return np.zeros(3)

best_u, best_v, best_Phi = None, None, np.inf
for v_try in np.linspace(0, 2 * np.pi, 36):
    u_try = float(np.clip(R0[2], E2.u_min, E2.u_max))
    dummy = DummyTraj(R0)
    try:
        u_c, v_c, Phi_c, _, conv = newton_corrector(
            E2, dummy, u_try, v_try, 0.0,
            eps_Phi=1e-6, max_iter=50
        )
        if abs(Phi_c) < best_Phi:
            best_Phi = abs(Phi_c)
            best_u, best_v = u_c, v_c
    except Exception:
        pass

if best_Phi < 1e-3:
    u0, v0 = best_u, best_v
else:
    u0, v0 = safe_initial_point(E2, R0)
    print(f"WARNING: newton не сошёлся, используем safe_initial_point")

print(f"Старт: u={u0:.3f}, v={v0:.4f}, Φ={best_Phi:.2e}")


# ---------- 4. Каскадная коллокация ----------
N_levels = [25]
result_prev = None

for idx, N in enumerate(N_levels):
    print(f"\n{'='*60}")
    print(f"Каскад {idx+1}/{len(N_levels)}: N = {N} точек")
    print(f"{'='*60}")

    if result_prev is not None:
        # Интерполируем предыдущее решение на новую сетку
        z_prev = result_prev['z_eval']
        u_prev = result_prev['u']
        v_prev = result_prev['v']

        z_new = np.linspace(0, traj.total_length, N)
        # unwrap v перед интерполяцией, иначе скачки 2π сломают сплайн
        v_prev_unwrapped = np.unwrap(v_prev)

        u_interp = np.interp(z_new, z_prev, u_prev)
        v_interp = np.interp(z_new, z_prev, v_prev_unwrapped)

        X0_custom = np.zeros(2 * (N - 1))
        X0_custom[0::2] = u_interp[1:]
        X0_custom[1::2] = v_interp[1:]
        init_method = 'custom'
    else:
        X0_custom = None
        init_method = 'radial'

    # result = solve_collocation_scaled(
    #     E2, traj, u0, v0,
    #     count_points=N,
    #     w_Phi=1.0, w_diff=1.0, w_smooth=0.05,   # небольшая гладкость подавляет зигзаги
    #     init_method=init_method,
    #     X0_custom=X0_custom,
    #     max_nfev=15000,
    #     tol=1e-8,
    #     verbose=True
    # )
    result = solve_collocation_scaled(
    E2, traj, u0, v0,
    count_points=200,
    w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
    init_method='radial',
    max_nfev=50000,
    snapshot_file='balloon_best.npz',  # <-- файл снапшота
    snapshot_interval=1,                # сохранять при каждом улучшении
    restore_from_snapshot=True,         # если есть старый снапшот — загрузить
    verbose=True
)   
    result_prev = result

# ---------- Финальная статистика ----------
print(f"\n{'='*60}")
print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
print(f"{'='*60}")
print(f"|F|_opt   = {result['res_norm']:.3e}")
print(f"Макс |Φ|  = {np.max(np.abs(result['Phi'])):.2e}")
print(f"Сред |Φ|  = {np.mean(np.abs(result['Phi'])):.2e}")
print(f"nfev      = {result['nfev']}")
print(f"success   = {result['success']}")
print(f"Масштабы: u={result['scale_u']:.2f}, v={result['scale_v']:.4f}")
print(f"Веса: w_Phi={result['weights']['w_Phi']:.3e}, "
      f"w_diff={result['weights']['w_diff']:.3e}, "
      f"w_smooth={result['weights']['w_smooth']:.3e}")


# ---------- 5. Эталонная линия укладки ----------
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']
    print(f"Эталон: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    r_etalon = None


# ---------- 6. Визуализация ----------
line_E2 = result['points_3d']
tsn_pts = np.array([traj.R(z) for z in result['z_eval']])

fig = go.Figure()

# Поверхность E2
u_grid = np.linspace(E2.u_min, E2.u_max, 50)
v_grid = np.linspace(0, 2 * np.pi, 40)
Um, Vm = np.meshgrid(u_grid, v_grid)
X2 = np.zeros_like(Um)
Y2 = np.zeros_like(Um)
Z2 = np.zeros_like(Um)
for i in range(Um.shape[0]):
    for j in range(Um.shape[1]):
        p = E2.position(Um[i, j], Vm[i, j])
        X2[i, j], Y2[i, j], Z2[i, j] = p
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds',
                         showscale=False, name='E2 (оправка)'))

# ТСН
fig.add_trace(go.Scatter3d(x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
                             mode='lines', line=dict(color='blue', width=4),
                             name='ТСН'))

# ЛУ (коллокация)
fig.add_trace(go.Scatter3d(x=line_E2[:, 0], y=line_E2[:, 1], z=line_E2[:, 2],
                             mode='lines+markers', line=dict(color='green', width=3),
                             marker=dict(size=3), name='ЛУ (scaled collocation)'))

# Эталон
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(x=r_etalon[:, 0], y=r_etalon[:, 1], z=r_etalon[:, 2],
                                 mode='lines', line=dict(color='orange', width=2, dash='dot'),
                                 name='Эталон'))

fig.update_layout(
    title='Коллокация с масштабированием: баллон (каскад N=25→200)',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=1200, height=900
)
fig.write_html('collocation_balloon_scaled.html')
print("\nГрафик: collocation_balloon_scaled.html")