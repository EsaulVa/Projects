# clients/client_collocation_jax.py
"""
Клиент JAX-коллокации с snapshot-механизмом.
Запускает решение, сохраняет лучший промежуточный результат,
и может визуализировать как финальный результат, так и snapshot.
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
from helpers.inverse_collocation_jax import solve_collocation_jax, load_snapshot
from helpers.inverse_method_fixed import newton_corrector


# ---------- 1. Поверхность ----------
phi_c = [0.0000000005642, -0.0000003012748, 0.0000605882383,
         -0.0099656628535, 2.9503573330764]
R_c = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
       39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bounds = [0, 234.27, 534.27, 768.54]
cyl_r = 251.705
E2 = FixedPiecewisePolynomialRevolution(phi_c, R_c, bounds, cyl_r)
print(f"E2: u∈[{E2.u_min}, {E2.u_max}]")


# ---------- 2. ТСН ----------
df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True].copy()
points_tsn = df_valid[['X', 'Y', 'Z']].values
traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"ТСН: {len(points_tsn)} точек, длина={traj.total_length:.2f}")


# ---------- 3. Начальная точка ----------
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
for v_try in np.linspace(0, 2*np.pi, 36):
    u_try = float(np.clip(R0[2], E2.u_min, E2.u_max))
    try:
        u_c, v_c, Phi_c, _, conv = newton_corrector(
            E2, DummyTraj(R0), u_try, v_try, 0.0, eps_Phi=1e-6, max_iter=50
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
print(f"Старт: u={u0:.3f}, v={v0:.4f}, Φ={best_Phi:.2e}")


# ---------- 4. JAX коллокация ----------
print(f"\n{'='*60}")
print("JAX COLLOCATION")
print(f"{'='*60}")

result = solve_collocation_jax(
    E2, traj, u0, v0,
    count_points=50,          # начнём с 50 для скорости
    w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
    init_method='radial',
    jac_mode='3-point',         # 'jax' для exact (медленнее), '3-point' для скорости
    max_nfev=20000,
    tol=1e-8,
    snapshot_path='best_snapshot.npz',
    verbose=True
)

print(f"\n>>> |F|={result['res_norm']:.3e}")
print(f">>> Макс |Φ| = {np.max(np.abs(result['Phi'])):.2e}")
print(f">>> Сред |Φ| = {np.mean(np.abs(result['Phi'])):.2e}")
print(f">>> Время = {result['time']:.1f}s")
print(f">>> Snapshot сохранён: {result['snapshot_path']}")


# ---------- 5. Визуализация финального результата ----------
line_E2 = result['points_3d']
tsn_pts = np.array([traj.R(z) for z in result['z_eval']])

fig = go.Figure()

u_grid = np.linspace(E2.u_min, E2.u_max, 40)
v_grid = np.linspace(0, 2*np.pi, 40)
Um, Vm = np.meshgrid(u_grid, v_grid)
X2 = np.zeros_like(Um); Y2 = np.zeros_like(Um); Z2 = np.zeros_like(Um)
for i in range(Um.shape[0]):
    for j in range(Um.shape[1]):
        p = E2.position(Um[i,j], Vm[i,j])
        X2[i,j], Y2[i,j], Z2[i,j] = p

fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds',
                         showscale=False, name='E2'))
fig.add_trace(go.Scatter3d(x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
                             mode='lines', line=dict(color='blue', width=4), name='ТСН'))
fig.add_trace(go.Scatter3d(x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
                             mode='lines+markers', line=dict(color='green', width=3),
                             marker=dict(size=3), name='ЛУ (JAX final)'))

fig.update_layout(title='JAX Collocation: Final Result', scene_aspectmode='data',
                  width=1200, height=900)
fig.write_html('collocation_jax_final.html')
print("\nСохранено: collocation_jax_final.html")


# ---------- 6. ВИЗУАЛИЗАЦИЯ SNAPSHOT (опционально) ----------
"""
Раскомментируйте этот блок, если хотите посмотреть,
как выглядело лучшее промежуточное решение.
"""

# snap = load_snapshot('best_snapshot.npz')
# u_min = E2.u_min
# scale_u = result['scale_u']
# scale_v = result['scale_v']
# X_tail = snap['X_tail']
# u_snap = np.concatenate([[snap['u0']], u_min + X_tail[0::2] * scale_u])
# v_snap = np.concatenate([[snap['v0']], X_tail[1::2] * scale_v])
# pts_snap = np.array([E2.position(u_snap[k], v_snap[k]) for k in range(snap['N'])])

# fig2 = go.Figure()
# fig2.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.3, colorscale='Reds', showscale=False))
# fig2.add_trace(go.Scatter3d(x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
#                               mode='lines', line=dict(color='blue', width=4)))
# fig2.add_trace(go.Scatter3d(x=pts_snap[:,0], y=pts_snap[:,1], z=pts_snap[:,2],
#                               mode='lines+markers', line=dict(color='orange', width=3),
#                               marker=dict(size=3), name='ЛУ (SNAPSHOT)'))
# fig2.update_layout(title=f'JAX Snapshot (iter {snap["iter"]}, max|Phi|={snap["max_Phi"]:.2e})',
#                    scene_aspectmode='data', width=1200, height=900)
# fig2.write_html('collocation_jax_snapshot.html')
# print("Сохранено: collocation_jax_snapshot.html")