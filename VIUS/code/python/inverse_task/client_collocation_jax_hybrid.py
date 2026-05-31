# clients/client_collocation_jax_hybrid.py
"""
Гибридная адаптивная коллокация: каскад N=25 -> 50 -> 100 -> 200.
На каждом уровне проверяется max|Phi|. Если < tresh — остановка.
Иначе интерполяция на следующий уровень (warm start).
"""
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy.io
import sys
import time
import json
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from core.trajectory import Trajectory
from helpers.inverse_collocation_jax_v2 import solve_collocation_jax
from helpers.inverse_method_fixed import newton_corrector


# ==================== 1. ПОВЕРХНОСТЬ И ТСН ====================
phi_c = [0.0000000005642, -0.0000003012748, 0.0000605882383,
         -0.0099656628535, 2.9503573330764]
R_c = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
       39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bounds = [0, 234.27, 534.27, 768.54]
cyl_r = 251.705
E2 = FixedPiecewisePolynomialRevolution(phi_c, R_c, bounds, cyl_r)
print(f"E2: u∈[{E2.u_min:.1f}, {E2.u_max:.1f}], R_cyl={cyl_r}")

df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True].copy()
points_tsn = df_valid[['X', 'Y', 'Z']].values
traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"ТСН: {len(points_tsn)} точек, длина={traj.total_length:.2f}")


# ==================== 2. НАЧАЛЬНАЯ ТОЧКА ====================
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


# ==================== 3. ГИБРИДНЫЙ КАСКАД ====================
def solve_hybrid_adaptive(surface, traj, u0, v0,
                          N_levels=[25, 50, 100, 200],
                          tresh=1e-3,
                          w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
                          max_nfev=20000, tol=1e-8,
                          verbose=True):
    """
    Каскад: решаем сначала на грубой сетке, проверяем max|Phi|.
    Если < tresh -> stop. Иначе интерполируем на следующий уровень.
    """
    prev_result = None
    history = []
    final_result = None

    for level_idx, N in enumerate(N_levels, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"HYBRID LEVEL {level_idx}/{len(N_levels)}: N={N}")
            print(f"{'='*60}")

        if prev_result is not None:
            # Интерполяция с предыдущего уровня
            z_new = np.linspace(0, traj.total_length, N)
            z_prev = prev_result['z_eval']
            u_prev = prev_result['u']
            v_prev = np.unwrap(prev_result['v'])

            u_interp = np.interp(z_new, z_prev, u_prev)
            v_interp = np.interp(z_new, z_prev, v_prev)

            X0_custom = np.zeros(2 * (N - 1))
            X0_custom[0::2] = u_interp[1:]
            X0_custom[1::2] = v_interp[1:]
            init_method = 'custom'
        else:
            X0_custom = None
            init_method = 'radial'

        snap_name = f"snapshot_hybrid_{N}.npz"

        res = solve_collocation_jax(
            surface, traj, u0, v0,
            count_points=N,
            w_Phi=w_Phi, w_diff=w_diff, w_smooth=w_smooth,
            init_method=init_method,
            X0_custom=X0_custom,
            jac_mode='3-point',
            max_nfev=max_nfev,
            tol=tol,
            snapshot_path=snap_name,
            verbose=verbose
        )

        max_phi = np.max(np.abs(res['Phi']))
        mean_phi = np.mean(np.abs(res['Phi']))

        history.append({
            'level': level_idx,
            'N': N,
            'max_Phi': float(max_phi),
            'mean_Phi': float(mean_phi),
            'res_norm': float(res['res_norm']),
            'nfev': int(res['nfev']),
            'time': float(res['time']),
            'stopped': False,
        })

        if verbose:
            print(f"\n>>> N={N}: max|Phi|={max_phi:.2e}, mean|Phi|={mean_phi:.2e}, "
                  f"|F|={res['res_norm']:.2e}, nfev={res['nfev']}")

        if max_phi < tresh:
            print(f"\n[STOP] max|Phi|={max_phi:.2e} < tresh={tresh}. "
                  f"Уровень N={N} достаточен. Экономия: пропущено {len(N_levels)-level_idx} уровней.")
            history[-1]['stopped'] = True
            final_result = res
            break

        prev_result = res
        final_result = res

    return final_result, history


# ==================== 4. ЗАПУСК ====================
if __name__ == "__main__":
    t_start = time.time()

    result, history = solve_hybrid_adaptive(
        E2, traj, u0, v0,
        N_levels=[25, 50, 100, 200],
        tresh=1e-3,      # <--- порог остановки
        w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
        max_nfev=20000,
        tol=1e-8,
        verbose=True
    )

    t_total = time.time() - t_start
    final_N = len(result['z_eval'])
    final_max_phi = np.max(np.abs(result['Phi']))

    print(f"\n{'='*60}")
    print(f"HYBRID CASCADE FINISHED")
    print(f"{'='*60}")
    print(f"Total time: {t_total:.1f}s")
    print(f"Final N: {final_N}")
    print(f"Final max|Phi|: {final_max_phi:.2e}")
    print(f"Levels passed: {len(history)}/{4}")
    if history[-1]['stopped']:
        print(f"Stopped early at N={history[-1]['N']} (saved {4-len(history)} levels)")
    print(f"{'='*60}")

    # ==================== 5. СОХРАНЕНИЕ ====================
    np.savez('hybrid_result.npz',
             z_eval=result['z_eval'],
             u=result['u'], v=result['v'],
             points_3d=result['points_3d'],
             Phi=result['Phi'])

    with open('hybrid_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("Saved: hybrid_result.npz, hybrid_history.json")

    # ==================== 6. ВИЗУАЛИЗАЦИЯ ====================
    tsn_pts = np.array([traj.R(z) for z in np.linspace(0, traj.total_length, 300)])
    fig = go.Figure()

    # Поверхность
    u_grid = np.linspace(E2.u_min, E2.u_max, 30)
    v_grid = np.linspace(0, 2*np.pi, 30)
    Um, Vm = np.meshgrid(u_grid, v_grid)
    X2 = np.zeros_like(Um); Y2 = np.zeros_like(Um); Z2 = np.zeros_like(Um)
    for i in range(Um.shape[0]):
        for j in range(Um.shape[1]):
            p = E2.position(Um[i,j], Vm[i,j])
            X2[i,j], Y2[i,j], Z2[i,j] = p
    fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.25, colorscale='Reds',
                             showscale=False, name='E2'))

    # ТСН
    fig.add_trace(go.Scatter3d(x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
                                 mode='lines', line=dict(color='blue', width=4), name='ТСН'))

    # ЛУ
    pts = result['points_3d']
    z_eval = result['z_eval']
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='lines+markers',
        line=dict(color='green', width=3),
        marker=dict(size=3, color=z_eval, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='z')),
        name=f'ЛУ (N={final_N}, max|Φ|={final_max_phi:.1e})'
    ))

    fig.update_layout(
        title=f'Гибридная коллокация: final N={final_N}, max|Φ|={final_max_phi:.1e}',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1400, height=1000
    )
    fig.write_html("hybrid_result.html")
    print("Saved: hybrid_result.html")
