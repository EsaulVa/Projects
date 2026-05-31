# clients/client_collocation_jax_multiscale_adaptive.py
"""
Многомасштабная JAX-коллокация с адаптивным h-измельчением.
На каждом уровне (25 -> 50 -> 100 -> 200) после решения проверяется
локальная невязка интервала. Где > tresh — вставляется узел.
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
from helpers.inverse_collocation_jax_v2 import solve_collocation_jax, load_snapshot
from helpers.inverse_method_fixed import newton_corrector, compute_dr_dz


# ==================== 1. ПОВЕРХНОСТЬ И ТСН ====================
phi_c = [0.0000000005642, -0.0000003012748, 0.0000605882383,
         -0.0099656628535, 2.9503573330764]
R_c = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
       39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bounds = [0, 234.27, 534.27, 768.54]
cyl_r = 251.705
E2 = FixedPiecewisePolynomialRevolution(phi_c, R_c, bounds, cyl_r)
print(f"E2: u∈[{E2.u_min:.1f}, {E2.u_max:.1f}]")

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


# ==================== 3. АДАПТИВНОЕ ИЗМЕЛЬЧЕНИЕ ====================
def compute_interval_errors(surface, traj, z_eval, u, v):
    """
    Вычисляет локальную невязку на каждом интервале [k, k+1]:
    interval_err = 0.5*(|Phi_k|+|Phi_{k+1}|) + |diff_u| + |diff_v|
    """
    N = len(z_eval)
    errors = []
    for k in range(N - 1):
        dz = z_eval[k + 1] - z_eval[k]
        if abs(dz) < 1e-12:
            errors.append(0.0)
            continue

        # Phi в узлах
        r_k = surface.position(u[k], v[k])
        m_k = surface.normal(u[k], v[k])
        Phi_k = abs(np.dot(traj.R(z_eval[k]) - r_k, m_k))

        r_k1 = surface.position(u[k + 1], v[k + 1])
        m_k1 = surface.normal(u[k + 1], v[k + 1])
        Phi_k1 = abs(np.dot(traj.R(z_eval[k + 1]) - r_k1, m_k1))

        # Diff (Hermite residual)
        try:
            du_k, dv_k = compute_dr_dz(surface, traj, u[k], v[k], z_eval[k])
            du_next, dv_next = compute_dr_dz(surface, traj, u[k + 1], v[k + 1], z_eval[k + 1])
        except Exception:
            du_k = dv_k = du_next = dv_next = 0.0

        u_mid = 0.5 * (u[k] + u[k + 1]) + (dz / 8.0) * (du_k - du_next)
        v_mid = 0.5 * (v[k] + v[k + 1]) + (dz / 8.0) * (dv_k - dv_next)
        z_mid = 0.5 * (z_eval[k] + z_eval[k + 1])

        du_spline = 1.5 * (u[k + 1] - u[k]) / dz - 0.25 * (du_k + du_next)
        dv_spline = 1.5 * (v[k + 1] - v[k]) / dz - 0.25 * (dv_k + dv_next)

        try:
            du_rhs, dv_rhs = compute_dr_dz(surface, traj, u_mid, v_mid, z_mid)
        except Exception:
            du_rhs = dv_rhs = 0.0

        diff_err = abs(du_spline - du_rhs) + abs(dv_spline - dv_rhs)
        interval_err = 0.5 * (Phi_k + Phi_k1) + diff_err
        errors.append(float(interval_err))
    return np.array(errors)


def adaptive_refinement(surface, traj, z_eval, u, v, tresh=1e-3):
    """
    Вставляет средние точки в интервалы, где interval_err > tresh.
    Возвращает новую неравномерную сетку.
    """
    N = len(z_eval)
    errors = compute_interval_errors(surface, traj, z_eval, u, v)

    new_z = [float(z_eval[0])]
    new_u = [float(u[0])]
    new_v = [float(v[0])]

    for k in range(N - 1):
        new_z.append(float(z_eval[k + 1]))
        new_u.append(float(u[k + 1]))
        new_v.append(float(v[k + 1]))

        if errors[k] > tresh:
            z_mid = 0.5 * (z_eval[k] + z_eval[k + 1])
            u_mid = 0.5 * (u[k] + u[k + 1])
            v_mid = 0.5 * (v[k] + v[k + 1])
            # Вставляем перед последним добавленным
            new_z.insert(-1, float(z_mid))
            new_u.insert(-1, float(u_mid))
            new_v.insert(-1, float(v_mid))

    order = np.argsort(new_z)
    return np.array(new_z)[order], np.array(new_u)[order], np.array(new_v)[order], errors


# ==================== 4. АНСАМБЛЬ НА ГРУБОМ УРОВНЕ ====================
def run_ensemble(N, methods, weights_grid, u0, v0):
    results = []
    for i, (method, (wP, wD, wS)) in enumerate(zip(methods, weights_grid), 1):
        print(f"\n--- Ensemble {i}/{len(methods)}: N={N}, method={method}, w=({wP:.1f},{wD:.1f},{wS:.2f}) ---")
        snap_name = f"snapshot_ensemble_{method}_{N}.npz"
        try:
            res = solve_collocation_jax(
                E2, traj, u0, v0,
                count_points=N,
                w_Phi=wP, w_diff=wD, w_smooth=wS,
                init_method=method,
                jac_mode='3-point',
                max_nfev=15000,
                tol=1e-7,
                snapshot_path=snap_name,
                verbose=False
            )
            max_phi = np.max(np.abs(res['Phi']))
            results.append({
                'method': method,
                'weights': (wP, wD, wS),
                'max_Phi': max_phi,
                'mean_Phi': np.mean(np.abs(res['Phi'])),
                'res_norm': res['res_norm'],
                'nfev': res['nfev'],
                'time': res['time'],
                'u': res['u'], 'v': res['v'],
                'points_3d': res['points_3d'],
                'z_eval': res['z_eval'],
                'snapshot_path': snap_name,
            })
            print(f"  -> max|Phi|={max_phi:.2e}, |F|={res['res_norm']:.2e}, time={res['time']:.1f}s")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append(None)

    valid = [r for r in results if r is not None]
    if not valid:
        raise RuntimeError("All ensemble members failed")
    winner = min(valid, key=lambda x: x['max_Phi'])
    print(f"\n[ENSEMBLE WINNER] method={winner['method']}, max|Phi|={winner['max_Phi']:.2e}")
    return winner, valid


# ==================== 5. КАСКАД С АДАПТИВНЫМ ИЗМЕЛЬЧЕНИЕМ ====================
def interpolate_solution(prev_result, N_new):
    z_prev = prev_result['z_eval']
    u_prev = prev_result['u']
    v_prev = prev_result['v']
    v_prev_unwrapped = np.unwrap(v_prev)
    z_new = np.linspace(0, traj.total_length, N_new)
    u_interp = np.interp(z_new, z_prev, u_prev)
    v_interp = np.interp(z_new, z_prev, v_prev_unwrapped)
    X0 = np.zeros(2 * (N_new - 1))
    X0[0::2] = u_interp[1:]
    X0[1::2] = v_interp[1:]
    return X0


def run_adaptive_cascade(base_levels, tresh, max_refinement_iters, winner_coarse):
    prev = winner_coarse
    cascade_results = []
    wP, wD, wS = winner_coarse['weights']

    for level_idx, (N_base, max_nfev, tol) in enumerate(base_levels, 1):
        print(f"\n{'='*60}")
        print(f"BASE LEVEL {level_idx}: N={N_base}")
        print(f"{'='*60}")

        # Начальная равномерная сетка
        z_uniform = np.linspace(0, traj.total_length, N_base)
        X0_custom = interpolate_solution(prev, N_base)

        res = solve_collocation_jax(
            E2, traj, u0, v0,
            count_points=N_base,
            z_eval=z_uniform,  # равномерная
            w_Phi=wP, w_diff=wD, w_smooth=wS,
            init_method='custom',
            X0_custom=X0_custom,
            jac_mode='3-point',
            max_nfev=max_nfev,
            tol=tol,
            snapshot_path=f'snapshot_base_{N_base}.npz',
            verbose=True
        )

        current_z = res['z_eval']
        current_u = res['u']
        current_v = res['v']

        # --- Адаптивное измельчение ---
        for ref_iter in range(max_refinement_iters):
            z_new, u_new, v_new, errors = adaptive_refinement(
                E2, traj, current_z, current_u, current_v, tresh=tresh
            )

            if len(z_new) == len(current_z):
                print(f"[Refinement {ref_iter+1}] No new points. Converged.")
                break

            print(f"[Refinement {ref_iter+1}] Added {len(z_new)-len(current_z)} points. "
                  f"Max interval_err={np.max(errors):.2e}")

            # Подготовка X0 для новой сетки
            u_min = E2.u_min
            scale_u = max(E2.u_max - u_min, 1.0)
            scale_v = 2 * np.pi
            X0_ref = np.zeros(2 * (len(z_new) - 1))
            X0_ref[0::2] = (u_new[1:] - u_min) / scale_u
            X0_ref[1::2] = v_new[1:] / scale_v

            res = solve_collocation_jax(
                E2, traj, u0, v0,
                z_eval=z_new,  # <--- неравномерная сетка
                w_Phi=wP, w_diff=wD, w_smooth=wS,
                init_method='custom',
                X0_custom=X0_ref,
                jac_mode='3-point',
                max_nfev=max_nfev,
                tol=tol,
                snapshot_path=f'snapshot_ref_{level_idx}_{ref_iter}.npz',
                verbose=True
            )

            current_z = res['z_eval']
            current_u = res['u']
            current_v = res['v']

        cascade_results.append({
            'N_base': N_base,
            'N_final': len(current_z),
            'z_eval': current_z,
            'u': current_u,
            'v': current_v,
            'points_3d': res['points_3d'],
            'Phi': res['Phi'],
            'result': res,
        })
        prev = {
            'z_eval': current_z,
            'u': current_u,
            'v': current_v,
            'weights': winner_coarse['weights'],
        }

    return cascade_results


# ==================== 6. ЗАПУСК ====================
if __name__ == "__main__":
    t_start = time.time()

    # --- Уровень 0: Ансамбль N=25 ---
    print(f"\n{'='*60}")
    print("LEVEL 0: ENSEMBLE N=25")
    print(f"{'='*60}")

    ensemble_methods = ['radial', 'radial', 'linear']
    ensemble_weights = [
        (10.0, 0.1, 0.01),
        (10.0, 0.1, 0.0),
        (100.0, 0.1, 0.1),
    ]

    winner_coarse, all_ensemble = run_ensemble(25, ensemble_methods, ensemble_weights, u0, v0)

    # --- Каскад с адаптивным измельчением ---
    cascade_levels = [
        (50, 20000, 1e-7),
        (100, 30000, 1e-8),
        (200, 50000, 1e-9),
    ]

    tresh = 1e-3  # <--- порог локальной невязки интервала
    max_refinement_iters = 3

    cascade = run_adaptive_cascade(cascade_levels, tresh, max_refinement_iters, winner_coarse)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {t_total:.1f}s")
    print(f"{'='*60}")

    # ==================== 7. СОХРАНЕНИЕ ====================
    for item in cascade:
        N = item['N_base']
        np.savez(f"result_level_{N}.npz",
                 z_eval=item['z_eval'],
                 u=item['u'], v=item['v'],
                 points_3d=item['points_3d'],
                 Phi=item['Phi'])
        print(f"Saved: result_level_{N}.npz (N_final={item['N_final']})")

    meta = {
        'ensemble': [
            {'method': r['method'], 'max_Phi': float(r['max_Phi']),
             'mean_Phi': float(r['mean_Phi']), 'nfev': int(r['nfev'])}
            for r in all_ensemble
        ],
        'cascade': [
            {'N_base': c['N_base'], 'N_final': c['N_final'],
             'max_Phi': float(np.max(np.abs(c['Phi']))),
             'mean_Phi': float(np.mean(np.abs(c['Phi']))),
             'nfev': int(c['result']['nfev']), 'time': float(c['result']['time'])}
            for c in cascade
        ],
        'total_time': t_total,
        'tresh': tresh,
    }
    with open("multiscale_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved: multiscale_meta.json")

    # ==================== 8. ВИЗУАЛИЗАЦИЯ ====================
    tsn_pts = np.array([traj.R(z) for z in np.linspace(0, traj.total_length, 300)])

    fig = go.Figure()

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

    fig.add_trace(go.Scatter3d(x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
                                 mode='lines', line=dict(color='blue', width=4), name='ТСН'))

    colors = {50: 'green', 100: 'orange', 200: 'magenta'}
    for item in cascade:
        N = item['N_base']
        pts = item['points_3d']
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='lines+markers',
            line=dict(color=colors.get(N, 'gray'), width=3),
            marker=dict(size=2),
            name=f'ЛУ base={N}, final={item["N_final"]} (max|Φ|={np.max(np.abs(item["Phi"])):.1e})'
        ))

    fig.update_layout(
        title='Адаптивная коллокация: каскад + h-refinement',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1400, height=1000,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.write_html("multiscale_adaptive.html")
    print("\nSaved: multiscale_adaptive.html")
