# clients/client_collocation_jax_adaptive_pure.py
"""
Чистый адаптивный solver с PARTIAL OPTIMIZATION.
На итерациях > 1 оптимизируются только узлы, соседние с плохими интервалами.
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


# ==================== 3. УТИЛИТЫ ====================
def compute_interval_errors(surface, traj, z_eval, u, v):
    N = len(z_eval)
    errors = []
    for k in range(N - 1):
        dz = z_eval[k + 1] - z_eval[k]
        if abs(dz) < 1e-12:
            errors.append(0.0); continue
        r_k = surface.position(u[k], v[k])
        m_k = surface.normal(u[k], v[k])
        Phi_k = abs(np.dot(traj.R(z_eval[k]) - r_k, m_k))
        r_k1 = surface.position(u[k + 1], v[k + 1])
        m_k1 = surface.normal(u[k + 1], v[k + 1])
        Phi_k1 = abs(np.dot(traj.R(z_eval[k + 1]) - r_k1, m_k1))
        try:
            du_k, dv_k = compute_dr_dz(surface, traj, u[k], v[k], z_eval[k])
            du_next, dv_next = compute_dr_dz(surface, traj, u[k+1], v[k+1], z_eval[k+1])
        except Exception:
            du_k = dv_k = du_next = dv_next = 0.0
        u_mid = 0.5 * (u[k] + u[k+1]) + (dz/8.0)*(du_k - du_next)
        v_mid = 0.5 * (v[k] + v[k+1]) + (dz/8.0)*(dv_k - dv_next)
        z_mid = 0.5 * (z_eval[k] + z_eval[k+1])
        du_spline = 1.5*(u[k+1]-u[k])/dz - 0.25*(du_k+du_next)
        dv_spline = 1.5*(v[k+1]-v[k])/dz - 0.25*(dv_k+dv_next)
        try:
            du_rhs, dv_rhs = compute_dr_dz(surface, traj, u_mid, v_mid, z_mid)
        except Exception:
            du_rhs = dv_rhs = 0.0
        diff_err = abs(du_spline - du_rhs) + abs(dv_spline - dv_rhs)
        interval_err = 0.5*(Phi_k + Phi_k1) + diff_err
        errors.append(float(interval_err))
    return np.array(errors)


def refine_and_correct(surface, traj, z_eval, u, v, errors, tresh):
    N = len(z_eval)
    z_new = [float(z_eval[0])]
    u_new = [float(u[0])]
    v_new = [float(v[0])]
    for k in range(N - 1):
        dz = z_eval[k+1] - z_eval[k]
        z_new.append(float(z_eval[k+1]))
        u_new.append(float(u[k+1]))
        v_new.append(float(v[k+1]))
        if errors[k] > tresh:
            z_mid = 0.5*(z_eval[k] + z_eval[k+1])
            try:
                du_k, dv_k = compute_dr_dz(surface, traj, u[k], v[k], z_eval[k])
                du_next, dv_next = compute_dr_dz(surface, traj, u[k+1], v[k+1], z_eval[k+1])
                u_mid = 0.5*(u[k]+u[k+1]) + (dz/8.0)*(du_k - du_next)
                v_mid = 0.5*(v[k]+v[k+1]) + (dz/8.0)*(dv_k - dv_next)
            except Exception:
                u_mid = 0.5*(u[k]+u[k+1])
                v_mid = 0.5*(v[k]+v[k+1])
            u_mid = np.clip(float(u_mid), float(surface.u_min), float(surface.u_max))
            try:
                u_c, v_c, Phi_c, _, conv = newton_corrector(
                    surface, traj, u_mid, v_mid, z_mid, eps_Phi=1e-8, max_iter=20
                )
                if conv and abs(Phi_c) < 1e-3:
                    u_mid = np.clip(float(u_c), float(surface.u_min), float(surface.u_max))
                    v_mid = float(v_c)
            except Exception:
                pass
            z_new.insert(-1, float(z_mid))
            u_new.insert(-1, float(u_mid))
            v_new.insert(-1, float(v_mid))
    order = np.argsort(z_new)
    return np.array(z_new)[order], np.array(u_new)[order], np.array(v_new)[order]


def build_fixed_indices(N, bad_intervals, neighbor_width=1):
    """Индексы в X_tail для фиксации. Оптимизируются только соседи плохих интервалов."""
    if len(bad_intervals) == 0:
        return np.array([], dtype=int)
    free_nodes = set()
    for k in bad_intervals:
        for node in range(max(1, k - neighbor_width), min(N, k + 1 + neighbor_width + 1)):
            free_nodes.add(node)
    all_nodes = set(range(1, N))
    fixed_nodes = sorted(all_nodes - free_nodes)
    fixed_indices = []
    for j in fixed_nodes:
        fixed_indices.extend([2*(j-1), 2*(j-1)+1])
    return np.array(fixed_indices, dtype=int)


def prepare_X0(z_new, u_new, v_new, u0, v0, u_min, scale_u, scale_v):
    X0 = np.zeros(2*(len(z_new)-1))
    X0[0::2] = (np.clip(u_new[1:], u_min, u_min+scale_u) - u_min)/scale_u
    X0[1::2] = v_new[1:]/scale_v
    return X0


# ==================== 4. SOLVER ====================
def solve_adaptive_pure(surface, traj, u0, v0,
                        N_start=50, tresh=1e-3, max_iters=10,
                        w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
                        max_nfev=20000, tol=1e-8,
                        snapshot_path='adaptive_snapshot.npz',
                        verbose=True):
    u_min = float(surface.u_min)
    scale_u = max(float(surface.u_max) - u_min, 1.0)
    scale_v = 2.0 * np.pi

    z = np.linspace(0, traj.total_length, N_start)
    u_rad, v_rad = [], []
    for k in range(N_start):
        Rk = traj.R(z[k])
        u_g = float(np.clip(Rk[2], surface.u_min, surface.u_max))
        v_g = float(np.arctan2(Rk[1], Rk[0]))
        try:
            r_s = float(surface.radius(u_g))
        except:
            r_s = 0.0
        r_ray = float(np.hypot(Rk[0], Rk[1]))
        if r_ray > 1e-6:
            sc = r_s / r_ray
            r_proj = np.array([Rk[0]*sc, Rk[1]*sc, Rk[2]])
        else:
            r_proj = np.array([r_s, 0.0, Rk[2]])
        try:
            up, vp = surface.uv_from_point(r_proj)
        except:
            up, vp = u_g, v_g
        up = np.clip(float(up), surface.u_min, surface.u_max)
        try:
            uc, vc, Phic, _, conv = newton_corrector(
                surface, traj, up, vp, z[k], eps_Phi=1e-8, max_iter=20
            )
            if conv and abs(Phic) < 1e-6:
                u_rad.append(uc); v_rad.append(vc); continue
        except:
            pass
        u_rad.append(up); v_rad.append(float(vp))
    u = np.array(u_rad)
    v = np.unwrap(np.array(v_rad))

    history = []
    fixed_indices_next = None  # для первой итерации — ничего не фиксируем

    for it in range(max_iters):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ADAPTIVE ITER {it+1}/{max_iters}: N={len(z)}")
            print(f"{'='*60}")

        X0 = prepare_X0(z, u, v, u0, v0, u_min, scale_u, scale_v)

        res = solve_collocation_jax(
            surface, traj, u0, v0,
            z_eval=z,
            w_Phi=w_Phi, w_diff=w_diff, w_smooth=w_smooth,
            init_method='radial',
            X0_custom=X0,
            jac_mode='3-point',
            fixed_indices=fixed_indices_next,
            eps_fixed=1e-9,
            max_nfev=max_nfev,
            tol=tol,
            snapshot_path=snapshot_path,
            verbose=verbose
        )

        max_phi = np.max(np.abs(res['Phi']))
        mean_phi = np.mean(np.abs(res['Phi']))
        if verbose:
            print(f"  -> max|Phi|={max_phi:.2e}, mean|Phi|={mean_phi:.2e}, "
                  f"|F|={res['res_norm']:.2e}, nfev={res['nfev']}")

        errors = compute_interval_errors(surface, traj, z, res['u'], res['v'])
        e_max = np.max(errors)
        n_bad = np.sum(errors > tresh)

        history.append({
            'iter': it+1, 'N': len(z),
            'max_Phi': float(max_phi), 'mean_Phi': float(mean_phi),
            'e_max': float(e_max), 'n_bad_intervals': int(n_bad),
            'nfev': int(res['nfev']), 'time': float(res['time']),
        })

        if e_max < tresh:
            print(f"\n[CONVERGED] e_max={e_max:.2e} < tresh={tresh}. Final N={len(z)}.")
            return res, history

        if it == max_iters - 1:
            print(f"\n[MAX ITERS] e_max={e_max:.2e}. Final N={len(z)}.")
            return res, history

        # Дробим
        z, u, v = refine_and_correct(surface, traj, z, res['u'], res['v'], errors, tresh)
        print(f"  [Refinement] N {history[-1]['N']} -> {len(z)} ({len(z)-history[-1]['N']} new)")

        # Готовим fixed_indices для СЛЕДУЩЕЙ итерации
        # Вычисляем preview-errors на новой сетке (по интерполированным u,v)
        errors_preview = compute_interval_errors(surface, traj, z, u, v)
        bad_intervals = np.where(errors_preview > tresh)[0]
        if len(bad_intervals) > 0:
            fixed_indices_next = build_fixed_indices(len(z), bad_intervals, neighbor_width=1)
            n_free = 2*(len(z)-1) - len(fixed_indices_next)
            print(f"  [Partial] Next iter: {n_free} free, {len(fixed_indices_next)} fixed vars")
        else:
            fixed_indices_next = None

    return res, history


# ==================== 5. ЗАПУСК ====================
if __name__ == "__main__":
    t_start = time.time()
    result, history = solve_adaptive_pure(
        E2, traj, u0, v0,
        N_start=40, tresh=1e-3, max_iters=10,
        w_Phi=10.0, w_diff=1e-2, w_smooth=1e-4,
        max_nfev=120000, tol=1e-8,
        snapshot_path='adaptive_pure_snapshot.npz',
        verbose=True
    )
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {t_total:.1f}s | FINAL N: {len(result['z_eval'])}")
    print(f"FINAL max|Phi|: {np.max(np.abs(result['Phi'])):.2e}")
    print(f"{'='*60}")

    np.savez('adaptive_pure_result.npz',
             z_eval=result['z_eval'], u=result['u'], v=result['v'],
             points_3d=result['points_3d'], Phi=result['Phi'])
    with open('adaptive_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("Saved: adaptive_pure_result.npz, adaptive_history.json")

    # Визуализация
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
    fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.25, colorscale='Reds', showscale=False, name='E2'))
    fig.add_trace(go.Scatter3d(x=tsn_pts[:,0], y=tsn_pts[:,1], z=tsn_pts[:,2],
                                 mode='lines', line=dict(color='blue', width=4), name='ТСН'))
    pts = result['points_3d']
    z_eval = result['z_eval']
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='lines+markers', line=dict(color='green', width=3),
        marker=dict(size=3, color=z_eval, colorscale='Viridis', showscale=True, colorbar=dict(title='z')),
        name=f'ЛУ (N={len(z_eval)}, max|Φ|={np.max(np.abs(result["Phi"])):.1e})'
    ))
    fig.update_layout(title=f'Partial Adaptive: N_start=50, final N={len(z_eval)}',
                      scene=dict(aspectmode='data'), width=1400, height=1000)
    fig.write_html("adaptive_pure.html")
    print("Saved: adaptive_pure.html")