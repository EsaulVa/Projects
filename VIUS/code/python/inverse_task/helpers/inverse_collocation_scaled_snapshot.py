import numpy as np
from scipy.optimize import least_squares, brentq
import time

# ============================================================
# 1. Встроенный корректор Ньютона
# ============================================================

def newton_corrector(surface, traj, u, v, z, eps_Phi=1e-8, max_iter=20):
    """Проекция точки на многообразие связи Phi=0 (формула 10 отчёта)."""
    for i in range(max_iter):
        r = surface.position(u, v)
        m = surface.normal(u, v)
        R = traj.R(z)
        Phi = np.dot(R - r, m)
        if abs(Phi) < eps_Phi:
            return u, v, Phi, i, True

        # Численный градиент Phi по (u,v)
        du = 1e-6
        r_u = surface.position(u + du, v)
        m_u = surface.normal(u + du, v)
        Phi_u = np.dot(R - r_u, m_u)
        dPhi_du = (Phi_u - Phi) / du

        r_v = surface.position(u, v + du)
        m_v = surface.normal(u, v + du)
        Phi_v = np.dot(R - r_v, m_v)
        dPhi_dv = (Phi_v - Phi) / du

        # Метрика для поднятия индексов
        E, F, G = surface.first_fundamental_form(u, v)
        det = E * G - F * F
        if abs(det) < 1e-12:
            return u, v, Phi, i, False
        g_inv = np.array([[G, -F], [-F, E]]) / det

        grad_cov = np.array([dPhi_du, dPhi_dv])
        grad_contra = g_inv @ grad_cov
        grad_sq = grad_cov @ grad_contra

        if grad_sq < 1e-12:
            return u, v, Phi, i, False

        delta = -Phi / grad_sq * grad_contra
        u += delta[0]
        v += delta[1]

    return u, v, Phi, max_iter, False


# ============================================================
# 2. Radial init для баллона (по радиусу, а не по Z!)
# ============================================================

# def radial_init_balloon(surface, traj, z_eval, u0, v0):
#     """Начальное приближение: проекция ТСН на оправку по радиусу."""
#     N = len(z_eval)
#     u = np.zeros(N)
#     v = np.zeros(N)

#     for k in range(N):
#         Rk = traj.R(z_eval[k])
#         r_target = np.hypot(Rk[0], Rk[1])

#         # Находим u так, что surface.radius(u) = r_target
#         try:
#             u_k = brentq(
#                 lambda u: surface.radius(u) - r_target,
#                 surface.u_min, surface.u_max,
#                 xtol=1e-9
#             )
#         except ValueError:
#             r_min = surface.radius(surface.u_min)
#             r_max = surface.radius(surface.u_max)
#             u_k = surface.u_min if abs(r_target - r_min) < abs(r_target - r_max) else surface.u_max

#         v_k = np.arctan2(Rk[1], Rk[0])
#         u_k, v_k, _, _, _ = newton_corrector(surface, traj, u_k, v_k, z_eval[k])
#         u[k] = u_k
#         v[k] = v_k

#     return u, v
def radial_init_balloon(surface, traj, z_eval, u0, v0, grid_size=10000):
    """Начальное приближение: проекция ТСН на оправку по радиусу."""
    N = len(z_eval)
    u = np.zeros(N)
    v = np.zeros(N)

    u_grid = np.linspace(surface.u_min, surface.u_max, grid_size)
    r_grid = np.array([surface.radius(ui) for ui in u_grid])

    for k in range(N):
        Rk = traj.R(z_eval[k])
        r_target = np.hypot(Rk[0], Rk[1])

        # Топ-5 ближайших по радиусу (включая обе ветви баллона)
        idx_sorted = np.argsort(np.abs(r_grid - r_target))[:5]
        u_candidates = u_grid[idx_sorted]

        # Выбираем кандидата, ближайшего к предыдущей точке (или к u0 для k=0)
        if k == 0:
            u_prev = u0
        else:
            u_prev = u[k-1]

        u_k = u_candidates[np.argmin(np.abs(u_candidates - u_prev))]

        v_k = np.arctan2(Rk[1], Rk[0])
        u_k, v_k, _, _, _ = newton_corrector(surface, traj, u_k, v_k, z_eval[k])
        u[k] = u_k
        v[k] = v_k

    v = np.unwrap(v)
    return u, v

# ============================================================
# 3. Snapshot
# ============================================================

def load_collocation_snapshot(snapshot_file='collocation_snapshot.npz'):
    try:
        data = np.load(snapshot_file, allow_pickle=True)
        X = data['X']
        meta = {k: data[k] for k in data.files if k != 'X'}
        print(f"[Snapshot] Loaded: max|Phi|={meta.get('best_Phi', '?'):.3e}, "
              f"N={len(meta.get('z_eval', []))}")
        return X, meta
    except FileNotFoundError:
        return None, None


# ============================================================
# 4. Основной solver
# ============================================================

def solve_collocation_scaled(
    surface, traj, u0, v0,
    count_points=100,
    w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
    init_method='radial',
    X0_custom=None,
    max_nfev=10000,
    tol=1e-8,
    verbose=True,
    snapshot_file='collocation_snapshot.npz',
    snapshot_interval=1,
    restore_from_snapshot=True
):
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = count_points

    # --- Масштабирование ---
    u_min = getattr(surface, 'u_min', 0.0)
    u_max = getattr(surface, 'u_max', 1.0)
    du = u_max - u_min
    scale_u = du if du > 0 else 1.0
    scale_v = 2.0 * np.pi

    def unscale(X):
        u = X[0::2] * scale_u + u_min
        v = X[1::2] * scale_v
        return u, v

    # --- Восстановление из снапшота ---
    if restore_from_snapshot and X0_custom is None:
        X_snap, meta_snap = load_collocation_snapshot(snapshot_file)
        if X_snap is not None and len(X_snap) == 2 * N:
            if verbose:
                print(f"[Init] Restored from snapshot: max|Phi|={meta_snap.get('best_Phi', '?'):.3e}")
            X0_custom = X_snap.copy()
        elif X_snap is not None:
            if verbose:
                print(f"[Init] Snapshot N={len(X_snap)//2} != current N={N}. Ignoring.")

    # --- Построение X0 (масштабированного!) ---
    if X0_custom is not None:
        X0 = X0_custom.copy()
    else:
        if init_method == 'radial':
            u_init, v_init = radial_init_balloon(surface, traj, z_eval, u0, v0)
        else:
            raise ValueError(f"init_method='{init_method}' not supported. Use 'radial'.")

        X0 = np.zeros(2 * N)
        X0[0::2] = (u_init - u_min) / scale_u
        X0[1::2] = v_init / scale_v

    # --- Диагностика init ---
    u_test, v_test = unscale(X0)
    phi_all = []
    for k in range(N):
        r = surface.position(u_test[k], v_test[k])
        m = surface.normal(u_test[k], v_test[k])
        phi = np.dot(traj.R(z_eval[k]) - r, m)
        phi_all.append(abs(phi))
        if abs(phi) > 1.0:
            print(f"[BAD POINT k={k:4d}] z={z_eval[k]:.1f}, "
                  f"u={u_test[k]:.3f}, v={v_test[k]:.3f}, Phi={phi:.3e}")

    print(f"[INIT STAT] max|Phi|={max(phi_all):.3e} at k={np.argmax(phi_all)}")
    print(f"[INIT STAT] mean|Phi|={np.mean(phi_all):.3e}")
    print(f"[INIT STAT] bad points (|Phi|>1): {sum(1 for p in phi_all if p>1)} / {N}")
    if max(phi_all) > 100:
        print("!!! RADIAL INIT FAILED. Остановка.")
        return None

    # --- Residual (с демасштабированием) ---
    def residual_logged(X):
        u, v = unscale(X)
        res = []
        dz = z_eval[1] - z_eval[0] if N > 1 else 1.0

        # A. Связь
        for k in range(N):
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            Phi = np.dot(traj.R(z_eval[k]) - r, m)
            res.append(w_Phi * Phi)

        # B. Динамика
        for k in range(N - 1):
            du_ = u[k + 1] - u[k]
            dv_ = v[k + 1] - v[k]
            res.append(w_diff * du_ / dz)
            res.append(w_diff * dv_ / dz)

        # C. Гладкость
        for k in range(N - 1):
            E, F, G = surface.first_fundamental_form(u[k], v[k])
            du_ = u[k + 1] - u[k]
            dv_ = v[k + 1] - v[k]
            ds = np.sqrt(max(E * du_**2 + 2 * F * du_ * dv_ + G * dv_**2, 0.0))
            res.append(w_smooth * ds)

        # D. Граничные
        res.append(10.0 * (u[0] - u0))
        res.append(10.0 * (v[0] - v0))
        return np.array(res)

    # --- Трекер лучшего |Phi| ---
    best_state = {'Phi': np.inf, 'X': None, 'iter': 0}
    improve_count = 0

    def compute_max_Phi(X):
        u, v = unscale(X)
        max_phi = 0.0
        for k in range(N):
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            phi = abs(np.dot(traj.R(z_eval[k]) - r, m))
            if phi > max_phi:
                max_phi = phi
        return max_phi

    # --- Callback ---
    def callback(xk, *args, **kwargs):
        nonlocal improve_count
        if hasattr(xk, 'x'):
            x = np.copy(xk.x)
        else:
            x = np.copy(xk)

        max_phi = compute_max_Phi(x)
        if max_phi < best_state['Phi']:
            best_state['Phi'] = max_phi
            best_state['X'] = x.copy()
            best_state['iter'] += 1
            improve_count += 1

            if improve_count % snapshot_interval == 0:
                np.savez(snapshot_file,
                         X=best_state['X'],
                         best_Phi=float(best_state['Phi']),
                         z_eval=z_eval,
                         u0=float(u0), v0=float(v0),
                         count_points=int(N),
                         improve_iter=int(best_state['iter']),
                         timestamp=time.time())
                if verbose:
                    print(f"[Snapshot] New best max|Phi|={max_phi:.3e} "
                          f"(iter {best_state['iter']}) -> {snapshot_file}")

    # --- Запуск ---
    t0 = time.time()
    sol = least_squares(
        residual_logged,
        X0,
        method='trf',
        jac='2-point',
        max_nfev=max_nfev,
        ftol=tol, xtol=tol, gtol=tol,
        callback=callback,
        verbose=2 if verbose else 0
    )
    t1 = time.time()

    # --- Финальная коррекция ---
    final_Phi = compute_max_Phi(sol.x)
    if best_state['X'] is not None and best_state['Phi'] < final_Phi:
        if verbose:
            print(f"\n[Restore] Final max|Phi|={final_Phi:.3e} WORSE than "
                  f"snapshot={best_state['Phi']:.3e}. Restoring best snapshot.")
        sol.x = best_state['X'].copy()
        np.savez(snapshot_file,
                 X=sol.x,
                 best_Phi=float(best_state['Phi']),
                 z_eval=z_eval,
                 u0=float(u0), v0=float(v0),
                 count_points=int(N),
                 final=True,
                 timestamp=time.time())

    if verbose:
        max_Phi = compute_max_Phi(sol.x)
        print(f"\n{'='*60}")
        print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
        print(f"|F|_opt   = {np.linalg.norm(sol.fun):.3e}")
        print(f"Макс |Φ|  = {max_Phi:.3e}")
        print(f"nfev      = {sol.nfev}")
        print(f"success   = {sol.success}")
        print(f"Время     = {t1-t0:.1f} сек")
        print(f"{'='*60}")

    return sol