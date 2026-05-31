import numpy as np
from scipy.optimize import least_squares
import time

# ============================================================
# Функция загрузки снапшота (вызывайте при перезапуске)
# ============================================================

def load_collocation_snapshot(snapshot_file='collocation_snapshot.npz'):
    """
    Загружает лучший сохранённый снапшот.
    Возвращает:
        X, meta  или  None, None  если файла нет.
    """
    try:
        data = np.load(snapshot_file, allow_pickle=True)
        X = data['X']
        meta = {k: data[k] for k in data.files if k != 'X'}
        print(f"[Snapshot] Loaded: max|Phi|={meta.get('best_Phi', '?'):.3e}, "
              f"iter={meta.get('iter', '?')}, N={len(meta.get('z_eval', []))}")
        return X, meta
    except FileNotFoundError:
        return None, None


# ============================================================
# Основная функция с checkpointing
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
    # --- новые параметры ---
    snapshot_file='collocation_snapshot.npz',
    snapshot_interval=1,   # сохранять каждое улучшение (1) или каждые N улучшений
    restore_from_snapshot=True  # если True, пытается загрузить X0 из snapshot
):
    """
    Глобальная коллокация с сохранением лучшего по |Phi| снапшота.
    """
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = count_points

    # --- Попытка восстановления предыдущего лучшего снапшота ---
    if restore_from_snapshot and X0_custom is None:
        X_snap, meta_snap = load_collocation_snapshot(snapshot_file)
        if X_snap is not None and len(X_snap) == 2 * N:
            if verbose:
                print(f"[Init] Restored from snapshot: max|Phi|={meta_snap.get('best_Phi', '?'):.3e}")
            X0_custom = X_snap.copy()
        elif X_snap is not None:
            if verbose:
                print(f"[Init] Snapshot N={len(X_snap)//2} != current N={N}. Ignoring.")

    # --- Построение начального приближения (если не восстановили) ---
    if X0_custom is not None:
        X0 = X0_custom.copy()
    else:
        # ... ваш текущий init-код (radial / linear / etc) ...
        # [здесь оставьте свой существующий блок init]
        X0 = np.zeros(2 * N)
        # ==== ВСТАВЬТЕ СЮДА ВАШ INIT ====
        # Пример-заглушка (замените на radial_init или что у вас там):
        for k in range(N):
            Rk = traj.R(z_eval[k])
            # ... ваш radial init ...
            X0[2*k] = 0.0   # u_k
            X0[2*k+1] = 0.0 # v_k
        # ==================================

    # --- Масштабирование (ваш существующий код) ---
    u_min, u_max = surface.u_bounds()
    du = u_max - u_min
    scale_u = du if du > 0 else 1.0
    scale_v = 2.0 * np.pi

    # --- Residual (ваш существующий make_residual) ---
    def make_residual():
        # ... здесь весь ваш текущий residual ...
        # Для примера — минимальная заглушка:
        def residual_logged(X):
            u = X[0::2]
            v = X[1::2]
            res = []
            # A. Phi
            for k in range(N):
                r = surface.position(u[k], v[k])
                m = surface.normal(u[k], v[k])
                Phi = np.dot(traj.R(z_eval[k]) - r, m)
                res.append(w_Phi * Phi)
            # B. diff (заглушка)
            for k in range(N-1):
                res.append(w_diff * (u[k+1]-u[k]))
                res.append(w_diff * (v[k+1]-v[k]))
            # C. smooth (заглушка)
            for k in range(N-1):
                E,F,G = surface.first_fundamental_form(u[k], v[k])
                du_ = u[k+1]-u[k]
                dv_ = v[k+1]-v[k]
                ds = np.sqrt(max(E*du_**2 + 2*F*du_*dv_ + G*dv_**2, 0.0))
                res.append(w_smooth * ds)
            # D. Граничные
            res.append(10.0 * (u[0] - u0))
            res.append(10.0 * (v[0] - v0))
            return np.array(res)
        return residual_logged

    residual_logged = make_residual()

    # --- Трекер лучшего |Phi| ---
    best_state = {
        'Phi': np.inf,
        'X': None,
        'iter': 0,
        'nfev': 0,
        'time': time.time()
    }
    improve_count = 0

    def compute_max_Phi(X):
        """Вычисляет max |Phi| по текущему X (вне callback, для финальной проверки)."""
        u = X[0::2]
        v = X[1::2]
        max_phi = 0.0
        for k in range(N):
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            phi = abs(np.dot(traj.R(z_eval[k]) - r, m))
            if phi > max_phi:
                max_phi = phi
        return max_phi

    # --- Callback для least_squares ---
    def callback(x, f, *args, **kwargs):
        """
        Вызывается после каждой итерации least_squares.
        x — текущий вектор параметров (2N,)
        f — текущий вектор невязок (M,)
        """
        nonlocal improve_count
        # Вычисляем max |Phi| напрямую из x (точнее, чем из f, т.к. в f смешаны веса)
        u = x[0::2]
        v = x[1::2]
        max_phi = 0.0
        for k in range(N):
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            phi = abs(np.dot(traj.R(z_eval[k]) - r, m))
            if phi > max_phi:
                max_phi = phi

        # Если улучшились — сохраняем
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

    # --- Запуск оптимизатора ---
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

    # --- Финальная проверка: если снапшот лучше финального результата ---
    final_Phi = compute_max_Phi(sol.x)
    if best_state['X'] is not None and best_state['Phi'] < final_Phi:
        if verbose:
            print(f"\n[Restore] Final max|Phi|={final_Phi:.3e} WORSE than "
                  f"snapshot={best_state['Phi']:.3e}. Restoring best snapshot.")
        sol.x = best_state['X'].copy()
        # Перезаписываем финальный снапшот
        np.savez(snapshot_file,
                 X=sol.x,
                 best_Phi=float(best_state['Phi']),
                 z_eval=z_eval,
                 u0=float(u0), v0=float(v0),
                 count_points=int(N),
                 final=True,
                 timestamp=time.time())

    if verbose:
        u_opt = sol.x[0::2]
        v_opt = sol.x[1::2]
        max_Phi = compute_max_Phi(sol.x)
        print(f"\n{'='*60}")
        print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
        print(f"{'='*60}")
        print(f"|F|_opt   = {np.linalg.norm(sol.fun):.3e}")
        print(f"Макс |Φ|  = {max_Phi:.3e}")
        print(f"nfev      = {sol.nfev}")
        print(f"success   = {sol.success}")
        print(f"Время     = {t1-t0:.1f} сек")
        print(f"Snapshot  = {snapshot_file}")
        print(f"{'='*60}")

    return sol