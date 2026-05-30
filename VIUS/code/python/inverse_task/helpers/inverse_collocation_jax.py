# helpers/inverse_collocation_jax.py
"""
JAX-ускоренная коллокация для обратной задачи намотки.
Поверхность вращения предвычисляется на плотной сетке и интерполируется
внутри JAX-графа — это даёт jit + exact Jacobian без переписывания геометрии.
"""
import numpy as np
import time
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("WARNING: jax not installed, falling back to numpy")

from scipy.optimize import least_squares

try:
    from helpers.inverse_method_fixed import compute_dr_dz, newton_corrector
except ImportError:
    from helpers.inverse_method import compute_dr_dz, newton_corrector


# ==================== 1. Предвычисление геометрии ====================

def build_jax_surface_data(surface, n_dense=2000):
    """
    Предвычисляет R(u), R'(u), R''(u), Z(u) на плотной сетке.
    Для поверхности вращения всё выражается через радиус-функцию.
    """
    u_min = float(surface.u_min)
    u_max = float(surface.u_max)
    u_dense = np.linspace(u_min, u_max, n_dense)

    # Радиус и высота
    R_dense = np.array([float(surface.radius(ui)) for ui in u_dense])
    Z_dense = u_dense.copy()  # для вашей параметризации Z=u

    # Производные (numpy gradient — достаточно точно для сплайна)
    dR_dense = np.gradient(R_dense, u_dense)
    d2R_dense = np.gradient(dR_dense, u_dense)

    return {
        'u_min': u_min,
        'u_max': u_max,
        'u_dense': jnp.array(u_dense),
        'R': jnp.array(R_dense),
        'dR': jnp.array(dR_dense),
        'd2R': jnp.array(d2R_dense),
        'Z': jnp.array(Z_dense),
    }


def _interp(u, xp, fp):
    """JAX-интерполяция (дифференцируема по u)."""
    return jnp.interp(u, xp, fp)


# ==================== 2. JAX-геометрия ====================

def _position_jax(u, v, R_data):
    R = _interp(u, R_data['u_dense'], R_data['R'])
    Z = _interp(u, R_data['u_dense'], R_data['Z'])
    return jnp.array([R * jnp.cos(v), R * jnp.sin(v), Z])


def _normal_jax(u, v, R_data):
    dR = _interp(u, R_data['u_dense'], R_data['dR'])
    denom = jnp.sqrt(1.0 + dR ** 2)
    nr = -1.0 / denom
    nz = dR / denom
    return jnp.array([nr * jnp.cos(v), nr * jnp.sin(v), nz])


def _compute_dr_dz_jax(u, v, z, R_data, R_pt, dR_pt):
    """
    DAE-предиктор (формулы 10-13 из дополнения), полностью на JAX.
    """
    R = _interp(u, R_data['u_dense'], R_data['R'])
    dR = _interp(u, R_data['u_dense'], R_data['dR'])
    d2R = _interp(u, R_data['u_dense'], R_data['d2R'])

    denom = jnp.sqrt(1.0 + dR ** 2)

    # Первые формы
    E = dR ** 2 + 1.0
    F = 0.0
    G = R ** 2

    # Вторые формы
    L11 = -d2R / denom
    L12 = 0.0
    L22 = R / denom

    # Векторы
    r = _position_jax(u, v, R_data)
    m = _normal_jax(u, v, R_data)
    ru = jnp.array([dR * jnp.cos(v), dR * jnp.sin(v), 1.0])
    rv = jnp.array([-R * jnp.sin(v), R * jnp.cos(v), 0.0])

    # Зазор и производные
    Rmr = R_pt - r
    Phi = jnp.dot(Rmr, m)

    dg_du = L11 * jnp.dot(Rmr, ru) + L12 * jnp.dot(Rmr, rv)
    dg_dv = L12 * jnp.dot(Rmr, ru) + L22 * jnp.dot(Rmr, rv)

    # Базовая скорость (проекция dR/dz на касательную)
    v0u = jnp.dot(dR_pt, ru) / E
    v0v = jnp.dot(dR_pt, rv) / G

    dg_dz = jnp.dot(dR_pt, m)

    grad_sq = dg_du ** 2 + dg_dv ** 2
    mu = -(dg_du * v0u + dg_dv * v0v + dg_dz) / (grad_sq + 1e-12)

    du = v0u + mu * dg_du
    dv = v0v + mu * dg_dv
    return du, dv


# ==================== 3. JAX residual ====================

def _make_residual_jax(z_eval, u0, v0, scale_u, scale_v,
                       w_Phi, w_diff, w_smooth,
                       R_data, traj_R, traj_dR):
    """
    Фабрика чистой JAX-функции невязки.
    Возвращает функцию residual(X_tail_tilde) -> jnp.array.
    """
    N = len(z_eval)
    u_min = float(R_data['u_min'])
    u0_tilde = (u0 - u_min) / scale_u
    v0_tilde = v0 / scale_v

    z_eval_j = jnp.array(z_eval)
    traj_R_j = jnp.array(traj_R)
    traj_dR_j = jnp.array(traj_dR)

    def residual_fn(X_tail_tilde):
        # Восстановление физических переменных
        u_tilde = jnp.concatenate([jnp.array([u0_tilde]), X_tail_tilde[0::2]])
        v_tilde = jnp.concatenate([jnp.array([v0_tilde]), X_tail_tilde[1::2]])
        u = u_min + u_tilde * scale_u
        v = v_tilde * scale_v

        # --- A. Phi = 0 (векторизовано) ---
        def phi_k(k):
            Rk = traj_R_j[k]
            uk = u[k]
            vk = v[k]
            rk = _position_jax(uk, vk, R_data)
            mk = _normal_jax(uk, vk, R_data)
            return w_Phi * jnp.dot(Rk - rk, mk)

        Phi_vec = jax.vmap(phi_k)(jnp.arange(N))

        # --- B. Дифф. уравнения (Hermite) ---
        def diff_k(k):
            dz = z_eval_j[k + 1] - z_eval_j[k]
            # Если dz слишком мал — невязка 0
            mask = dz < 1e-12
            du_k, dv_k = _compute_dr_dz_jax(u[k], v[k], z_eval_j[k], R_data, traj_R_j[k], traj_dR_j[k])
            du_next, dv_next = _compute_dr_dz_jax(u[k + 1], v[k + 1], z_eval_j[k + 1], R_data, traj_R_j[k + 1], traj_dR_j[k + 1])

            u_mid = 0.5 * (u[k] + u[k + 1]) + (dz / 8.0) * (du_k - du_next)
            v_mid = 0.5 * (v[k] + v[k + 1]) + (dz / 8.0) * (dv_k - dv_next)
            z_mid = 0.5 * (z_eval_j[k] + z_eval_j[k + 1])

            # Интерполяция траектории в z_mid
            R_mid = jnp.array([
                jnp.interp(z_mid, z_eval_j, traj_R_j[:, 0]),
                jnp.interp(z_mid, z_eval_j, traj_R_j[:, 1]),
                jnp.interp(z_mid, z_eval_j, traj_R_j[:, 2]),
            ])
            dR_mid = jnp.array([
                jnp.interp(z_mid, z_eval_j, traj_dR_j[:, 0]),
                jnp.interp(z_mid, z_eval_j, traj_dR_j[:, 1]),
                jnp.interp(z_mid, z_eval_j, traj_dR_j[:, 2]),
            ])

            du_spline = 1.5 * (u[k + 1] - u[k]) / dz - 0.25 * (du_k + du_next)
            dv_spline = 1.5 * (v[k + 1] - v[k]) / dz - 0.25 * (dv_k + dv_next)

            du_rhs, dv_rhs = _compute_dr_dz_jax(u_mid, v_mid, z_mid, R_data, R_mid, dR_mid)

            res_u = w_diff * (du_spline - du_rhs)
            res_v = w_diff * (dv_spline - dv_rhs)
            return jnp.where(mask, jnp.array([0.0, 0.0]), jnp.array([res_u, res_v]))

        diff_vec = jax.vmap(diff_k)(jnp.arange(N - 1))  # (N-1, 2)

        # --- C. Гладкость ---
        def smooth_k(k):
            uk = u[k]
            R_val = _interp(uk, R_data['u_dense'], R_data['R'])
            dR_val = _interp(uk, R_data['u_dense'], R_data['dR'])
            E = dR_val ** 2 + 1.0
            G = R_val ** 2
            du = u[k + 1] - u[k]
            dv = v[k + 1] - v[k]
            ds = jnp.sqrt(jnp.maximum(E * du ** 2 + G * dv ** 2, 0.0))
            return w_smooth * ds

        smooth_vec = jax.vmap(smooth_k)(jnp.arange(N - 1))

        return jnp.concatenate([Phi_vec, diff_vec.flatten(), smooth_vec])

    return residual_fn


# ==================== 4. Инициализация ====================

def _radial_init(surface, traj, z_eval, N, u0, v0):
    u = []
    v = []
    for k in range(N):
        Rk = traj.R(z_eval[k])
        u_guess = float(np.clip(Rk[2], surface.u_min, surface.u_max))
        v_guess = float(np.arctan2(Rk[1], Rk[0]))
        try:
            r_surf = float(surface.radius(u_guess))
        except Exception:
            r_surf = 0.0
        r_ray = float(np.hypot(Rk[0], Rk[1]))
        if r_ray > 1e-6:
            scale = r_surf / r_ray
            r_proj = np.array([Rk[0] * scale, Rk[1] * scale, Rk[2]])
        else:
            r_proj = np.array([r_surf, 0.0, Rk[2]])
        try:
            u_proj, v_proj = surface.uv_from_point(r_proj)
        except Exception:
            u_proj, v_proj = u_guess, v_guess
        u_proj = np.clip(float(u_proj), surface.u_min, surface.u_max)
        try:
            u_c, v_c, Phi_c, _, conv = newton_corrector(
                surface, traj, u_proj, v_proj, z_eval[k], eps_Phi=1e-8, max_iter=20
            )
            if conv and abs(Phi_c) < 1e-6:
                u.append(float(u_c))
                v.append(float(v_c))
                continue
        except Exception:
            pass
        u.append(u_proj)
        v.append(float(v_proj))
    return np.array(u), np.array(v)


def _dae_init(surface, traj, z_eval, N, u0, v0):
    u = [float(u0)]
    v = [float(v0)]
    for k in range(1, N):
        dz = z_eval[k] - z_eval[k - 1]
        try:
            du, dv = compute_dr_dz(surface, traj, u[-1], v[-1], z_eval[k - 1])
        except Exception:
            du = dv = 0.0
        u.append(np.clip(u[-1] + du * dz, surface.u_min, surface.u_max))
        v.append(v[-1] + dv * dz)
    return np.array(u), np.array(v)


def _linear_init(surface, traj, z_eval, N, u0, v0):
    R_end = traj.R(traj.total_length)
    Z_end = float(R_end[2])
    u_min = float(surface.u_min)
    u_max = float(surface.u_max)
    u_full = np.linspace(float(u0), np.clip(Z_end, u_min, u_max), N)
    v_full = np.linspace(float(v0), float(v0) + 6 * np.pi, N)
    return u_full, v_full


# ==================== 5. Solver ====================

def solve_collocation_jax(surface, traj, u0, v0,
                          count_points=50,
                          w_Phi=1.0, w_diff=1.0, w_smooth=0.0,
                          init_method='radial',
                          X0_custom=None,
                          jac_mode='3-point',  # 'jax' | '3-point' | '2-point'
                          max_nfev=20000, tol=1e-8,
                          snapshot_path='best_snapshot.npz',
                          verbose=True):
    """
    JAX-ускоренная коллокация.

    jac_mode:
      'jax'     — exact Jacobian через jax.jacfwd (медленно для N>100, но точно)
      '3-point' — центральные разности (надёжно, по умолчанию)
      '2-point' — односторонние разности (быстрее, но менее точно)
    """
    if not HAS_JAX:
        raise RuntimeError("jax not installed. Run: pip install jax jaxlib")

    N = count_points
    z_eval = np.linspace(0, traj.total_length, N)
    u_min = float(surface.u_min)
    u_max = float(surface.u_max)
    scale_u = max(u_max - u_min, 1.0)
    scale_v = 2.0 * np.pi

    # --- Предвычисление данных ---
    R_data = build_jax_surface_data(surface, n_dense=2000)

    # Траектория ТСН на сетке
    traj_R = np.array([traj.R(zk) for zk in z_eval])
    traj_dR = np.array([traj.R_deriv(zk) for zk in z_eval])

    # --- Начальное приближение ---
    if X0_custom is not None:
        u_full = np.concatenate([[float(u0)], X0_custom[0::2]])
        v_full = np.concatenate([[float(v0)], X0_custom[1::2]])
    else:
        if init_method == 'radial':
            u_full, v_full = _radial_init(surface, traj, z_eval, N, u0, v0)
        elif init_method == 'dae':
            u_full, v_full = _dae_init(surface, traj, z_eval, N, u0, v0)
        else:
            u_full, v_full = _linear_init(surface, traj, z_eval, N, u0, v0)

    v_full = np.unwrap(v_full)

    X0 = np.zeros(2 * (N - 1))
    X0[0::2] = (u_full[1:] - u_min) / scale_u
    X0[1::2] = v_full[1:] / scale_v

    # --- Автобалансировка весов ---
    residual_fn = _make_residual_jax(
        z_eval, u0, v0, scale_u, scale_v,
        1.0, 1.0, 0.0, R_data, traj_R, traj_dR
    )
    residual_jit = jax.jit(residual_fn)

    F_test = np.array(residual_jit(jnp.array(X0)))
    phi_part = F_test[:N]
    diff_part = F_test[N:N + 2 * (N - 1)]
    phi_rms = np.sqrt(np.mean(phi_part ** 2)) if len(phi_part) > 0 else 1.0
    diff_rms = np.sqrt(np.mean(diff_part ** 2)) if len(diff_part) > 0 else 1.0

    w_Phi_eff = w_Phi / max(phi_rms, 1e-12)
    w_diff_eff = w_diff / max(diff_rms, 1e-12)
    w_smooth_eff = w_smooth / max(scale_u, 1.0)

    # --- Пересобираем с эффективными весами ---
    residual_fn = _make_residual_jax(
        z_eval, u0, v0, scale_u, scale_v,
        w_Phi_eff, w_diff_eff, w_smooth_eff, R_data, traj_R, traj_dR
    )
    residual_jit = jax.jit(residual_fn)

    # --- Jacobian ---
    if jac_mode == 'jax':
        jac_jax = jax.jit(jax.jacfwd(residual_fn, argnums=0))

        def jac_np(X):
            return np.array(jac_jax(jnp.array(X)), dtype=float)
    else:
        jac_np = jac_mode  # '2-point' или '3-point' — scipy сам разберётся

    # --- Snapshot state ---
    best_snapshot = {'max_Phi': np.inf, 'X': None, 'iter': 0}
    eval_counter = [0]

    def residual_logged(X_tail):
        eval_counter[0] += 1
        F = np.array(residual_jit(jnp.array(X_tail)), dtype=float)

        # --- SNAPSHOT LOGIC ---
        Phi_part = F[:N]
        max_Phi = np.max(np.abs(Phi_part))
        if max_Phi < best_snapshot['max_Phi']:
            best_snapshot['max_Phi'] = max_Phi
            best_snapshot['X'] = X_tail.copy()
            best_snapshot['iter'] = eval_counter[0]
            # Сохраняем на диск немедленно
            np.savez(snapshot_path,
                     X_tail=X_tail,
                     max_Phi=float(max_Phi),
                     iter=int(eval_counter[0]),
                     z_eval=z_eval,
                     u0=float(u0), v0=float(v0),
                     scale_u=float(scale_u), scale_v=float(scale_v),
                     weights={'w_Phi': float(w_Phi_eff),
                              'w_diff': float(w_diff_eff),
                              'w_smooth': float(w_smooth_eff)},
                     N=int(N))

        if verbose and (eval_counter[0] % 50 == 0 or eval_counter[0] == 1):
            diff_part = F[N:N + 2 * (N - 1)]
            smooth_part = F[N + 2 * (N - 1):] if len(F) > N + 2 * (N - 1) else np.array([])
            msg = (f"  Eval {eval_counter[0]:4d}: |F|={np.linalg.norm(F):.3e}  "
                   f"|Phi|={np.linalg.norm(Phi_part):.3e}  "
                   f"|diff|={np.linalg.norm(diff_part):.3e}")
            if len(smooth_part) > 0:
                msg += f"  |smooth|={np.linalg.norm(smooth_part):.3e}"
            print(msg)

        return F

    # --- Bounds ---
    lb = np.full(2 * (N - 1), -np.inf)
    ub = np.full(2 * (N - 1), np.inf)
    lb[0::2] = 0.0
    ub[0::2] = 1.0

    if verbose:
        print(f"[JAX] N={N}, init={init_method}, jac={jac_mode}")
        print(f"  scale_u={scale_u:.2f}, scale_v={scale_v:.4f}")
        print(f"  Auto-weights: w_Phi={w_Phi_eff:.3e}, w_diff={w_diff_eff:.3e}, w_smooth={w_smooth_eff:.3e}")

    # --- Optimize ---
    t0 = time.time()
    sol = least_squares(
        residual_logged,
        X0,
        jac=jac_np if jac_mode == 'jax' else jac_mode,
        method='trf',
        bounds=(lb, ub),
        max_nfev=max_nfev,
        ftol=tol, xtol=tol, gtol=tol,
        verbose=0
    )
    t1 = time.time()

    # --- Восстановление лучшего решения ---
    if best_snapshot['X'] is not None:
        final_Phi = np.max(np.abs(sol.fun[:N]))
        snap_Phi = best_snapshot['max_Phi']
        if snap_Phi < final_Phi:
            print(f"\n[SNAPSHOT] Используем лучший снимок (iter {best_snapshot['iter']}, "
                  f"max|Phi|={snap_Phi:.2e}) вместо финального ({final_Phi:.2e})")
            X_opt = best_snapshot['X']
        else:
            X_opt = sol.x
    else:
        X_opt = sol.x

    # --- Физические переменные ---
    u_opt = np.concatenate([[float(u0)], u_min + X_opt[0::2] * scale_u])
    v_opt = np.concatenate([[float(v0)], X_opt[1::2] * scale_v])

    # --- Верификация ---
    Phi_vals = np.zeros(N)
    points_3d = np.zeros((N, 3))
    for k in range(N):
        r = surface.position(u_opt[k], v_opt[k])
        m = surface.normal(u_opt[k], v_opt[k])
        Phi_vals[k] = float(np.dot(traj.R(z_eval[k]) - r, m))
        points_3d[k] = r

    return {
        'z_eval': z_eval,
        'u': u_opt,
        'v': v_opt,
        'Phi': Phi_vals,
        'points_3d': points_3d,
        'res_norm': float(np.linalg.norm(sol.fun)),
        'nfev': sol.nfev,
        'success': sol.success,
        'time': t1 - t0,
        'best_snapshot': best_snapshot,
        'scale_u': scale_u,
        'scale_v': scale_v,
        'weights': {'w_Phi': w_Phi_eff, 'w_diff': w_diff_eff, 'w_smooth': w_smooth_eff},
        'snapshot_path': snapshot_path,
    }


# ==================== 6. Загрузка snapshot ====================

def load_snapshot(snapshot_path='best_snapshot.npz'):
    """
    Загружает сохранённый снимок для повторной визуализации или warm-start.
    """
    data = np.load(snapshot_path, allow_pickle=True)
    result = {
        'X_tail': data['X_tail'],
        'max_Phi': float(data['max_Phi']),
        'iter': int(data['iter']),
        'z_eval': data['z_eval'],
        'u0': float(data['u0']),
        'v0': float(data['v0']),
        'scale_u': float(data['scale_u']),
        'scale_v': float(data['scale_v']),
        'weights': data['weights'].item(),
        'N': int(data['N']),
    }
    return result