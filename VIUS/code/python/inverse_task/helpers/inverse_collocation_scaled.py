# helpers/inverse_collocation_scaled.py
import numpy as np
from scipy.optimize import least_squares

try:
    from helpers.inverse_method_fixed import compute_dr_dz, newton_corrector
except ImportError:
    from helpers.inverse_method import compute_dr_dz, newton_corrector


def collocation_residual_scaled(X_tail_tilde, z_eval, surface, traj,
                                u0, v0, scale_u, scale_v,
                                w_Phi=1.0, w_diff=1.0, w_smooth=0.0):
    """
    Вектор невязок с масштабированными переменными.
    Преобразование:
        u = u_min + u_tilde * scale_u
        v = v_tilde * scale_v
    """
    N = len(z_eval)
    u_min = getattr(surface, 'u_min', -np.inf)

    # --- Обратное масштабирование ---
    u_tilde = np.concatenate([[0.0], X_tail_tilde[0::2]])   # u0_tilde = 0  => u = u_min
    v_tilde = np.concatenate([[v0 / scale_v], X_tail_tilde[1::2]])

    u = u_min + u_tilde * scale_u
    v = v_tilde * scale_v

    res = []

    # --- A. Связь Phi = 0 ---
    for k in range(N):
        try:
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            Phi = float(np.dot(traj.R(z_eval[k]) - r, m))
        except Exception:
            Phi = 1e3
        res.append(w_Phi * Phi)

    # --- B. Дифференциальные уравнения (Hermite коллокация) ---
    for k in range(N - 1):
        dz = z_eval[k + 1] - z_eval[k]
        if abs(dz) < 1e-12:
            res.extend([0.0, 0.0])
            continue

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

        res.append(w_diff * (du_spline - du_rhs))
        res.append(w_diff * (dv_spline - dv_rhs))

    # --- C. Гладкость (опционально) ---
    if w_smooth > 0.0:
        for k in range(N - 1):
            try:
                E, F, G = surface.first_fundamental_form(u[k], v[k])
                du = u[k + 1] - u[k]
                dv = v[k + 1] - v[k]
                ds = np.sqrt(max(E * du ** 2 + 2 * F * du * dv + G * dv ** 2, 0.0))
            except Exception:
                ds = 0.0
            res.append(w_smooth * ds)

    return np.array(res, dtype=float)


def solve_collocation_scaled(surface, traj, u0, v0,
                             count_points=50,
                             w_Phi=1.0, w_diff=1.0, w_smooth=0.0,
                             init_method='radial',
                             X0_custom=None,
                             max_nfev=10000, tol=1e-8,
                             verbose=True):
    """
    Коллокация с МАСШТАБИРОВАНИЕМ и АВТОБАЛАНСИРОВКОЙ весов.
    """
    N = count_points
    z_eval = np.linspace(0, traj.total_length, N)

    u_min = getattr(surface, 'u_min', -np.inf)
    u_max = getattr(surface, 'u_max', np.inf)
    scale_u = max(u_max - u_min, 1.0)
    scale_v = 2.0 * np.pi          # характерный масштаб угла

    # ---------- Генерация начального приближения (физические переменные) ----------
    if X0_custom is not None:
        u_full = np.concatenate([[float(u0)], X0_custom[0::2]])
        v_full = np.concatenate([[float(v0)], X0_custom[1::2]])
    else:
        if init_method == 'radial':
            u = []
            v = []
            for k in range(N):
                R_k = traj.R(z_eval[k])
                u_guess = float(np.clip(R_k[2], u_min, u_max))
                v_guess = float(np.arctan2(R_k[1], R_k[0]))

                try:
                    r_surf = float(surface.radius(u_guess))
                except Exception:
                    r_surf = 0.0

                r_ray = float(np.hypot(R_k[0], R_k[1]))
                if r_ray > 1e-6:
                    scale = r_surf / r_ray
                    r_proj = np.array([R_k[0] * scale, R_k[1] * scale, R_k[2]])
                else:
                    r_proj = np.array([r_surf, 0.0, R_k[2]])

                try:
                    u_proj, v_proj = surface.uv_from_point(r_proj)
                except Exception:
                    u_proj = u_guess
                    v_proj = v_guess

                u_proj = np.clip(float(u_proj), u_min, u_max)
                try:
                    u_c, v_c, Phi_c, _, conv = newton_corrector(
                        surface, traj, u_proj, v_proj, z_eval[k],
                        eps_Phi=1e-8, max_iter=20
                    )
                    if conv and abs(Phi_c) < 1e-6:
                        u.append(float(u_c))
                        v.append(float(v_c))
                        continue
                except Exception:
                    pass

                u.append(u_proj)
                v.append(float(v_proj))

            u_full = np.array(u)
            v_full = np.array(v)

        elif init_method == 'dae':
            u = [float(u0)]
            v = [float(v0)]
            for k in range(1, N):
                dz = z_eval[k] - z_eval[k - 1]
                try:
                    du, dv = compute_dr_dz(surface, traj, u[-1], v[-1], z_eval[k - 1])
                except Exception:
                    du = dv = 0.0
                u.append(np.clip(u[-1] + du * dz, u_min, u_max))
                v.append(v[-1] + dv * dz)
            u_full = np.array(u)
            v_full = np.array(v)

        else:  # linear
            R_end = traj.R(traj.total_length)
            Z_end = float(R_end[2])
            u_full = np.linspace(float(u0), np.clip(Z_end, u_min, u_max), N)
            v_full = np.linspace(float(v0), float(v0) + 6 * np.pi, N)

    # --- Развертывание угла v (unwrap) — критично для сплайнов ---
    v_full = np.unwrap(v_full)

    # --- Переход в масштабированные координаты ---
    X0 = np.zeros(2 * (N - 1))
    X0[0::2] = (u_full[1:] - u_min) / scale_u      # u_tilde in [0, 1]
    X0[1::2] = v_full[1:] / scale_v                # v_tilde ~ безразмерный угол

    # Bounds в масштабированных координатах
    lb = np.full(2 * (N - 1), -np.inf)
    ub = np.full(2 * (N - 1),  np.inf)
    lb[0::2] = 0.0
    ub[0::2] = 1.0

    # --- АВТОБАЛАНСИРОВКА ВЕСОВ ---
    F_test = collocation_residual_scaled(
        X0, z_eval, surface, traj, u0, v0, scale_u, scale_v,
        w_Phi=1.0, w_diff=1.0, w_smooth=0.0
    )
    phi_part = F_test[:N]
    diff_part = F_test[N:N + 2 * (N - 1)]

    phi_rms = np.sqrt(np.mean(phi_part ** 2)) if len(phi_part) > 0 else 1.0
    diff_rms = np.sqrt(np.mean(diff_part ** 2)) if len(diff_part) > 0 else 1.0

    # Приводим начальные вклады к одному порядку (целевой RMS = 1)
    w_Phi_eff = w_Phi / max(phi_rms, 1e-12)
    w_diff_eff = w_diff / max(diff_rms, 1e-12)
    w_smooth_eff = w_smooth / max(scale_u, 1.0)   # нормируем на масштаб длины

    if verbose:
        print(f"[CollocationScaled] N={N}, init={init_method}")
        print(f"  scale_u={scale_u:.2f}, scale_v={scale_v:.4f}")
        print(f"  u0={u0:.3f}, v0={v0:.4f}")
        print(f"  Auto-weights: w_Phi={w_Phi_eff:.3e}, w_diff={w_diff_eff:.3e}, w_smooth={w_smooth_eff:.3e}")
        print(f"  Initial RMS: |Phi|={phi_rms:.3e}, |diff|={diff_rms:.3e}")

    eval_counter = [0]

    def residual_logged(X_tail_tilde):
        eval_counter[0] += 1
        F = collocation_residual_scaled(
            X_tail_tilde, z_eval, surface, traj, u0, v0,
            scale_u, scale_v,
            w_Phi_eff, w_diff_eff, w_smooth_eff
        )
        if eval_counter[0] % 50 == 0 or eval_counter[0] == 1:
            n_phi = N
            phi_part = F[:n_phi]
            diff_part = F[n_phi:n_phi + 2 * (N - 1)]
            smooth_part = F[n_phi + 2 * (N - 1):] if w_smooth > 0 else np.array([])
            msg = (f"  Eval {eval_counter[0]:4d}: |F|={np.linalg.norm(F):.3e}  "
                   f"|Phi|={np.linalg.norm(phi_part):.3e}  "
                   f"|diff|={np.linalg.norm(diff_part):.3e}")
            if len(smooth_part) > 0:
                msg += f"  |smooth|={np.linalg.norm(smooth_part):.3e}"
            print(msg)
        return F

    sol = least_squares(
        residual_logged,
        X0,
        method='trf',
        bounds=(lb, ub),
        max_nfev=max_nfev,
        ftol=tol, xtol=tol, gtol=tol,
        verbose=0
    )

    # --- Обратное преобразование в физические переменные ---
    u_opt = np.concatenate([[float(u0)], u_min + sol.x[0::2] * scale_u])
    v_opt = np.concatenate([[float(v0)], sol.x[1::2] * scale_v])
    res_norm = float(np.linalg.norm(sol.fun))

    if verbose:
        print(f"[CollocationScaled] TRF: |F|={res_norm:.3e}, nfev={sol.nfev}, success={sol.success}")

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
        'res_norm': res_norm,
        'nfev': sol.nfev,
        'success': sol.success,
        'scale_u': scale_u,
        'scale_v': scale_v,
        'weights': {'w_Phi': w_Phi_eff, 'w_diff': w_diff_eff, 'w_smooth': w_smooth_eff}
    }