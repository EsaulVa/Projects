# helpers/inverse_collocation.py
import numpy as np
from scipy.optimize import least_squares
import warnings

try:
    from helpers.inverse_method_fixed import compute_dr_dz, newton_corrector
except ImportError:
    from helpers.inverse_method import compute_dr_dz, newton_corrector


def collocation_residual(X_tail, z_eval, surface, traj, u0, v0,
                         w_Phi=1.0, w_diff=1.0, w_smooth=0.0):
    """
    Вектор невязок. X_tail = [u1, v1, ..., u_{N-1}, v_{N-1}].
    u0, v0 фиксированы.
    """
    N = len(z_eval)
    u = np.concatenate([[float(u0)], X_tail[0::2]])
    v = np.concatenate([[float(v0)], X_tail[1::2]])
    res = []

    # --- A. Связь Φ = 0 ---
    for k in range(N):
        try:
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            Phi = float(np.dot(traj.R(z_eval[k]) - r, m))
        except Exception:
            Phi = 1e3
        res.append(w_Phi * Phi)

    # --- B. Дифференциальные уравнения (Hermite) ---
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

    # --- C. Гладкость ---
    if w_smooth > 0:
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


def solve_collocation(surface, traj, u0, v0,
                      count_points=50,
                      w_Phi=1.0, w_diff=1.0, w_smooth=0.0,
                      init_method='radial',
                      max_nfev=10000, tol=1e-8,
                      verbose=True):
    """
    Решает обратную задачу методом прямой коллокации.

    init_method:
      'radial'  — радиальная проекция ТСН на оправку + корректор Ньютона
      'dae'     — явный Эйлер через compute_dr_dz
      'linear'  — линейная интерполяция
    """
    N = count_points
    z_eval = np.linspace(0, traj.total_length, N)
    u_min = getattr(surface, 'u_min', -np.inf)
    u_max = getattr(surface, 'u_max', np.inf)

    # ---------- Генерация начального приближения ----------
    if init_method == 'radial':
        u = []
        v = []
        for k in range(N):
            R_k = traj.R(z_eval[k])

            # Грубая радиальная проекция: масштабируем XY до радиуса поверхности
            u_guess = float(R_k[2])
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

            # Преобразуем в (u, v) на поверхности
            try:
                u_proj, v_proj = surface.uv_from_point(r_proj)
            except Exception:
                u_proj = u_guess
                v_proj = v_guess

            # Корректор Ньютона: точное касание Φ = 0
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

            # Fallback: принимаем проекцию как есть
            u.append(np.clip(float(u_proj), u_min, u_max))
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

    X0 = np.zeros(2 * (N - 1))
    X0[0::2] = u_full[1:]
    X0[1::2] = v_full[1:]

    # Bounds
    lb = np.full(2 * (N - 1), -np.inf)
    ub = np.full(2 * (N - 1), np.inf)
    lb[0::2] = u_min
    ub[0::2] = u_max

    if verbose:
        print(f"[Collocation] N={N}, init={init_method}")
        print(f"  u0={u0:.3f}, v0={v0:.4f}")
        print(f"  u_end_guess={u_full[-1]:.3f}, v_end_guess={v_full[-1]:.3f}")
        print(f"  bounds u∈[{u_min:.1f}, {u_max:.1f}]")

    eval_counter = [0]

    def residual_logged(X_tail):
        eval_counter[0] += 1
        F = collocation_residual(X_tail, z_eval, surface, traj, u0, v0,
                                 w_Phi, w_diff, w_smooth)
        if eval_counter[0] % 50 == 0 or eval_counter[0] == 1:
            n_phi = N
            phi_part = F[:n_phi]
            diff_part = F[n_phi:n_phi + 2 * (N - 1)]
            print(f"  Eval {eval_counter[0]:4d}: |F|={np.linalg.norm(F):.3e}  "
                  f"|Phi|={np.linalg.norm(phi_part):.3e}  "
                  f"|diff|={np.linalg.norm(diff_part):.3e}")
        return F

    sol = least_squares(
        residual_logged,
        X0,
        method='dogbox',
        bounds=(lb, ub),
        max_nfev=max_nfev,
        ftol=tol, xtol=tol, gtol=tol,
        verbose=0
    )

    u_opt = np.concatenate([[float(u0)], sol.x[0::2]])
    v_opt = np.concatenate([[float(v0)], sol.x[1::2]])
    res_norm = float(np.linalg.norm(sol.fun))

    if verbose:
        print(f"[Collocation] TRF: |F|={res_norm:.3e}, nfev={sol.nfev}, success={sol.success}")

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
        'success': sol.success
    }
# # helpers/inverse_collocation.py
# import numpy as np
# from scipy.optimize import least_squares
# import warnings

# try:
#     from helpers.inverse_method_fixed import compute_dr_dz
# except ImportError:
#     from helpers.inverse_method import compute_dr_dz


# def collocation_residual(X_tail, z_eval, surface, traj, u0, v0,
#                          w_Phi=1.0, w_diff=1.0, w_smooth=0.0):
#     """
#     Вектор невязок. X_tail = [u1, v1, ..., u_{N-1}, v_{N-1}].
#     u0, v0 фиксированы.
#     """
#     N = len(z_eval)
#     u = np.concatenate([[float(u0)], X_tail[0::2]])
#     v = np.concatenate([[float(v0)], X_tail[1::2]])
#     res = []

#     # --- A. Связь Φ = 0 ---
#     for k in range(N):
#         try:
#             r = surface.position(u[k], v[k])
#             m = surface.normal(u[k], v[k])
#             Phi = float(np.dot(traj.R(z_eval[k]) - r, m))
#         except Exception:
#             Phi = 1e3
#         res.append(w_Phi * Phi)

#     # --- B. Дифференциальные уравнения (Hermite коллокация) ---
#     for k in range(N - 1):
#         dz = z_eval[k + 1] - z_eval[k]
#         if abs(dz) < 1e-12:
#             res.extend([0.0, 0.0])
#             continue

#         try:
#             du_k, dv_k = compute_dr_dz(surface, traj, u[k], v[k], z_eval[k])
#             du_next, dv_next = compute_dr_dz(surface, traj, u[k + 1], v[k + 1], z_eval[k + 1])
#         except Exception:
#             du_k = dv_k = du_next = dv_next = 0.0

#         u_mid = 0.5 * (u[k] + u[k + 1]) + (dz / 8.0) * (du_k - du_next)
#         v_mid = 0.5 * (v[k] + v[k + 1]) + (dz / 8.0) * (dv_k - dv_next)
#         z_mid = 0.5 * (z_eval[k] + z_eval[k + 1])

#         du_spline = 1.5 * (u[k + 1] - u[k]) / dz - 0.25 * (du_k + du_next)
#         dv_spline = 1.5 * (v[k + 1] - v[k]) / dz - 0.25 * (dv_k + dv_next)

#         try:
#             du_rhs, dv_rhs = compute_dr_dz(surface, traj, u_mid, v_mid, z_mid)
#         except Exception:
#             du_rhs = dv_rhs = 0.0

#         res.append(w_diff * (du_spline - du_rhs))
#         res.append(w_diff * (dv_spline - dv_rhs))

#     # --- C. Гладкость (опционально) ---
#     if w_smooth > 0:
#         for k in range(N - 1):
#             try:
#                 E, F, G = surface.first_fundamental_form(u[k], v[k])
#                 du = u[k + 1] - u[k]
#                 dv = v[k + 1] - v[k]
#                 ds = np.sqrt(max(E * du ** 2 + 2 * F * du * dv + G * dv ** 2, 0.0))
#             except Exception:
#                 ds = 0.0
#             res.append(w_smooth * ds)

#     return np.array(res, dtype=float)


# def solve_collocation(surface, traj, u0, v0,
#                       count_points=50,
#                       w_Phi=1.0, w_diff=1.0, w_smooth=0.0,
#                       init_method='dae',
#                       ray_tracer=None,
#                       max_nfev=10000, tol=1e-8,
#                       verbose=True):
#     """
#     Решает обратную задачу методом прямой коллокации.
    
#     init_method:
#       'dae'     — явный Эйлер через compute_dr_dz с клиппингом
#       'linear'  — линейная интерполяция u и v
#       'optical' — трассировка лучей от R(z_k) к предыдущей точке на поверхности
#     """
#     N = count_points
#     z_eval = np.linspace(0, traj.total_length, N)
#     u_min = getattr(surface, 'u_min', -np.inf)
#     u_max = getattr(surface, 'u_max', np.inf)

#     # ---------- Генерация начального приближения ----------
#     if init_method == 'optical':
#         if ray_tracer is None:
#             from helpers.fixed_intersections import FixedRobustRevolutionIntersection
#             from helpers.intersection import RayTracer
#             ray_tracer = RayTracer()
#             ray_tracer.register(type(surface), FixedRobustRevolutionIntersection())

#         u = [float(u0)]
#         v = [float(v0)]

#         for k in range(1, N):
#             origin = traj.R(z_eval[k])
#             r_prev = surface.position(u[-1], v[-1])
#             direction = r_prev - origin
#             dist = np.linalg.norm(direction)

#             if dist > 1e-6:
#                 direction = direction / dist
#                 try:
#                     t, r_hit = ray_tracer.trace(
#                         surface, origin, direction,
#                         t_min=1e-3, t_max=5000.0
#                     )
#                     if t is not None:
#                         u_hit, v_hit = surface.uv_from_point(r_hit)
#                         u_new = np.clip(float(u_hit), u_min, u_max)
#                         v_new = float(v_hit)
#                         u.append(u_new)
#                         v.append(v_new)
#                         continue
#                 except Exception:
#                     pass

#             # Fallback: DAE-шаг если оптика не сработала
#             dz = z_eval[k] - z_eval[k - 1]
#             try:
#                 du, dv = compute_dr_dz(surface, traj, u[-1], v[-1], z_eval[k - 1])
#             except Exception:
#                 du = dv = 0.0
#             u.append(np.clip(u[-1] + du * dz, u_min, u_max))
#             v.append(v[-1] + dv * dz)

#         u_full = np.array(u)
#         v_full = np.array(v)

#     elif init_method == 'dae':
#         u = [float(u0)]
#         v = [float(v0)]
#         for k in range(1, N):
#             dz = z_eval[k] - z_eval[k - 1]
#             try:
#                 du, dv = compute_dr_dz(surface, traj, u[-1], v[-1], z_eval[k - 1])
#             except Exception:
#                 du = dv = 0.0
#             u.append(np.clip(u[-1] + du * dz, u_min, u_max))
#             v.append(v[-1] + dv * dz)
#         u_full = np.array(u)
#         v_full = np.array(v)

#     else:  # linear
#         R_end = traj.R(traj.total_length)
#         Z_end = float(R_end[2])
#         u_full = np.linspace(float(u0), np.clip(Z_end, u_min, u_max), N)
#         v_full = np.linspace(float(v0), float(v0) + 2 * np.pi, N)

#     X0 = np.zeros(2 * (N - 1))
#     X0[0::2] = u_full[1:]
#     X0[1::2] = v_full[1:]

#     # Bounds: u внутри [u_min, u_max], v без ограничений
#     lb = np.full(2 * (N - 1), -np.inf)
#     ub = np.full(2 * (N - 1), np.inf)
#     lb[0::2] = u_min
#     ub[0::2] = u_max

#     if verbose:
#         print(f"[Collocation] N={N}, init={init_method}")
#         print(f"  u0={u0:.3f}, v0={v0:.4f}")
#         print(f"  u_end_guess={u_full[-1]:.3f}, v_end_guess={v_full[-1]:.3f}")
#         print(f"  bounds u∈[{u_min:.1f}, {u_max:.1f}]")

#     eval_counter = [0]

#     def residual_logged(X_tail):
#         eval_counter[0] += 1
#         F = collocation_residual(X_tail, z_eval, surface, traj, u0, v0,
#                                  w_Phi, w_diff, w_smooth)
#         if eval_counter[0] % 50 == 0 or eval_counter[0] == 1:
#             n_phi = N
#             phi_part = F[:n_phi]
#             diff_part = F[n_phi:n_phi + 2 * (N - 1)]
#             print(f"  Eval {eval_counter[0]:4d}: |F|={np.linalg.norm(F):.3e}  "
#                   f"|Phi|={np.linalg.norm(phi_part):.3e}  "
#                   f"|diff|={np.linalg.norm(diff_part):.3e}")
#         return F

#     sol = least_squares(
#         residual_logged,
#         X0,
#         method='trf',
#         bounds=(lb, ub),
#         max_nfev=max_nfev,
#         ftol=tol, xtol=tol, gtol=tol,
#         verbose=0
#     )

#     u_opt = np.concatenate([[float(u0)], sol.x[0::2]])
#     v_opt = np.concatenate([[float(v0)], sol.x[1::2]])
#     res_norm = float(np.linalg.norm(sol.fun))

#     if verbose:
#         print(f"[Collocation] TRF: |F|={res_norm:.3e}, nfev={sol.nfev}, success={sol.success}")

#     Phi_vals = np.zeros(N)
#     points_3d = np.zeros((N, 3))
#     for k in range(N):
#         r = surface.position(u_opt[k], v_opt[k])
#         m = surface.normal(u_opt[k], v_opt[k])
#         Phi_vals[k] = float(np.dot(traj.R(z_eval[k]) - r, m))
#         points_3d[k] = r

#     return {
#         'z_eval': z_eval,
#         'u': u_opt,
#         'v': v_opt,
#         'Phi': Phi_vals,
#         'points_3d': points_3d,
#         'res_norm': res_norm,
#         'nfev': sol.nfev,
#         'success': sol.success
#     }
# # helpers/inverse_collocation.py
# import numpy as np
# from scipy.optimize import least_squares
# import warnings

# try:
#     from helpers.inverse_method_fixed import compute_dr_dz
# except ImportError:
#     from helpers.inverse_method import compute_dr_dz


# def collocation_residual(X_tail, z_eval, surface, traj, u0, v0,
#                          w_Phi=1.0, w_diff=1.0, w_smooth=0.0):
#     N = len(z_eval)
#     u = np.concatenate([[float(u0)], X_tail[0::2]])
#     v = np.concatenate([[float(v0)], X_tail[1::2]])
#     res = []

#     # A. Связь Φ = 0
#     for k in range(N):
#         try:
#             r = surface.position(u[k], v[k])
#             m = surface.normal(u[k], v[k])
#             Phi = float(np.dot(traj.R(z_eval[k]) - r, m))
#         except Exception:
#             Phi = 1e3
#         res.append(w_Phi * Phi)

#     # B. Дифференциальные уравнения (Hermite)
#     for k in range(N - 1):
#         dz = z_eval[k + 1] - z_eval[k]
#         if abs(dz) < 1e-12:
#             res.extend([0.0, 0.0])
#             continue

#         try:
#             du_k, dv_k = compute_dr_dz(surface, traj, u[k], v[k], z_eval[k])
#             du_next, dv_next = compute_dr_dz(surface, traj, u[k + 1], v[k + 1], z_eval[k + 1])
#         except Exception:
#             du_k = dv_k = du_next = dv_next = 0.0

#         u_mid = 0.5 * (u[k] + u[k + 1]) + (dz / 8.0) * (du_k - du_next)
#         v_mid = 0.5 * (v[k] + v[k + 1]) + (dz / 8.0) * (dv_k - dv_next)
#         z_mid = 0.5 * (z_eval[k] + z_eval[k + 1])

#         du_spline = 1.5 * (u[k + 1] - u[k]) / dz - 0.25 * (du_k + du_next)
#         dv_spline = 1.5 * (v[k + 1] - v[k]) / dz - 0.25 * (dv_k + dv_next)

#         try:
#             du_rhs, dv_rhs = compute_dr_dz(surface, traj, u_mid, v_mid, z_mid)
#         except Exception:
#             du_rhs = dv_rhs = 0.0

#         res.append(w_diff * (du_spline - du_rhs))
#         res.append(w_diff * (dv_spline - dv_rhs))

#     # C. Гладкость
#     if w_smooth > 0:
#         for k in range(N - 1):
#             try:
#                 E, F, G = surface.first_fundamental_form(u[k], v[k])
#                 du = u[k + 1] - u[k]
#                 dv = v[k + 1] - v[k]
#                 ds = np.sqrt(max(E * du ** 2 + 2 * F * du * dv + G * dv ** 2, 0.0))
#             except Exception:
#                 ds = 0.0
#             res.append(w_smooth * ds)

#     return np.array(res, dtype=float)


# def solve_collocation(surface, traj, u0, v0,
#                       count_points=50,
#                       w_Phi=1.0, w_diff=1.0, w_smooth=0.0,
#                       init_method='dae',
#                       max_nfev=10000, tol=1e-8,
#                       verbose=True):
#     N = count_points
#     z_eval = np.linspace(0, traj.total_length, N)

#     u_min = getattr(surface, 'u_min', -np.inf)
#     u_max = getattr(surface, 'u_max', np.inf)

#     # --- Генерация начального приближения с КЛИППИНГОМ ---
#     if init_method == 'dae':
#         u = [float(u0)]
#         v = [float(v0)]
#         for k in range(1, N):
#             dz = z_eval[k] - z_eval[k - 1]
#             try:
#                 du, dv = compute_dr_dz(surface, traj, u[-1], v[-1], z_eval[k - 1])
#             except Exception:
#                 du = dv = 0.0
#             u_new = np.clip(u[-1] + du * dz, u_min, u_max)
#             v_new = v[-1] + dv * dz
#             u.append(u_new)
#             v.append(v_new)
#         u_full = np.array(u)
#         v_full = np.array(v)
#     else:
#         R_end = traj.R(traj.total_length)
#         Z_end = float(R_end[2])
#         u_full = np.linspace(float(u0), np.clip(Z_end, u_min, u_max), N)
#         v_full = np.linspace(float(v0), float(v0) + 2 * np.pi, N)

#     X0 = np.zeros(2 * (N - 1))
#     X0[0::2] = u_full[1:]
#     X0[1::2] = v_full[1:]

#     # Bounds: u внутри [u_min, u_max], v без ограничений
#     lb = np.full(2 * (N - 1), -np.inf)
#     ub = np.full(2 * (N - 1), np.inf)
#     lb[0::2] = u_min
#     ub[0::2] = u_max

#     if verbose:
#         print(f"[Collocation] N={N}, init={init_method}")
#         print(f"  u0={u0:.3f}, v0={v0:.4f}")
#         print(f"  u_end_guess={u_full[-1]:.3f}, v_end_guess={v_full[-1]:.3f}")
#         print(f"  bounds u∈[{u_min:.1f}, {u_max:.1f}]")

#     eval_counter = [0]

#     def residual_logged(X_tail):
#         eval_counter[0] += 1
#         F = collocation_residual(X_tail, z_eval, surface, traj, u0, v0,
#                                  w_Phi, w_diff, w_smooth)
#         if eval_counter[0] % 50 == 0 or eval_counter[0] == 1:
#             n_phi = N
#             phi_part = F[:n_phi]
#             diff_part = F[n_phi:n_phi + 2 * (N - 1)]
#             print(f"  Eval {eval_counter[0]:4d}: |F|={np.linalg.norm(F):.3e}  "
#                   f"|Phi|={np.linalg.norm(phi_part):.3e}  "
#                   f"|diff|={np.linalg.norm(diff_part):.3e}")
#         return F

#     sol = least_squares(
#         residual_logged,
#         X0,
#         method='trf',
#         bounds=(lb, ub),
#         max_nfev=max_nfev,
#         ftol=tol, xtol=tol, gtol=tol,
#         verbose=0
#     )

#     u_opt = np.concatenate([[float(u0)], sol.x[0::2]])
#     v_opt = np.concatenate([[float(v0)], sol.x[1::2]])
#     res_norm = float(np.linalg.norm(sol.fun))

#     if verbose:
#         print(f"[Collocation] TRF: |F|={res_norm:.3e}, nfev={sol.nfev}, success={sol.success}")

#     Phi_vals = np.zeros(N)
#     points_3d = np.zeros((N, 3))
#     for k in range(N):
#         r = surface.position(u_opt[k], v_opt[k])
#         m = surface.normal(u_opt[k], v_opt[k])
#         Phi_vals[k] = float(np.dot(traj.R(z_eval[k]) - r, m))
#         points_3d[k] = r

#     return {
#         'z_eval': z_eval,
#         'u': u_opt,
#         'v': v_opt,
#         'Phi': Phi_vals,
#         'points_3d': points_3d,
#         'res_norm': res_norm,
#         'nfev': sol.nfev,
#         'success': sol.success
#     }