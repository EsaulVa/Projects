# inverse_winding_intermediate.py
import numpy as np
from typing import Optional,Dict,Any
from geometry.fixed_surfaces import get_surface_height_bounds
from helpers.dae_predictor import DAEPredictor
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method import (
    compute_dr_dz, newton_corrector, recompute_thread_geometry,
    normal_curvature
)
from geometry.tsurfaces import FixedPointTrajectory

# inverse_winding_hybrid.py
import numpy as np
from typing import Optional, Tuple
from helpers.predictor_base import Predictor
from helpers.inverse_method import (
    compute_dr_dz, newton_corrector, recompute_thread_geometry,
    normal_curvature
)

from core.exceptions import GeometryOutOfBoundsError
# helpers/inverse_winding_intermediate.py (только функция inverse_winding_hybrid, без лишних импортов)
import numpy as np
from typing import Optional, Dict, Any
from helpers.predictor_base import Predictor
from helpers.inverse_method import (
    compute_dr_dz, newton_corrector, recompute_thread_geometry,
    normal_curvature
)
from core.exceptions import GeometryOutOfBoundsError

# inverse_winding_hybrid.py
import numpy as np
from typing import Optional, Dict, Any
from geometry.fixed_surfaces import get_surface_height_bounds
from helpers.predictor_base import Predictor
from helpers.inverse_method import (
    compute_dr_dz, newton_corrector, recompute_thread_geometry,
    normal_curvature
)
from core.exceptions import GeometryOutOfBoundsError


def inverse_winding_hybrid(
    surface: Any,
    traj: Any,
    u0: float,
    v0: float,
    count_points: int = 300,
    eps_Phi: float = 1e-10,
    max_newton: int = 20,
    max_bisect: int = 4,
    jump_threshold: float = 3.0,
    predictor_dae: Optional[Predictor] = None,
    predictor_optical: Optional[Predictor] = None,
    eps_kappa: float = 1e-4,
    u_margin: float = 0.01,
    force_optical_after_fail: bool = True
) -> Dict[str, Any]:
    """
    Гибридная обратная задача намотки нити на оправку.
    Предиктор DAE + оптический fallback + корректор Ньютона + адаптивная бисекция.
    """
    if predictor_dae is None:
        from solvers.scipy_solver import SciPySolver
        from helpers.dae_predictor import DAEPredictor
        solver = SciPySolver(method='RK45', rtol=1e-8, atol=1e-10)
        predictor_dae = DAEPredictor(solver)

    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)

    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N)
    flags = np.zeros(N, dtype=int)

    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0
    use_optical_next = False

    # Универсальные аксиальные границы (кешируем, т.к. поверхность не меняется)
    z_min, z_max = get_surface_height_bounds(surface)

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]

        # --- 1. Геометрия в текущей точке ---
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k)
        except (ValueError, GeometryOutOfBoundsError) as e:
            print(f"Шаг {i}: ошибка геометрии: {e}")
            break

        lam_hist[i] = lam_k
        kappa_n = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)
        kappa_n_hist[i] = kappa_n

        # --- 2. Выбор предиктора ---
        # Универсальная проверка близости к аксиальным границам
        near_boundary = (u_cur < z_min + u_margin) or (u_cur > z_max - u_margin)
        need_optical = (
            (abs(kappa_n) < eps_kappa) or
            near_boundary or
            use_optical_next
        )

        predictor = predictor_dae if not need_optical else predictor_optical
        pred_result = predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj) if predictor else None

        if pred_result is not None:
            u_pred, v_pred = pred_result
        else:
            u_pred = v_pred = None

        # Fallback: если DAE не дал предсказания, пробуем оптику явно
        if u_pred is None and predictor is predictor_dae and predictor_optical is not None:
            opt_result = predictor_optical.predict(z_k, z_next, u_cur, v_cur, surface, traj)
            if opt_result is not None:
                u_pred, v_pred = opt_result
                flags[i + 1] = 2

        # --- 3. Корректор Ньютона (с универсальными границами) ---
        success = False
        if u_pred is not None:
            if u_pred < z_min or u_pred > z_max:
                # Предсказание вылетело за высотную границу — не пускаем корректор
                # в неопределённую зону, сразу fallback
                u_pred = np.clip(u_pred, z_min, z_max)
                success = False
                use_optical_next = True
            else:
                try:
                    u_c, v_c, Phi_c, nit, conv = newton_corrector(
                        surface, traj, u_pred, v_pred, z_next,
                        eps_Phi=eps_Phi, max_iter=max_newton
                    )
                    if conv and abs(Phi_c) < eps_Phi:
                        # Проверка скачка
                        du_j = u_c - u_cur
                        dv_j = v_c - v_cur
                        Ej, Fj, Gj = surface.first_fundamental_form(u_cur, v_cur)
                        ds_actual = np.sqrt(
                            max(Ej * du_j**2 + 2 * Fj * du_j * dv_j + Gj * dv_j**2, 0.0)
                        )
                        try:
                            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
                            E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
                            speed_expect = np.sqrt(
                                max(E_f * du_dz_k**2
                                    + 2 * F_f * du_dz_k * dv_dz_k
                                    + G_f * dv_dz_k**2, 0.0)
                            )
                            ds_expect = speed_expect * (z_next - z_k)
                            ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0

                            if ratio > jump_threshold or ratio < 1.0 / jump_threshold:
                                pass  # скачок — считаем неудачей
                            else:
                                u_cur, v_cur = u_c, v_c
                                u_hist[i + 1], v_hist[i + 1] = u_cur, v_cur
                                Phi_hist[i + 1] = Phi_c
                                newton_iters_hist[i + 1] = nit
                                use_optical_next = False
                                success = True
                        except (ValueError, GeometryOutOfBoundsError):
                            pass
                except GeometryOutOfBoundsError:
                    pass

        # --- 4. Fallback: явный Эйлер + адаптивная бисекция ---
        if not success:
            flags[i + 1] = 4 if u_pred is not None else 3

            try:
                du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
            except (ValueError, GeometryOutOfBoundsError):
                du_dz_k, dv_dz_k = 0.0, 0.0

            E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
            speed_expected = np.sqrt(
                max(E_f * du_dz_k**2
                    + 2 * F_f * du_dz_k * dv_dz_k
                    + G_f * dv_dz_k**2, 0.0)
            )

            n_sub = 1
            best_u, best_v, best_Phi, best_nit = u_cur, v_cur, 1.0, 0
            bisected = False

            for bisect_level in range(max_bisect + 1):
                sub_z = np.linspace(z_k, z_next, n_sub + 1)
                u_s, v_s = u_cur, v_cur
                total_nit = 0
                jump_detected = False

                try:
                    for j in range(n_sub):
                        z_a, z_b = sub_z[j], sub_z[j + 1]
                        dz_sub = z_b - z_a

                        try:
                            du_s, dv_s = compute_dr_dz(surface, traj, u_s, v_s, z_a)
                        except (ValueError, GeometryOutOfBoundsError):
                            du_s, dv_s = du_dz_k, dv_dz_k

                        u_p = u_s + du_s * dz_sub
                        v_p = v_s + dv_s * dz_sub

                        # --- ПАТЧ: жёсткий клиппинг высоты, чтобы не уйти в неопределённость ---
                        u_p = np.clip(u_p, z_min, z_max)

                        u_c, v_c, Phi_c, nit, conv = newton_corrector(
                            surface, traj, u_p, v_p, z_b,
                            eps_Phi=eps_Phi, max_iter=max_newton
                        )
                        total_nit += nit

                        du_j = u_c - u_s
                        dv_j = v_c - v_s
                        Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                        ds_actual = np.sqrt(
                            max(Ej * du_j**2 + 2 * Fj * du_j * dv_j + Gj * dv_j**2, 0.0)
                        )
                        ds_expect = speed_expected * dz_sub
                        ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0

                        if (ratio > jump_threshold or ratio < 1.0 / jump_threshold) and bisect_level < max_bisect:
                            jump_detected = True
                            break

                        u_s, v_s = u_c, v_c

                except (ValueError, GeometryOutOfBoundsError):
                    jump_detected = True if bisect_level < max_bisect else False
                    if not jump_detected:
                        best_u, best_v = u_s, v_s
                        r_f = surface.position(best_u, best_v)
                        m_f = surface.normal(best_u, best_v)
                        best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                        best_nit = total_nit
                        break

                if not jump_detected:
                    best_u, best_v = u_s, v_s
                    r_f = surface.position(best_u, best_v)
                    m_f = surface.normal(best_u, best_v)
                    best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                    best_nit = total_nit
                    if bisect_level > 0:
                        bisected = True
                    break
                else:
                    n_sub *= 2

            u_cur, v_cur = best_u, best_v
            u_hist[i + 1], v_hist[i + 1] = u_cur, v_cur
            Phi_hist[i + 1] = best_Phi
            newton_iters_hist[i + 1] = best_nit
            flags[i + 1] = 1 if bisected else flags[i + 1]
            use_optical_next = force_optical_after_fail

    # --- Финальная точка ---
    try:
        _, lam_last, up_last, vp_last = recompute_thread_geometry(
            surface, traj, u_hist[-1], v_hist[-1], z_eval[-1])
        lam_hist[-1] = lam_last
        kappa_n_hist[-1] = normal_curvature(
            surface, u_hist[-1], v_hist[-1], up_last, vp_last)
    except (ValueError, GeometryOutOfBoundsError):
        pass

    points_3d = np.array([surface.position(u_hist[k], v_hist[k]) for k in range(N)])
    return {
        'z_eval': z_eval,
        'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }


def inverse_winding_intermediate(
    surface, traj, u0, v0, count_points=300,
    eps_Phi=1e-10, max_newton=7, max_bisect=4, jump_threshold=3.0,
    predictor: Optional[DAEPredictor] = None
):
    if predictor is None:
        # По умолчанию создаём предиктор с Рунге-Куттой 5(4)
        solver = SciPySolver(method='RK45', rtol=1e-8, atol=1e-10)
        predictor = DAEPredictor(solver)

    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)
    u_hist, v_hist = np.zeros(N), np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N)
    flags = np.zeros(N, dtype=int)

    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]

        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k)
        except ValueError as e:
            print(f"Шаг {i}: {e}")
            break

        lam_hist[i] = lam_k
        kappa_n_hist[i] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)

        # --- Предиктор (DAE) ---
        u_pred, v_pred = predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj)

        if u_pred is not None:
            u_c, v_c, Phi_c, nit, conv = newton_corrector(
                surface, traj, u_pred, v_pred, z_next,
                eps_Phi=eps_Phi, max_iter=max_newton)

            if conv and abs(Phi_c) < eps_Phi:
                # Успех: проверяем скачок
                du_j = u_c - u_cur
                dv_j = v_c - v_cur
                Ej, Fj, Gj = surface.first_fundamental_form(u_cur, v_cur)
                ds_actual = np.sqrt(max(Ej*du_j**2 + 2*Fj*du_j*dv_j + Gj*dv_j**2, 0.0))
                try:
                    du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
                    E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
                    speed_expect = np.sqrt(max(E_f*du_dz_k**2 + 2*F_f*du_dz_k*dv_dz_k + G_f*dv_dz_k**2, 0.0))
                    ds_expect = speed_expect * (z_next - z_k)
                    ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0
                    if ratio > jump_threshold or ratio < 1.0/jump_threshold:
                        # скачок – идём на бисекцию
                        pass
                    else:
                        u_cur, v_cur = u_c, v_c
                        u_hist[i+1], v_hist[i+1] = u_cur, v_cur
                        Phi_hist[i+1] = Phi_c
                        newton_iters_hist[i+1] = nit
                        continue
                except ValueError:
                    pass

        # Если DAE-предиктор не сработал, используем явный Эйлер с бисекцией
        try:
            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
        except ValueError:
            du_dz_k, dv_dz_k = 0.0, 0.0

        E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
        speed_expected = np.sqrt(max(E_f*du_dz_k**2 + 2*F_f*du_dz_k*dv_dz_k + G_f*dv_dz_k**2, 0.0))

        n_sub = 1
        best_u, best_v, best_Phi, best_nit = u_cur, v_cur, 1.0, 0
        bisected = False

        for bisect_level in range(max_bisect + 1):
            sub_z = np.linspace(z_k, z_next, n_sub + 1)
            u_s, v_s = u_cur, v_cur
            total_nit, jump_detected = 0, False
            for j in range(n_sub):
                z_a, z_b = sub_z[j], sub_z[j+1]
                dz_sub = z_b - z_a
                try:
                    du_s, dv_s = compute_dr_dz(surface, traj, u_s, v_s, z_a)
                except ValueError:
                    du_s, dv_s = du_dz_k, dv_dz_k
                # u_p = np.clip(u_s + du_s*dz_sub, surface.u_min, surface.u_max)
                u_p=u_s + du_s*dz_sub
                v_p = v_s + dv_s*dz_sub
                u_c, v_c, Phi_c, nit, conv = newton_corrector(
                    surface, traj, u_p, v_p, z_b,
                    eps_Phi=eps_Phi, max_iter=max_newton)
                total_nit += nit
                du_j, dv_j = u_c - u_s, v_c - v_s
                Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                ds_actual = np.sqrt(max(Ej*du_j**2 + 2*Fj*du_j*dv_j + Gj*dv_j**2, 0.0))
                ds_expect = speed_expected * dz_sub
                ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0
                if (ratio > jump_threshold or ratio < 1.0/jump_threshold) and bisect_level < max_bisect:
                    jump_detected = True
                    break
                u_s, v_s = u_c, v_c
            if not jump_detected:
                best_u, best_v = u_s, v_s
                r_f = surface.position(best_u, best_v)
                m_f = surface.normal(best_u, best_v)
                best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                best_nit = total_nit
                if bisect_level > 0: bisected = True
                break
            else:
                n_sub *= 2

        u_cur, v_cur = best_u, best_v
        u_hist[i+1], v_hist[i+1] = u_cur, v_cur
        Phi_hist[i+1] = best_Phi
        newton_iters_hist[i+1] = best_nit
        flags[i+1] = 1 if bisected else 0

    # Финальная точка
    try:
        _, lam_last, up_last, vp_last = recompute_thread_geometry(
            surface, traj, u_hist[-1], v_hist[-1], z_eval[-1])
        lam_hist[-1] = lam_last
        kappa_n_hist[-1] = normal_curvature(surface, u_hist[-1], v_hist[-1], up_last, vp_last)
    except ValueError:
        pass

    points_3d = np.array([surface.position(u_hist[k], v_hist[k]) for k in range(N)])
    return {
        'z_eval': z_eval, 'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }