#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helpers/inverse_winding_intermediate_adaptive.py
================================================
Гибридная обратная задача намотки с ПРОАКТИВНОЙ адаптивной сеткой.

Вместо фиксированной сетки np.linspace шаг dz адаптируется на лету
по критериям:
  • кривизна κ_n (мелкий шаг при |κ_n| < eps_kappa)
  • невязка Φ (мелкий шаг при |Φ| > eps_Phi * 10)
  • итерации Ньютона (мелкий шаг при nit > max_newton/2)
  • близость к границе (мелкий шаг у днища/крышки)
  • успешность предыдущего шага (можно увеличить при идеальном Φ)

На базе inverse_winding_intermediate.py.
"""

import numpy as np
import csv
import os
from typing import Optional, Dict, Any
from geometry.fixed_surfaces import get_surface_height_bounds
from helpers.predictor_base import Predictor
from helpers.inverse_method_fixed import (
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
    eps_kappa: float = 1e-2,
    u_margin: float = 0.01,
    force_optical_after_fail: bool = True,
    # --- параметры адаптивной сетки ---
    dz_min_factor: float = 0.1,      # dz_min = dz_init * dz_min_factor
    dz_max_factor: float = 2.0,      # dz_max = dz_init * dz_max_factor
    adapt_shrink: float = 0.5,       # во сколько раз уменьшать при проблеме
    adapt_grow: float = 1.2,         # во сколько раз увеличивать при успехе
    phi_warn_factor: float = 10.0,   # |Phi| > eps_Phi * phi_warn_factor -> мелкий шаг
    nit_warn_factor: float = 0.5,    # nit > max_newton * nit_warn_factor -> мелкий шаг
) -> Dict[str, Any]:
    """
    Гибридная обратная задача намотки нити на оправку с адаптивной сеткой.
    """
    if predictor_dae is None:
        from solvers.scipy_solver import SciPySolver
        from helpers.dae_predictor import DAEPredictor
        solver = SciPySolver(method='RK45', rtol=1e-8, atol=1e-10)
        predictor_dae = DAEPredictor(solver)

    total_length = traj.total_length
    dz_init = total_length / count_points
    dz_min = dz_init * dz_min_factor
    dz_max = dz_init * dz_max_factor
    dz = dz_init

    # Динамические списки (вместо фиксированных массивов)
    z_list = [0.0]
    u_list = [u0]
    v_list = [v0]
    Phi_list = [0.0]          # Phi[0] = 0 по определению (начальная точка скорректирована)
    kappa_n_list = [0.0]
    newton_iters_list = [0]
    lam_list = [0.0]
    flags_list = [0]

    u_cur, v_cur = u0, v0
    use_optical_next = False
    step_idx = 0

    surf_u_min = getattr(surface, 'u_min', -float('inf'))
    surf_u_max = getattr(surface, 'u_max', float('inf'))
    surf_v_min = getattr(surface, 'v_min', -float('inf'))
    surf_v_max = getattr(surface, 'v_max', float('inf'))

    # --- CSV-лог ---
    log_path = 'debug_log.csv'
    log_file = open(log_path, 'w', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'step', 'z_k', 'z_next', 'predictor', 'u_cur', 'v_cur',
        'u_pred', 'v_pred', 'Phi_before', 'Phi_after',
        'newton_iters', 'newton_conv', 'u_after', 'v_after',
        'kappa_n', 'flag', 'ds_ratio', 'bisect_level', 'dz', 'note'
    ])

    while z_list[-1] < total_length - 1e-12:
        z_k = z_list[-1]

        # --- АДАПТАЦИЯ ШАГА dz ---
        # Оцениваем сложность в текущей точке
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k)
            kappa_n = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)
        except (ValueError, GeometryOutOfBoundsError):
            kappa_n = 0.0
            lam_k = 0.0

        # Критерии "опасности"
        near_boundary = (u_cur < surf_u_min + u_margin) or (u_cur > surf_u_max - u_margin)

        # Phi_prev — невязка на текущей точке (последняя в списке)
        Phi_prev = abs(Phi_list[-1])
        nit_prev = newton_iters_list[-1]

        danger = False
        if abs(kappa_n) < eps_kappa:
            danger = True
            reason = 'kappa_low'
        elif Phi_prev > eps_Phi * phi_warn_factor:
            danger = True
            reason = 'phi_high'
        elif nit_prev > max_newton * nit_warn_factor:
            danger = True
            reason = 'nit_high'
        elif near_boundary:
            danger = True
            reason = 'near_boundary'
        elif use_optical_next:
            danger = True
            reason = 'prev_fail'

        if danger:
            dz = max(dz_min, dz * adapt_shrink)
        else:
            dz = min(dz_max, dz * adapt_grow)

        # Не выходим за total_length
        z_next = min(z_k + dz, total_length)
        if z_next - z_k < 1e-12:
            break
        dz_actual = z_next - z_k

        # --- 1. Геометрия в текущей точке ---
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k)
        except (ValueError, GeometryOutOfBoundsError) as e:
            print(f"Шаг {step_idx}: ошибка геометрии: {e}")
            log_file.close()
            break

        lam_list[-1] = lam_k
        kappa_n_list[-1] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)

        # --- 2. Выбор предиктора ---
        near_boundary = (u_cur < surf_u_min + u_margin) or (u_cur > surf_u_max - u_margin)
        need_optical = (
            (abs(kappa_n_list[-1]) < eps_kappa) or
            near_boundary or
            use_optical_next
        )

        predictor = predictor_dae if not need_optical else predictor_optical
        pred_type = 'OPTICAL' if need_optical else 'DAE'
        pred_result = predictor.predict(z_k, z_next, u_cur, v_cur, surface, traj) if predictor else None

        u_pred = v_pred = None
        if pred_result is not None:
            u_pred, v_pred = pred_result

        # Fallback: если DAE не дал предсказания, пробуем оптику явно
        if u_pred is None and predictor is predictor_dae and predictor_optical is not None:
            opt_result = predictor_optical.predict(z_k, z_next, u_cur, v_cur, surface, traj)
            if opt_result is not None:
                u_pred, v_pred = opt_result
                pred_type = 'OPTICAL_FALLBACK'
                flags_list[-1] = 2  # флаг для текущей точки (будет перезаписан при добавлении новой)

        # --- 3. Корректор Ньютона ---
        success = False
        Phi_before = np.nan
        Phi_after = np.nan
        nit = 0
        conv = False
        u_after = u_cur
        v_after = v_cur
        ds_ratio = np.nan
        bisect_level_final = 0
        note = ''

        if u_pred is not None:
            if u_pred < surf_u_min or u_pred > surf_u_max:
                u_pred = np.clip(u_pred, surf_u_min, surf_u_max)
                success = False
                use_optical_next = True
                note = 'PREDICTOR_OUT_OF_BOUNDS'
            else:
                r_pred = surface.position(u_pred, v_pred)
                m_pred = surface.normal(u_pred, v_pred)
                Phi_before = float(np.dot(traj.R(z_next) - r_pred, m_pred))

                try:
                    u_c, v_c, Phi_c, nit, conv = newton_corrector(
                        surface, traj, u_pred, v_pred, z_next,
                        eps_Phi=eps_Phi, max_iter=max_newton
                    )
                    Phi_after = float(Phi_c)

                    if conv and abs(Phi_c) < eps_Phi:
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
                            ds_expect = speed_expect * dz_actual
                            ds_ratio = ds_actual / ds_expect if ds_expect > 1e-12 else 1.0

                            if ds_ratio > jump_threshold or ds_ratio < 1.0 / jump_threshold:
                                note = f'JUMP_DETECTED_RATIO={ds_ratio:.2f}'
                            else:
                                u_cur, v_cur = u_c, v_c
                                u_after = u_c
                                v_after = v_c
                                use_optical_next = False
                                success = True
                                note = 'OK'
                        except (ValueError, GeometryOutOfBoundsError) as e:
                            note = f'JUMP_CHECK_ERROR:{e}'
                    else:
                        note = f'NEWTON_FAIL:conv={conv},Phi={Phi_c:.2e},nit={nit}'
                        use_optical_next = True
                except GeometryOutOfBoundsError as e:
                    note = f'NEWTON_EXCEPTION:{e}'
                    use_optical_next = True

        # --- 4. Fallback: явный Эйлер + адаптивная бисекция ---
        if not success:
            flags_list[-1] = 4 if u_pred is not None else 3
            pred_type = 'FALLBACK_EULER'

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

                        u_p = np.clip(u_s + du_s * dz_sub, surf_u_min, surf_u_max)
                        v_p = np.clip(v_s + dv_s * dz_sub, surf_v_min, surf_v_max)

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

                except (ValueError, GeometryOutOfBoundsError) as e:
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
                    bisect_level_final = bisect_level
                    if bisect_level > 0:
                        bisected = True
                    break
                else:
                    n_sub *= 2

            u_cur, v_cur = best_u, best_v
            u_after = best_u
            v_after = best_v
            Phi_after = float(best_Phi)
            nit = best_nit
            conv = abs(best_Phi) < eps_Phi
            bisected_flag = 1 if bisected else flags_list[-1]
            flags_list[-1] = bisected_flag
            use_optical_next = force_optical_after_fail
            note += f';FALLBACK_BISECT={bisect_level_final},conv={conv}'

        # --- Добавляем новую точку в списки ---
        z_list.append(z_next)
        u_list.append(u_cur)
        v_list.append(v_cur)
        Phi_list.append(Phi_after if not np.isnan(Phi_after) else 1.0)
        newton_iters_list.append(nit)
        kappa_n_list.append(0.0)   # placeholder, будет обновлён на следующем шаге
        lam_list.append(0.0)       # placeholder
        flags_list.append(1 if bisected else 0)

        # --- Логирование ---
        log_writer.writerow([
            step_idx, round(z_k, 4), round(z_next, 4), pred_type,
            round(u_list[-2], 6), round(v_list[-2], 6),
            round(u_pred, 6) if u_pred is not None else 'None',
            round(v_pred, 6) if v_pred is not None else 'None',
            f'{Phi_before:.4e}' if not np.isnan(Phi_before) else 'N/A',
            f'{Phi_after:.4e}' if not np.isnan(Phi_after) else 'N/A',
            nit, int(conv),
            round(u_after, 6), round(v_after, 6),
            f'{kappa_n_list[-2]:.6e}',
            flags_list[-1],
            f'{ds_ratio:.4f}' if not np.isnan(ds_ratio) else 'N/A',
            bisect_level_final,
            round(dz_actual, 4),
            note
        ])

        if step_idx % 50 == 0 or not conv or abs(Phi_after) > 1.0:
            print(f"[{step_idx:4d}] z={z_k:7.2f} dz={dz_actual:6.3f} pred={pred_type:15s} "
                  f"u={u_after:8.3f} v={v_after:8.4f} "
                  f"Phi={Phi_after:10.3e} nit={nit:3d} conv={int(conv)} "
                  f"flag={flags_list[-1]} note={note}")

        step_idx += 1

    # --- Финальная точка ---
    try:
        _, lam_last, up_last, vp_last = recompute_thread_geometry(
            surface, traj, u_list[-1], v_list[-1], z_list[-1])
        lam_list[-1] = lam_last
        kappa_n_list[-1] = normal_curvature(
            surface, u_list[-1], v_list[-1], up_last, vp_last)
    except (ValueError, GeometryOutOfBoundsError):
        pass

    log_file.close()
    print(f"\nЛог сохранён в {os.path.abspath(log_path)}")
    print(f"Всего шагов: {len(z_list)-1} (запрошено ~{count_points})")

    # Преобразуем в numpy arrays
    N = len(z_list)
    z_eval = np.array(z_list)
    u_hist = np.array(u_list)
    v_hist = np.array(v_list)
    Phi_hist = np.array(Phi_list)
    kappa_n_hist = np.array(kappa_n_list)
    newton_iters_hist = np.array(newton_iters_list, dtype=int)
    lam_hist = np.array(lam_list)
    flags = np.array(flags_list, dtype=int)

    points_3d = np.array([surface.position(u_hist[k], v_hist[k]) for k in range(N)])

    return {
        'z_eval': z_eval,
        'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }
