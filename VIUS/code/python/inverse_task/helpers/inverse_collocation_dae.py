# helpers/inverse_collocation_dae.py
"""
Прямая коллокация обратной задачи намотки с DAE-инициализацией.

Постановка
----------
Ищем функции u(z), v(z), z in [0, L], удовлетворяющие связи

    Phi(u, v, z) = < R(z) - r(u, v),  m(u, v) > = 0.

Дискретизация: узлы z_0 < ... < z_{N-1}, неизвестные X = (u_i, v_i).
Невязка least_squares собирается из двух групп:

  1) алгебраическая связь в каждом узле:        w_Phi * Phi(u_i, v_i, z_i)
  2) дефект коллокации (Эрмит) в серединах:      w_diff * (производная узловая
     минус узловая разностная), что навязывает гладкое согласование u',v'
     с правой частью ОДУ du/dz, dv/dz из inverse_method_fixed.

ВАЖНО (фикс вылета GeometryOutOfBoundsError):
  - least_squares запускается С ГРАНИЦАМИ bounds по u (из surface.a/surface.d);
  - _phi/_rhs дополнительно клампят (u,v) перед обращением к поверхности и
    ловят GeometryOutOfBoundsError -> возвращают штрафную/нулевую невязку,
    чтобы пробный шаг trust-region не ронял весь решатель.

Интерфейс согласован с client_collocation_dae.py:

    solve_collocation_dae(surface, traj, u0, v0, count_points,
                          w_Phi, w_diff, init_method, dae_substeps,
                          dae_method, max_nfev, tol, verbose)
        -> dict(z=..., u=..., v=..., result=..., residual_norm=..., init=...)
"""
import numpy as np
from scipy.optimize import least_squares

from helpers.inverse_method_fixed import compute_dr_dz, newton_corrector
from helpers.dae_corrector_predictor import DAECorrectorPredictor

try:
    from core.exceptions import GeometryOutOfBoundsError
except Exception:  # на случай иной структуры пакета
    class GeometryOutOfBoundsError(Exception):
        pass


# ---------------------------------------------------------------------
# Границы области по u
# ---------------------------------------------------------------------
def _u_bounds(surface):
    """Достать допустимый диапазон u из поверхности (атрибуты a, d)."""
    a = getattr(surface, 'a', None)
    d = getattr(surface, 'd', None)
    if a is None or d is None:
        return -np.inf, np.inf
    return float(a), float(d)


def _clamp_u(u, u_lo, u_hi, margin=1e-9):
    """Зажать u строго внутрь [u_lo, u_hi] (с микро-отступом от границ)."""
    if np.isfinite(u_lo) and np.isfinite(u_hi):
        return float(np.clip(u, u_lo + margin, u_hi - margin))
    return float(u)


# ---------------------------------------------------------------------
# Базовые величины (защищены от выхода за границу)
# ---------------------------------------------------------------------
def _phi(surface, traj, u, v, z, u_lo=-np.inf, u_hi=np.inf):
    u = _clamp_u(u, u_lo, u_hi)
    try:
        r = surface.position(u, v)
        m = surface.normal(u, v)
    except GeometryOutOfBoundsError:
        return 0.0
    return float(np.dot(np.asarray(traj.R(z)) - np.asarray(r), np.asarray(m)))


def _rhs(surface, traj, u, v, z, u_lo=-np.inf, u_hi=np.inf):
    """du/dz, dv/dz вдоль связи; (0,0) если не определена/вне области."""
    u = _clamp_u(u, u_lo, u_hi)
    try:
        d = compute_dr_dz(surface, traj, u, v, z)
    except GeometryOutOfBoundsError:
        return 0.0, 0.0
    if d is None:
        return 0.0, 0.0
    du, dv = float(d[0]), float(d[1])
    if not (np.isfinite(du) and np.isfinite(dv)):
        return 0.0, 0.0
    return du, dv


# ---------------------------------------------------------------------
# Начальное приближение
# ---------------------------------------------------------------------
def _init_dae(surface, traj, u0, v0, z_nodes, substeps, method):
    n = len(z_nodes)
    u = np.empty(n)
    v = np.empty(n)
    u[0], v[0] = float(u0), float(np.mod(v0, 2 * np.pi))
    u_lo, u_hi = _u_bounds(surface)
    predictor = DAECorrectorPredictor(
        n_substeps=substeps, method=method,
        project_every_substep=True, eps_Phi=1e-10, max_newton_iter=20,
        u_lo=u_lo, u_hi=u_hi,
    )
    for k in range(n - 1):
        out = predictor.predict(z_nodes[k], z_nodes[k + 1], u[k], v[k],
                                surface, traj)
        if out is None:
            u[k + 1], v[k + 1] = u[k], v[k]
        else:
            u[k + 1], v[k + 1] = out
    return u, v


def _init_constant(surface, traj, u0, v0, z_nodes):
    n = len(z_nodes)
    return np.full(n, float(u0)), np.full(n, float(np.mod(v0, 2 * np.pi)))


# ---------------------------------------------------------------------
# Невязка коллокации
# ---------------------------------------------------------------------
def _residuals(X, surface, traj, z_nodes, w_Phi, w_diff, u_lo, u_hi):
    n = len(z_nodes)
    u = X[:n]
    v = X[n:]

    res = []

    # 1) связь в узлах
    for i in range(n):
        res.append(w_Phi * _phi(surface, traj, u[i], v[i], z_nodes[i],
                                u_lo, u_hi))

    # 2) эрмитов дефект коллокации в серединах интервалов
    for i in range(n - 1):
        h = z_nodes[i + 1] - z_nodes[i]
        if h == 0.0:
            res.append(0.0)
            res.append(0.0)
            continue
        # узловые производные из правой части ОДУ
        du_i, dv_i = _rhs(surface, traj, u[i], v[i], z_nodes[i], u_lo, u_hi)
        du_j, dv_j = _rhs(surface, traj, u[i + 1], v[i + 1], z_nodes[i + 1],
                          u_lo, u_hi)
        # разностная производная по хорде
        du_chord = (u[i + 1] - u[i]) / h
        dv_chord = (v[i + 1] - v[i]) / h
        # дефект: средняя узловая производная должна совпасть с хордовой
        res.append(w_diff * (0.5 * (du_i + du_j) - du_chord))
        res.append(w_diff * (0.5 * (dv_i + dv_j) - dv_chord))

    return np.asarray(res, dtype=float)


# ---------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------
def solve_collocation_dae(surface, traj, u0, v0,
                          count_points=100,
                          w_Phi=1.0, w_diff=1.0,
                          init_method='dae',
                          dae_substeps=4, dae_method='rk4',
                          max_nfev=10000, tol=1e-10,
                          verbose=False):
    """Решить обратную задачу намотки прямой коллокацией.

    Возвращает dict с ключами: z, u, v, result, residual_norm, init.
    """
    L = float(traj.total_length)
    z_nodes = np.linspace(0.0, L, int(count_points))
    n = len(z_nodes)

    u_lo, u_hi = _u_bounds(surface)

    # --- начальное приближение ---
    if init_method == 'dae':
        u_init, v_init = _init_dae(surface, traj, u0, v0,
                                   z_nodes, dae_substeps, dae_method)
    elif init_method == 'constant':
        u_init, v_init = _init_constant(surface, traj, u0, v0, z_nodes)
    else:
        raise ValueError("init_method must be 'dae' or 'constant'")

    # стартовую точку тоже зажимаем строго внутрь области
    if np.isfinite(u_lo) and np.isfinite(u_hi):
        u_init = np.clip(u_init, u_lo + 1e-9, u_hi - 1e-9)

    X0 = np.concatenate([u_init, v_init])

    # --- границы для least_squares ---
    # u в [u_lo, u_hi]; v без жёстких границ (поверхность вращения, v периодична)
    lb = np.concatenate([np.full(n, u_lo), np.full(n, -np.inf)])
    ub = np.concatenate([np.full(n, u_hi), np.full(n, np.inf)])
    use_bounds = np.isfinite(u_lo) and np.isfinite(u_hi)
    bounds = (lb, ub) if use_bounds else (-np.inf, np.inf)

    if verbose:
        r0 = _residuals(X0, surface, traj, z_nodes, w_Phi, w_diff, u_lo, u_hi)
        bnd_txt = f"u in [{u_lo:.4g}, {u_hi:.4g}]" if use_bounds else "без границ"
        print(f"  [collocation] init='{init_method}', узлов={n}, "
              f"{bnd_txt}, ||res0||={np.linalg.norm(r0):.3e}")

    # --- решение least_squares ---
    sol = least_squares(
        _residuals, X0,
        args=(surface, traj, z_nodes, w_Phi, w_diff, u_lo, u_hi),
        bounds=bounds,
        method='trf', max_nfev=int(max_nfev),
        ftol=tol, xtol=tol, gtol=tol,
    )

    u = sol.x[:n]
    v = np.mod(sol.x[n:], 2 * np.pi)
    res_norm = float(np.linalg.norm(sol.fun))

    if verbose:
        print(f"  [collocation] done: success={sol.success}, "
              f"nfev={sol.nfev}, ||res||={res_norm:.3e}")

    return {
        'z': z_nodes,
        'u': u,
        'v': v,
        'result': sol,
        'residual_norm': res_norm,
        'init': {'u': u_init, 'v': v_init, 'method': init_method},
    }
