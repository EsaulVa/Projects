# helpers/dae_corrector_predictor.py
"""
DAE предиктор-корректор (индекс 1) для обратной задачи намотки.

Идея
----
Связь (алгебраическое уравнение):

    Phi(u, v, z) = < R(z) - r(u, v),  m(u, v) > = 0,

где r(u,v) — точка оправки, m(u,v) — единичная нормаль, R(z) — ТСН-траектория
(точка схода нити). Параметр z играет роль "времени".

Дифференцируя Phi по z вдоль решения, получаем линейную систему относительно
(du/dz, dv/dz). Её правая часть — это compute_dr_dz из inverse_method_fixed
(согласованный знак градиента). Чистое интегрирование этой ОДУ накапливает
дрейф связи (constraint drift), поэтому после КАЖДОГО (под)шага состояние
проецируется обратно на многообразие Phi=0 ньютоновским корректором.

ВАЖНО (фикс вылета GeometryOutOfBoundsError):
  - после предиктора и корректора u зажимается в допустимый диапазон
    [u_lo, u_hi] (границы поверхности по образующей);
  - обращения к поверхности обёрнуты от GeometryOutOfBoundsError, чтобы марш
    не падал на пробных точках RK4, а штатно сообщал об отказе (None).

Схема (проекционный метод для DAE индекса 1):

    предиктор:  (u,v)  --RK4/Эйлер по du/dz, dv/dz-->  (u*, v*)
    клампинг u:  u* -> [u_lo, u_hi]
    корректор:  (u*,v*) --newton_corrector(Phi=0)-->   (u,v) на связи
    клампинг u:  u -> [u_lo, u_hi]

Интерфейс согласован с client_collocation_dae.py:

    DAECorrectorPredictor(n_substeps, method, project_every_substep,
                          eps_Phi, max_newton_iter, u_lo, u_hi)
    .predict(z_k, z_next, u_k, v_k, surface, traj) -> (u, v) | None
"""
import numpy as np

from helpers.inverse_method_fixed import compute_dr_dz, newton_corrector

try:
    from core.exceptions import GeometryOutOfBoundsError
except Exception:
    class GeometryOutOfBoundsError(Exception):
        pass


class DAECorrectorPredictor:
    """Проекционный DAE предиктор-корректор индекса 1.

    Параметры
    ---------
    n_substeps : int
        Число подшагов внутри одного интервала [z_k, z_next].
    method : {'rk4', 'euler'}
        Схема интегрирования дифференциальной части (предиктор).
    project_every_substep : bool
        Если True — проекция на Phi=0 после каждого подшага (надёжнее);
        если False — только в конце интервала (быстрее, но дрейф больше).
    eps_Phi : float
        Допуск корректора по |Phi|.
    max_newton_iter : int
        Максимум итераций ньютоновского корректора.
    max_du, max_dv : float | None
        Ограничители приращений предиктора на подшаг (защита от выброса).
        None — без ограничений.
    u_lo, u_hi : float | None
        Допустимый диапазон параметра u (образующая). Если заданы —
        u жёстко зажимается внутрь после предиктора и корректора.
    margin : float
        Микро-отступ от границ при клампинге (чтобы не попасть ровно на край).
    """

    def __init__(self, n_substeps=4, method='rk4',
                 project_every_substep=True,
                 eps_Phi=1e-10, max_newton_iter=20,
                 max_du=None, max_dv=None,
                 u_lo=None, u_hi=None, margin=1e-9):
        self.n_substeps = max(1, int(n_substeps))
        self.method = str(method).lower()
        if self.method not in ('rk4', 'euler'):
            raise ValueError("method must be 'rk4' or 'euler'")
        self.project_every_substep = bool(project_every_substep)
        self.eps_Phi = float(eps_Phi)
        self.max_newton_iter = int(max_newton_iter)
        self.max_du = max_du
        self.max_dv = max_dv
        self.u_lo = None if u_lo is None else float(u_lo)
        self.u_hi = None if u_hi is None else float(u_hi)
        self.margin = float(margin)

    def _clamp_u(self, u):
        if self.u_lo is not None and self.u_hi is not None:
            return float(np.clip(u, self.u_lo + self.margin,
                                 self.u_hi - self.margin))
        return float(u)

    # -- правая часть ОДУ: (du/dz, dv/dz) при заданных (u, v, z) --
    def _rhs(self, z, u, v, surface, traj):
        u = self._clamp_u(u)
        try:
            d = compute_dr_dz(surface, traj, u, v, z)
        except GeometryOutOfBoundsError:
            return None
        if d is None:
            return None
        du, dv = float(d[0]), float(d[1])
        if not (np.isfinite(du) and np.isfinite(dv)):
            return None
        return du, dv

    def _clip(self, du, dv):
        if self.max_du is not None:
            du = float(np.clip(du, -self.max_du, self.max_du))
        if self.max_dv is not None:
            dv = float(np.clip(dv, -self.max_dv, self.max_dv))
        return du, dv

    # -- один подшаг предиктора --
    def _predictor_step(self, z, u, v, h, surface, traj):
        if self.method == 'euler':
            k = self._rhs(z, u, v, surface, traj)
            if k is None:
                return None
            du, dv = self._clip(k[0] * h, k[1] * h)
            return self._clamp_u(u + du), v + dv

        # RK4
        k1 = self._rhs(z, u, v, surface, traj)
        if k1 is None:
            return None
        k2 = self._rhs(z + 0.5 * h, u + 0.5 * h * k1[0],
                       v + 0.5 * h * k1[1], surface, traj)
        if k2 is None:
            return None
        k3 = self._rhs(z + 0.5 * h, u + 0.5 * h * k2[0],
                       v + 0.5 * h * k2[1], surface, traj)
        if k3 is None:
            return None
        k4 = self._rhs(z + h, u + h * k3[0], v + h * k3[1], surface, traj)
        if k4 is None:
            return None
        du = (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dv = (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        du, dv = self._clip(du, dv)
        return self._clamp_u(u + du), v + dv

    # -- проекция на многообразие Phi(u,v,z)=0 --
    def _project(self, surface, traj, u, v, z):
        u = self._clamp_u(u)
        try:
            u_c, v_c, Phi, _, conv = newton_corrector(
                surface, traj, u, v, z,
                eps_Phi=self.eps_Phi, max_iter=self.max_newton_iter,
            )
        except GeometryOutOfBoundsError:
            return None
        if not np.isfinite(u_c) or not np.isfinite(v_c):
            return None
        u_c = self._clamp_u(u_c)
        # периодичность v
        v_c = float(np.mod(v_c, 2.0 * np.pi))
        return u_c, v_c, conv

    def predict(self, z_k, z_next, u_k, v_k, surface, traj):
        """Проинтегрировать от z_k до z_next с проекцией на связь.

        Возвращает (u, v) на связи Phi=0 или None при отказе.
        """
        h_total = float(z_next) - float(z_k)
        if h_total == 0.0:
            return self._clamp_u(u_k), float(np.mod(v_k, 2.0 * np.pi))

        h = h_total / self.n_substeps
        u, v, z = self._clamp_u(u_k), float(v_k), float(z_k)

        for s in range(self.n_substeps):
            step = self._predictor_step(z, u, v, h, surface, traj)
            if step is None:
                return None
            u_p, v_p = step
            z_p = z + h

            do_project = self.project_every_substep or (s == self.n_substeps - 1)
            if do_project:
                proj = self._project(surface, traj, u_p, v_p, z_p)
                if proj is None:
                    return None
                u, v, _ = proj
            else:
                u, v = self._clamp_u(u_p), float(np.mod(v_p, 2.0 * np.pi))
            z = z_p

        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return u, v
