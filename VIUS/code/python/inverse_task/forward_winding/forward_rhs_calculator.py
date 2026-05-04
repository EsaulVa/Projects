import numpy as np
from typing import Optional
from geometry.tsurfaces import AnalyticalSurface
from core.base_deviation_law import DeviationLaw


class ForwardRHS:
    """
    Вычислитель правых частей системы (2.10) для прямой задачи намотки.

    Система обыкновенных дифференциальных уравнений (ОДУ) описывает линию укладки
    на поверхности произвольной формы по заданному закону изменения угла
    геодезического отклонения θ(s). Интегрирование ведётся по натуральному
    параметру s (длине дуги) кривой.

    Уравнения (2.10):

        du/ds   = u'
        du'/ds  = γ1 * tgθ - Γ¹₁₁ u'² - 2Γ¹₁₂ u'v' - Γ¹₂₂ v'²
        dv/ds   = v'
        dv'/ds  = γ2 * tgθ - Γ²₁₁ u'² - 2Γ²₁₂ u'v' - Γ²₂₂ v'²

    где:
        u, v   – криволинейные координаты на поверхности,
        u', v' – производные по натуральному параметру s,
        Γⁱⱼₖ   – символы Кристоффеля второго рода,
        γ1, γ2 – коэффициенты, выражаемые через коэффициенты квадратичных форм
                 поверхности и нормальную кривизну в направлении касательной,
        tgθ    – тангенс угла геодезического отклонения в точке s.

    Класс инкапсулирует всю геометрическую информацию о поверхности и законе
    отклонения, предоставляя единый метод __call__(s, state) для использования
    в численных решателях ОДУ.
    """

    def __init__(
        self,
        surface: AnalyticalSurface,
        deviation_law: DeviationLaw,
        normalize_tangent: bool = True,
        eps: float = 1e-12
    ):
        """
        Параметры
        ---------
        surface : AnalyticalSurface
            Поверхность оправки, предоставляющая квадратичные формы и
            символы Кристоффеля.
        deviation_law : DeviationLaw
            Закон изменения тангенса угла геодезического отклонения θ(s).
        normalize_tangent : bool, optional
            Если True, на каждом шаге вычислений производится нормировка
            вектора касательной (u', v') к единичной длине. Это рекомендуется
            для подавления накопления ошибок численного интегрирования.
        eps : float, optional
            Малая величина для защиты от деления на ноль и проверок близости.
        """
        self.surface = surface
        self.deviation_law = deviation_law
        self.normalize_tangent = normalize_tangent
        self.eps = eps

    def __call__(self, s: float, state: np.ndarray) -> np.ndarray:
        """
        Вычисляет правые части системы (2.10) в точке s.

        Аргументы
        ---------
        s : float
            Текущее значение натурального параметра (длина дуги).
        state : np.ndarray формы (4,)
            Вектор состояния: [u, u_prime, v, v_prime].

        Возвращает
        ----------
        np.ndarray формы (4,)
            Производные: [du/ds, du'/ds, dv/ds, dv'/ds].
        """
        u, u_prime, v, v_prime = state

        # 1. Коэффициенты первой квадратичной формы E, F, G
        E, F, G = self.surface.first_fundamental_form(u, v)

        # 2. Коэффициенты второй квадратичной формы L, M, N
        L, M, N_coef = self.surface.second_fundamental_form(u, v)

        # 3. Нормировка касательного вектора (если включена)
        if self.normalize_tangent:
            norm2 = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
            # Защита от слишком малой нормы (например, в особых точках)
            if norm2 > self.eps:
                scale = 1.0 / np.sqrt(norm2)
                u_prime *= scale
                v_prime *= scale

        # 4. Вспомогательные величины
        det = E * G - F * F
        sqrt_det = np.sqrt(max(det, self.eps))   # √(EG - F²)

        # Нормальная кривизна поверхности в направлении касательной
        kN = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_coef * v_prime**2

        # Коэффициенты γ1 и γ2 из уравнений (2.10)
        # γ1 = kN * (F u' + G v') / √(EG - F²)
        # γ2 = -kN * (E u' + F v') / √(EG - F²)
        gamma1 = kN * (F * u_prime + G * v_prime) / sqrt_det
        gamma2 = -kN * (E * u_prime + F * v_prime) / sqrt_det

        # 5. Тангенс угла геодезического отклонения в точке s
        tan_theta = self.deviation_law.tan_theta(s)

        # 6. Символы Кристоффеля второго рода Γⁱⱼₖ
        #    Ожидается, что surface предоставляет метод christoffel_symbols(u, v)
        #    возвращающий тензор (2,2,2) с индексацией [i, j, k].
        if hasattr(self.surface, 'christoffel_symbols'):
            Gamma = self.surface.christoffel_symbols(u, v)
        else:
            # Если метод отсутствует, можно вычислить через статический метод
            # (требуются производные метрики)
            raise NotImplementedError(
                "Поверхность должна предоставлять метод christoffel_symbols(u, v)"
            )

        # 7. Вычисление правых частей
        du_ds = u_prime
        dv_ds = v_prime

        # Геодезическая часть (символы Кристоффеля)
        geo_u = (Gamma[0, 0, 0] * u_prime**2 +
                 2.0 * Gamma[0, 0, 1] * u_prime * v_prime +
                 Gamma[0, 1, 1] * v_prime**2)
        geo_v = (Gamma[1, 0, 0] * u_prime**2 +
                 2.0 * Gamma[1, 0, 1] * u_prime * v_prime +
                 Gamma[1, 1, 1] * v_prime**2)

        # Добавка от ненулевого угла геодезического отклонения
        du_prime_ds = gamma1 * tan_theta - geo_u
        dv_prime_ds = gamma2 * tan_theta - geo_v

        return np.array([du_ds, du_prime_ds, dv_ds, dv_prime_ds])

class QRegularizedForwardRHS:
    def __init__(self, surface: AnalyticalSurface, deviation_law: DeviationLaw,
                 q_param: float = 1.0, adaptive_q: bool = True, eps: float = 1e-12):
        self.surface = surface
        self.law = deviation_law
        self.q_param = q_param
        self.adaptive_q = adaptive_q
        self.eps = eps
        self._last_error: Optional[str] = None   # диагностика

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _compute_second_derivatives(self, u: float, v: float,
                                    u_prime: float, v_prime: float,
                                    tan_theta: float) -> tuple[float, float]:
        # Получаем первые производные и нормаль
        try:
            geom = self.surface.derivatives(u, v)
        except Exception as e:
            self._last_error = f"Ошибка получения первых производных: {e}"
            raise
        ru, rv, normal = geom['ru'], geom['rv'], geom['normal']

        # Пытаемся получить вторые производные через специальный метод
        try:
            second = self.surface.second_derivatives(u, v)
            ruu, ruv, rvv = second['ruu'], second['ruv'], second['rvv']
        except (AttributeError, KeyError, NotImplementedError) as e:
            self._last_error = (
                "Поверхность не предоставляет вторые производные (second_derivatives). "
                "Добавьте метод second_derivatives в класс поверхности."
            )
            raise RuntimeError(self._last_error) from e

        # Касательный вектор τ и комбинация Nd
        tau = ru * u_prime + rv * v_prime
        Nd = ruu * u_prime**2 + 2.0 * ruv * u_prime * v_prime + rvv * v_prime**2

        # Векторное произведение τ × m
        tau_cross_m = np.cross(tau, normal)

        # Матрица системы A (2×2)
        A = np.array([
            [np.dot(tau, ru), np.dot(tau, rv)],
            [np.dot(ru, tau_cross_m), np.dot(rv, tau_cross_m)]
        ])
        b = np.array([
            -np.dot(tau, Nd),
            np.dot(normal, Nd) * tan_theta - np.dot(Nd, tau_cross_m)
        ])

        # Оценка обусловленности и адаптивный q
        det_A = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        q = self.q_param
        if self.adaptive_q:
            norm_A = np.linalg.norm(A, 'fro')
            cond_est = norm_A / max(abs(det_A), self.eps)
            q = self.q_param + min(0.999, 1.0 / (cond_est + 1e-6))
            q = min(q, 2.0)

        # q-регуляризация через нормальные уравнения
        ATA = A.T @ A
        ATb = A.T @ b
        diag_ATA = np.diag(np.diag(ATA))
        reg_matrix = 2.0 * ATA + (q - 1.0) * diag_ATA
        rhs_reg = 2.0 * ATb

        try:
            sol = np.linalg.solve(reg_matrix, rhs_reg)
        except np.linalg.LinAlgError:
            self._last_error = "Регуляризованная матрица вырождена, использую lstsq"
            sol = np.linalg.lstsq(reg_matrix, rhs_reg, rcond=None)[0]

        self._last_error = None
        return sol[0], sol[1]

    def __call__(self, s: float, state: np.ndarray) -> np.ndarray:
        u, u_prime, v, v_prime = state

        # Нормировка
        E, F, G = self.surface.first_fundamental_form(u, v)
        norm2 = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
        if norm2 > self.eps:
            scale = 1.0 / np.sqrt(norm2)
            u_prime *= scale
            v_prime *= scale

        tan_theta = self.law.tan_theta(s)

        try:
            u2, v2 = self._compute_second_derivatives(u, v, u_prime, v_prime, tan_theta)
        except RuntimeError as e:
            self._last_error = str(e)
            raise

        return np.array([u_prime, u2, v_prime, v2])
    
# forward_winding/delegating_forward_rhs.py
# import numpy as np
# from geometry.tsurfaces import AnalyticalSurface
# from winding.deviation_law import DeviationLaw
from solvers.linear_solver import LocalLinearSolver

class DelegatingForwardRHS:
    """
    Правая часть системы (2.10) с делегированием решения локальной СЛАУ
    произвольному объекту LocalLinearSolver.
    """
    def __init__(self,
                 surface: AnalyticalSurface,
                 deviation_law: DeviationLaw,
                 linear_solver: LocalLinearSolver,
                 eps: float = 1e-12):
        self.surface = surface
        self.law = deviation_law
        self.linear_solver = linear_solver
        self.eps = eps

    def __call__(self, s: float, state: np.ndarray) -> np.ndarray:
        u, u_prime, v, v_prime = state

        # Нормировка
        E, F, G = self.surface.first_fundamental_form(u, v)
        norm2 = E*u_prime**2 + 2*F*u_prime*v_prime + G*v_prime**2
        if norm2 > self.eps:
            scale = 1.0 / np.sqrt(norm2)
            u_prime *= scale
            v_prime *= scale

        tan_theta = self.law.tan_theta(s)

        # Геометрия
        geom = self.surface.derivatives(u, v)
        ru = geom['ru']
        rv = geom['rv']
        normal = geom['normal']

        # Вторые производные (обязательно наличие second_derivatives)
        try:
            second = self.surface.second_derivatives(u, v)
            ruu, ruv, rvv = second['ruu'], second['ruv'], second['rvv']
        except AttributeError:
            raise RuntimeError(
                "Поверхность должна предоставлять метод second_derivatives "
                "для DelegatingForwardRHS. Добавьте его в класс поверхности."
            )

        tau = ru * u_prime + rv * v_prime
        Nd = ruu * u_prime**2 + 2.0 * ruv * u_prime * v_prime + rvv * v_prime**2
        tau_cross_m = np.cross(tau, normal)

        # Сборка системы A x = b
        A = np.array([
            [np.dot(tau, ru), np.dot(tau, rv)],
            [np.dot(ru, tau_cross_m), np.dot(rv, tau_cross_m)]
        ])
        b_vec = np.array([
            -np.dot(tau, Nd),
            np.dot(normal, Nd) * tan_theta - np.dot(Nd, tau_cross_m)
        ])

        # Делегируем решение линейной системы
        u2, v2 = self.linear_solver.solve(A, b_vec)

        return np.array([u_prime, u2, v_prime, v2])