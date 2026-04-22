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