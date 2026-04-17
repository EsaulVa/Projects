import numpy as np
from typing import Optional
from geometry.tsurfaces import AnalyticalSurface
from core.trajectory import Trajectory


class RightHandSideCalculator:
    """
    Вычислитель правых частей системы (3.41) для обратной задачи намотки.
    
    На каждом шаге интегрирования по натуральному параметру траектории z
    вычисляет производные du/dz, dv/dz криволинейных координат u, v линии укладки
    на поверхности оправки.
    """
    
    def __init__(
        self,
        surface: AnalyticalSurface,
        trajectory: Trajectory,
        k: float = 1.0,
        max_ds_dz: Optional[float] = None,
        delta_clip: float = 0.999,
        eps: float = 1e-12
    ):
        """
        Параметры
        ---------
        surface : AnalyticalSurface
            Поверхность оправки, предоставляющая метод derivatives() и квадратичные формы.
        trajectory : Trajectory
            Траектория точки схода нити с натуральной параметризацией.
        k : float, optional
            Коэффициент усиления корректирующего члена в ds/dz (по умолчанию 1.0).
        max_ds_dz : float, optional
            Максимально допустимое значение |ds/dz| (если None, не ограничивается).
        delta_clip : float, optional
            Максимально допустимое значение модуля невязки δ для предотвращения деления на ноль
            в выражении α = 1/√(1-δ²). По умолчанию 0.999.
        eps : float, optional
            Малая величина для защиты от деления на ноль в других местах.
        """
        self.surface = surface
        self.trajectory = trajectory
        self.k = k
        self.max_ds_dz = max_ds_dz
        self.delta_clip = delta_clip
        self.eps = eps

    def __call__(self, z: float, state: np.ndarray) -> tuple[float, float]:
        """
        Вычисляет правые части du/dz, dv/dz в точке z.
        
        Аргументы
        ---------
        z : float
            Текущее значение натурального параметра траектории.
        state : np.ndarray формы (2,)
            Текущие криволинейные координаты [u, v].
            
        Возвращает
        ----------
        du_dz : float
        dv_dz : float
        """
        u, v = state

        # 1. Получить положение и производную точки схода нити
        R = self.trajectory.R(z)          # вектор R(z)
        R_deriv = self.trajectory.R_deriv(z)  # единичный касательный вектор R'(z)

        # 2. Получить геометрию поверхности в точке (u, v)
        geom = self.surface.derivatives(u, v)
        r = geom['r']       # r(u,v)
        ru = geom['ru']     # ∂r/∂u
        rv = geom['rv']     # ∂r/∂v
        n = geom['normal']  # единичная нормаль m

        # 3. Вектор от точки на поверхности до точки схода и его норма
        diff = R - r
        diff_norm = np.linalg.norm(diff)
        if diff_norm < self.eps:
            # Точка схода совпадает с поверхностью — практически невозможно,
            # но возвращаем нули для устойчивости.
            return 0.0, 0.0

        # 4. Невязка δ = < (R-r)/|R-r|, m >
        diff_unit = diff / diff_norm
        delta = np.dot(diff_unit, n)

        # 5. Защита от выхода за пределы [-delta_clip, delta_clip]
        delta = np.clip(delta, -self.delta_clip, self.delta_clip)

        # 6. Коэффициент α = 1 / √(1 - δ²)
        alpha = 1.0 / np.sqrt(max(self.eps, 1.0 - delta * delta))

        # 7. Первая квадратичная форма E, F, G
        E, F, G = self.surface.first_fundamental_form(u, v)
        det = E * G - F * F
        if abs(det) < self.eps:
            raise ValueError(f"Singular metric at u={u}, v={v}")

        # 8. Вектор b = (diff_unit - δ * n) * α
        #    (это нормированная и скорректированная касательная)
        b = (diff_unit - delta * n) * alpha

        # 9. Компоненты для решения линейной системы относительно u', v'
        b1 = np.dot(b, ru)
        b2 = np.dot(b, rv)

        # 10. Решение системы:
        #     [E F] [u'] = [b1]
        #     [F G] [v']   [b2]
        #     => u' = (G*b1 - F*b2) / det
        #        v' = (-F*b1 + E*b2) / det
        u_prime = (G * b1 - F * b2) / det
        v_prime = (-F * b1 + E * b2) / det

        # 11. Вторая квадратичная форма L, M, N
        L, M, N_coef = self.surface.second_fundamental_form(u, v)

        # 12. Вторая квадратичная форма на направлении (u', v')
        II = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_coef * v_prime**2
        if abs(II) < self.eps:
            # Избегаем деления на ноль, знак сохраняем
            II = np.copysign(self.eps, II) if II != 0 else self.eps

        # 13. Скалярное произведение <R'(z), m>
        R_deriv_dot_n = np.dot(R_deriv, n)

        # 14. Вычисление ds/dz по формуле (3.41)
        #     основная часть: (R_deriv_dot_n / (diff_norm * II)) * alpha
        #     корректирующая часть: (delta / II) * alpha * k
        main_term = (R_deriv_dot_n / (diff_norm * II)) * alpha
        correction_term = (delta / II) * alpha * self.k
        ds_dz = main_term + correction_term

        # 15. Ограничение максимального шага по s
        if self.max_ds_dz is not None:
            if abs(ds_dz) > self.max_ds_dz:
                ds_dz = np.copysign(self.max_ds_dz, ds_dz)

        # 16. Итоговые производные du/dz, dv/dz
        du_dz = u_prime * ds_dz
        dv_dz = v_prime * ds_dz

        return du_dz, dv_dz