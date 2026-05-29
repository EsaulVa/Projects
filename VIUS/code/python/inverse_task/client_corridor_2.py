import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.trajectory import Trajectory
from helpers.intersection import RayTracer


@dataclass
class CorridorShadowResult:
    """Контейнер для результатов расчёта ТСН в модели тени."""
    s_array: np.ndarray
    lu_points: np.ndarray              # 3D точки на линии укладки
    safety_points: np.ndarray          # 3D точки на поверхности безопасности
    lambda_max: np.ndarray              # Длины лучей
    valid_mask: np.ndarray              # Маска валидных точек
    safety_trajectory: Optional[Trajectory]
    phi_values: np.ndarray              # Невязка Φ для диагностики


class CorridorShadowCalculator:
    """
    Вычисляет траекторию схода нити (ТСН) в модели тени.

    Отличие от CorridorMaxCalculator:
    - Луч строится не от касательной к ЛУ, а от касательной ПРОЕКЦИИ
      направления на касательную плоскость поверхности E2.

    Это обеспечивает выполнение условия Φ = 0 в модели тени:
        Φ = ⟨R(z) - r(z), m(r(z))⟩ = 0

    Геометрия:
        E2 — внутренняя поверхность (оправка), где лежит линия укладки
        E1 — внешняя поверхность (безопасность), куда приходит ТСН

        Для каждой точки r(z) ∈ E2:
        1. Находим касательную τ к линии укладки
        2. Проецируем τ на касательную плоскость T_{r(z)}E2
        3. Строим луч r(z) + t · τ_proj, направленный к E1
        4. Находим пересечение с E1
    """

    def __init__(self,
                 lu_trajectory: Trajectory,
                 mandrel_surface,           # E2 — поверхность оправки
                 safety_surface,             # E1 — поверхность безопасности
                 ray_tracer: RayTracer,
                 safe_distance: float = 10.0):
        """
        Параметры
        ----------
        lu_trajectory : Trajectory
            Линия укладки на поверхности E2.
        mandrel_surface : Surface
            Поверхность E2 (оправка) с методом normal(u, v).
        safety_surface : Surface
            Поверхность E1 (безопасность) для трассировки лучей.
        ray_tracer : RayTracer
            Настроенный трассировщик лучей.
        safe_distance : float
            Минимальное смещение от поверхности для начала поиска.
        """
        self.traj = lu_trajectory
        self.mandrel = mandrel_surface      # E2
        self.safety = safety_surface        # E1
        self.tracer = ray_tracer
        self.safe_dist = safe_distance

    def _get_surface_normal(self, point_3d):
        """
        Получить нормаль к поверхности E2 вблизи точки point_3d.
        Использует uv_from_point если доступен, иначе — приближение.
        """
        if hasattr(self.mandrel, 'uv_from_point'):
            try:
                u, v = self.mandrel.uv_from_point(point_3d)
                return self.mandrel.normal(u, v)
            except (ValueError, AttributeError):
                pass

        # Fallback: приближение через gradient функции уровня
        # (для поверхностей вращения это радиальное направление)
        r_xy = np.sqrt(point_3d[0]**2 + point_3d[1]**2)
        if r_xy > 1e-6:
            return np.array([point_3d[0]/r_xy, point_3d[1]/r_xy, 0.0])
        return np.array([1.0, 0.0, 0.0])

    def _project_to_tangent_plane(self, tangent_3d, normal):
        """
        Проекция вектора на касательную плоскость.

        τ_proj = τ - ⟨τ, m⟩ · m

        Parameters
        ----------
        tangent_3d : np.ndarray (3,)
            Касательный вектор в 3D.
        normal : np.ndarray (3,)
            Нормаль к поверхности.

        Returns
        -------
        np.ndarray (3,)
            Компонента вектора, лежащая в касательной плоскости.
        """
        dot = np.dot(tangent_3d, normal)
        return tangent_3d - dot * normal

    def _build_shadow_direction(self, r_point, tau_lu):
        """
        Построить направление луча в модели тени.

        1. Находим нормаль m к E2 в точке r_point
        2. Проецируем τ_ЛУ на касательную плоскость T_{r}E2
        3. Направление к E1 = τ_proj + α·m, где α подбирается
           из условия достижения E1

        Для простоты: возвращаем τ_proj (он уже лежит в касательной
        плоскости), а α определяется при трассировке.
        """
        # Нормаль к E2
        m = self._get_surface_normal(r_point)

        # Проекция касательной на касательную плоскость E2
        tau_proj = self._project_to_tangent_plane(tau_lu, m)

        # Нормализуем, если длина > 0
        norm = np.linalg.norm(tau_proj)
        if norm > 1e-10:
            tau_proj = tau_proj / norm

        return tau_proj, m

    def _compute_phi(self, r_point, R_point):
        """
        Вычислить невязку связи Φ = ⟨R - r, m⟩.

        Для корректной ТСН должно быть Φ ≈ 0.
        """
        m = self._get_surface_normal(r_point)
        delta = R_point - r_point
        return np.dot(delta, m)

    def calculate(self, num_points: int = 200, t_max: float = 1500.0) -> CorridorShadowResult:
        """
        Выполнить расчёт ТСН в модели тени.

        Parameters
        ----------
        num_points : int
            Количество точек сэмплирования по длине дуги ЛУ.
        t_max : float
            Максимальная длина луча.

        Returns
        -------
        CorridorShadowResult
        """
        s_array = np.linspace(0, self.traj.total_length, num_points)

        lu_points = np.zeros((num_points, 3))
        safety_points = np.zeros((num_points, 3))
        lambda_max = np.zeros(num_points)
        valid_mask = np.zeros(num_points, dtype=bool)
        phi_values = np.zeros(num_points)

        print(f"Расчёт ТСН в модели тени: {num_points} точек")

        for i, s in enumerate(s_array):
            # Точка на линии укладки
            r = self.traj.R(s)
            lu_points[i] = r

            # Касательная к линии укладки
            tau_lu = self.traj.R_deriv(s)

            # Строим направление в модели тени
            tau_proj, normal_m = self._build_shadow_direction(r, tau_lu)

            # Если проекция слишком мала (ЛУ перпендикулярна нормали),
            # используем направление к оси
            if np.linalg.norm(tau_proj) < 1e-6:
                print(f"  Точка {i}: проекция мала, используем радиальное направление")
                tau_proj = self._project_to_tangent_plane(
                    np.array([-r[0], -r[1], 0.0]), normal_m
                )
                if np.linalg.norm(tau_proj) > 1e-6:
                    tau_proj = tau_proj / np.linalg.norm(tau_proj)

            # Трассируем луч от точки на E2 в направлении tau_proj
            # с минимальным смещением safe_dist
            start_point = r + self.safe_dist * tau_proj

            try:
                t, pt = self.tracer.trace(
                    self.safety,           # E1 — поверхность безопасности
                    start_point,           # Старт от E2 + small offset
                    tau_proj,              # Направление в модели тени
                    t_min=1.0,             # Отступаем от start_point
                    t_max=t_max
                )

                if t is not None:
                    safety_points[i] = pt
                    lambda_max[i] = t + self.safe_dist
                    valid_mask[i] = True

                    # Диагностика Φ
                    phi_values[i] = self._compute_phi(r, pt)

                    if i % 50 == 0:
                        print(f"  [{i:3d}/{num_points}] Φ = {phi_values[i]:.2e}, λ = {lambda_max[i]:.1f}")
                else:
                    safety_points[i] = start_point + t_max * tau_proj
                    lambda_max[i] = np.inf
                    phi_values[i] = np.nan

            except Exception as e:
                safety_points[i] = start_point + t_max * tau_proj
                lambda_max[i] = np.inf
                phi_values[i] = np.nan
                if i % 20 == 0:
                    print(f"  Точка {i}: ошибка трассировки: {e}")

        # Статистика по Φ
        valid_phi = phi_values[valid_mask]
        if len(valid_phi) > 0:
            print(f"\nСтатистика Φ (невязка связи):")
            print(f"  |Φ| mean: {np.mean(np.abs(valid_phi)):.2e}")
            print(f"  |Φ| max:  {np.max(np.abs(valid_phi)):.2e}")
            print(f"  Φ вблизи 0 (|Φ| < 1e-6): {np.sum(np.abs(valid_phi) < 1e-6)}/{len(valid_phi)}")

        # Строим сглаженную траекторию
        valid_pts = safety_points[valid_mask]
        safety_trajectory = None
        if len(valid_pts) > 4:
            try:
                safety_trajectory = Trajectory.from_points(valid_pts, method='cubic')
            except Exception as e:
                print(f"Не удалось построить сплайн: {e}")

        return CorridorShadowResult(
            s_array=s_array,
            lu_points=lu_points,
            safety_points=safety_points,
            lambda_max=lambda_max,
            valid_mask=valid_mask,
            safety_trajectory=safety_trajectory,
            phi_values=phi_values
        )


def run_shadow_corridor_test():
    """
    Пример использования: тест на соосных баллонах.
    """
    import scipy.io
    from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
    from helpers.intersection import RayTracer, PiecewisePolynomialIntersection

    # Загрузка данных (как в client_corridor_1.py)
    phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                     -0.0099656628535, 2.9503573330764]
    R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
                   39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
    bound_opravka = [0, 234.27, 534.27, 768.54]
    cyl_r_opravka = 251.705
    E2 = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka,
                                        bound_opravka, cyl_r_opravka)

    phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076,
                  -0.0066486075257, 2.9473869159379]
    R_c_safe = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463,
                27152.4105364360366, -31195.5446114188999, 14397.6607910855146]
    bound_safe = [0, 327.978, 627.978, 955.956]
    cyl_r_safe = 352.387
    E1 = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)

    # Траектория безопасности
    try:
        data = scipy.io.loadmat('LU_data.mat')
        r_etalon = data['r']
        lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')
    except FileNotFoundError:
        print("Тестовый режим: случайная ЛУ")
        t_test = np.linspace(0, 100, 50)
        points = np.column_stack([
            260 * np.cos(t_test * 0.1),
            260 * np.sin(t_test * 0.1),
            t_test
        ])
        lu_trajectory = Trajectory.from_points(points, method='cubic')

    # Трассировщик
    tracer = RayTracer()
    tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())

    # Расчёт в модели тени
    calc = CorridorShadowCalculator(
        lu_trajectory=lu_trajectory,
        mandrel_surface=E2,           # E2 — оправка
        safety_surface=E1,             # E1 — безопасность
        ray_tracer=tracer,
        safe_distance=15.0
    )

    result = calc.calculate(num_points=200)

    print(f"\nРезультат:")
    print(f"  Валидных точек: {np.sum(result.valid_mask)}/{len(result.s_array)}")
    valid_lam = result.lambda_max[result.valid_mask]
    if len(valid_lam) > 0:
        print(f"  λ: [{np.min(valid_lam):.1f}, {np.max(valid_lam):.1f}]")

    return result


if __name__ == "__main__":
    result = run_shadow_corridor_test()
