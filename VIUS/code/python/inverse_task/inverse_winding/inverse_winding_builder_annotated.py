#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inverse_winding_builder_annotated.py
===================================
Аннотированная версия InvWindingLineBuilder + InverseWindingLineBuilder.

НАУЧНОЕ ОБОСНОВАНИЕ
-------------------
Этот модуль реализует ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ системы (3.41) Савина А.Г.
для восстановления линии укладки по известной траектории точки схода.

АРХИТЕКТУРА
-----------
• RightHandSideCalculator — вычисляет du/dz, dv/dz (см. rhs_calculator_annotated.py).
• InvWindingLineBuilder — инкапсулирует процесс интегрирования.
• InverseWindingLineBuilder — адаптер под интерфейс WindingLineBuilderBase.

ПРИНЦИП РАБОТЫ
--------------
1. Задаётся начальная точка (u₀, v₀) на поверхности оправки.
2. SciPySolver интегрирует систему ОДУ du/dz = f₁(u,v,z), dv/dz = f₂(u,v,z)
   от z=0 до z=z_end (полная длина траектории ТСН).
3. Решатель использует адаптивный шаг (DOP853) — автоматически уменьшает
   шаг в зонах с резкими изменениями геометрии.
4. Полученные (uᵢ, vᵢ) преобразуются в 3D-точки r(uᵢ, vᵢ).
5. Вычисляются невязки δᵢ = <(R(zᵢ)-rᵢ)/|R-r|, mᵢ> для диагностики.

ОГРАНИЧЕНИЯ КЛАССИЧЕСКОГО ПОДХОДА
----------------------------------
1. Нет ЯВНОГО корректора Ньютона. Коррекция встроена только в rhs
   через пропорциональный член k·δ/II. Это уменьшает, но не обнуляет δ.

2. Дрейф точки от поверхности. Поскольку интегрируется в (u,v),
   а затем проецируется на r(u,v), точка формально лежит на поверхности.
   НО: если (u,v) получены с ошибкой, невязка δ растёт.
   В отчёте это называется "неустранимым накоплением геометрической ошибки".

3. Начальное приближение (u₀, v₀) должно быть точным.
   Если Φ₀ ≠ 0, вся траектория интегрируется с систематическим сдвигом.
   В отчёте предлагается newton_corrector для коррекции начальной точки.

СРАВНЕНИЕ С DAE-ПОДХОДОМ
------------------------
В отчёте предлагается схема предиктор-корректор:
• Предиктор — явный Эйлер (или Рунге-Кутта) в параметрах (u,v).
• Корректор — итерации Ньютона до Φ=0 на каждом шаге.
• Адаптивная бисекция — при скачках.

Этот модуль использует ТОЛЬКО предиктор (через SciPySolver).
Корректор отсутствует. Для сравнения обоих подходов рекомендуется
запустить один и тот же тест:
  а) с InvWindingLineBuilder (классика),
  б) с inverse_winding_hybrid (DAE, отчёт).
"""

import numpy as np
from typing import Optional, Tuple
from geometry.tsurfaces import AnalyticalSurface
from core.trajectory import Trajectory
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.base_solver import ODESolver
from winding.winding_base import WindingLineBuilderBase, WindingResultProvider


class InvWindingLineBuilder:
    """
    Построитель линии укладки на поверхности оправки.

    Инкапсулирует численное интегрирование системы (3.41) Савина.
    Не занимается визуализацией или сохранением — только расчёт.
    """

    def __init__(
        self,
        surface: AnalyticalSurface,
        trajectory: Trajectory,
        rhs_calculator: RightHandSideCalculator,
        solver: ODESolver
    ):
        """
        Параметры
        ---------
        surface : AnalyticalSurface
            Поверхность оправки (предоставляет геометрию и квадратичные формы).
        trajectory : Trajectory
            Траектория точки схода нити R(z) с натуральной параметризацией.
        rhs_calculator : RightHandSideCalculator
            Вычислитель правых частей du/dz, dv/dz (система 3.41).
        solver : ODESolver
            Решатель ОДУ (например, SciPySolver с методом DOP853).
            DOP853 — адаптивный метод Рунге-Кутта 8(7) порядка,
            подходит для гладких задач (эллипсоиды, цилиндры).
            Для жёстких задач рекомендуется BDF.
        """
        self._surface = surface
        self._trajectory = trajectory
        self._rhs_calc = rhs_calculator
        self._solver = solver
        self._diagnostics = {}

        # Результаты последнего расчёта (None до первого вызова compute)
        self._z_values: Optional[np.ndarray] = None
        self._uv_states: Optional[np.ndarray] = None   # массив (N, 2) с [u, v]
        self._points_3d: Optional[np.ndarray] = None   # массив (N, 3) с точками ЛУ

    def compute(
        self,
        u0: float,
        v0: float,
        z_end: Optional[float] = None,
        z_eval: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запускает интегрирование системы (3.41) от z=0 до z=z_end.

        Параметры
        ---------
        u0, v0 : float
            Начальные криволинейные координаты на поверхности.
            ДОЛЖНЫ соответствовать началу траектории (z=0).
            Если u0, v0 не удовлетворяют условию контакта Φ=0,
            вся траектория будет смещена (систематическая ошибка).
            Рекомендация: скорректировать через newton_corrector перед вызовом.
        z_end : float, optional
            Конечное значение параметра z (длина дуги ТСН).
            Если None, используется trajectory.total_length.
        z_eval : np.ndarray, optional
            Явный массив значений z, в которых требуется решение.
            Если None, решатель выбирает точки адаптивно (по точности).
            Для сравнения с DAE-подходом рекомендуется задавать равномерную
            сетку z_eval = np.linspace(0, z_end, N) — тогда шаг фиксирован.
        **solver_kwargs
            Дополнительные аргументы для solver.solve_with_diagnostics().
            Например: rtol=1e-8, atol=1e-10, max_step=0.5.

        Возвращает
        ----------
        z_values : np.ndarray
            Значения z, в которых вычислено решение.
        points_3d : np.ndarray, shape (N, 3)
            Соответствующие 3D-точки линии укладки на поверхности.
        """
        # Определяем конечную точку интегрирования
        if z_end is None:
            z_end = self._trajectory.total_length

        # Обёртка правых частей в формат, ожидаемый решателем ОДУ.
        # SciPySolver (и аналоги) требуют функцию fun(z, state) -> derivatives.
        # state = [u, v], derivatives = [du/dz, dv/dz].
        def rhs_wrapper(z: float, state: np.ndarray) -> np.ndarray:
            du, dv = self._rhs_calc(z, state)
            return np.array([du, dv])

        # Начальный вектор состояния [u₀, v₀]
        y0 = np.array([u0, v0], dtype=float)

        # === ВЫЗОВ РЕШАТЕЛЯ ОДУ ===
        # SciPySolver.solve_with_diagnostics() возвращает:
        #   z_vals — массив точек, где вычислено решение,
        #   uv — массив состояний [u, v] в этих точках,
        #   diag — словарь с информацией о сходимости.
        #
        # ПРИМЕЧАНИЕ: решатель использует адаптивный шаг.
        # Он сам уменьшает шаг при росте локальной ошибки,
        но НЕ корректирует невязку Φ — это задача rhs_calculator.
        try:
            z_vals, uv, diag = self._solver.solve_with_diagnostics(
                fun=rhs_wrapper,
                t_span=(0.0, z_end),
                y0=y0,
                t_eval=z_eval,
                **solver_kwargs
            )
        except Exception as e:
            # Исключение на этапе интегрирования — обычно из-за:
            # • выхода u за границы поверхности,
            # • вырождения метрики (det=0),
            # • слишком большого шага (невязка взрывается).
            self._diagnostics = {
                'success': False,
                'message': f'Исключение при интегрировании: {str(e)}',
                'num_points': 0,
                'final_param': diag.get('final_t', None) if 'diag' in dir() else None
            }
            self._success = False
            return np.array([]), np.array([])

        # Сохраняем диагностику решателя
        self._diagnostics = {
            'success': diag['success'],
            'message': diag['message'],
            'num_points': len(z_vals),
            'final_param': diag['final_t'],
            'solver_message': diag.get('solver_message', '')
        }

        if not diag['success']:
            # Решатель сообщил о неуспехе (например, достигнут max_step,
            # или локальная ошибка превысила допуск).
            self._success = False
            return z_vals, np.array([])

        # Сохраняем сырые результаты интегрирования
        self._z_values = z_vals
        self._uv_states = uv

        # Преобразуем криволинейные координаты (u, v) в 3D-точки на поверхности
        # через r(u, v). Это гарантирует, что точки ЛУ ЛЕЖАТ на поверхности
        # (в отличие от интегрирования в 3D, где точка может дрейфовать).
        # Однако если (u, v) получены с ошибкой, невязка Φ будет ненулевой.
        self._points_3d = np.array([
            self._surface.position(u, v) for u, v in uv
        ])

        return self._z_values, self._points_3d

    def get_uv_states(self) -> Optional[np.ndarray]:
        """
        Возвращает массив криволинейных координат (u, v) после последнего расчёта.

        Returns
        -------
        np.ndarray or None
            Массив формы (N, 2) с колонками [u, v].
            None, если compute() ещё не вызывался.
        """
        return self._uv_states

    def get_z_values(self) -> Optional[np.ndarray]:
        """Возвращает массив z, на которых было получено решение."""
        return self._z_values

    def get_3d_points(self) -> Optional[np.ndarray]:
        """Возвращает 3D-точки линии укладки r(u,v)."""
        return self._points_3d

    def get_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет невязку δ(z) = <(R(z)-r(z))/|R-r|, m(z)> для всех точек.

        Это КЛЮЧЕВАЯ ДИАГНОСТИКА качества решения.
        Идеально: |δ| ≈ 0 везде.
        Реально: |δ| растёт с длиной траектории (накопление ошибки).

        Returns
        -------
        z_vals : np.ndarray
        deltas : np.ndarray
            Невязки δ в каждой точке. Массив той же длины, что z_vals.
        """
        if self._z_values is None or self._uv_states is None:
            raise RuntimeError("Сначала выполните compute().")

        z_vals = self._z_values
        uv = self._uv_states
        deltas = np.zeros_like(z_vals)

        for i in range(len(z_vals)):
            z = z_vals[i]
            u, v = uv[i]

            # Точка схода нити (ТСН)
            R = self._trajectory.R(z)
            # Точка на оправке (ЛУ)
            r = self._surface.position(u, v)
            # Нормаль к оправке
            n = self._surface.normal(u, v)

            # Вектор нити
            diff = np.array(R) - np.array(r)
            diff_norm = np.linalg.norm(diff)

            if diff_norm > 1e-12:
                # Невязка δ = <(R-r)/|R-r|, m> — формула (3.18) Савина
                delta = np.dot(diff / diff_norm, np.array(n))
            else:
                delta = 0.0
            deltas[i] = delta

        return z_vals, deltas


# ============================================================================
# АДАПТЕР: InverseWindingLineBuilder
# ============================================================================
# Этот класс адаптирует InvWindingLineBuilder к интерфейсу WindingLineBuilderBase,
# используемому в остальной части проекта (в частности, в ForwardWindingBuilder).
# Позволяет использовать обратную задачу в тех же клиентах, что и прямую.
# ============================================================================

class InverseWindingLineBuilder(WindingLineBuilderBase, WindingResultProvider):
    """
    Адаптер обратной задачи под общий интерфейс WindingLineBuilderBase.

    Позволяет использовать обратную задачу (восстановление ЛУ по ТСН)
    в тех же клиентах, что и прямую задачу (построение ТСН по ЛУ).
    """

    def __init__(self, inverse_builder: InvWindingLineBuilder):
        """
        Параметры
        ---------
        inverse_builder : InvWindingLineBuilder
            Экземпляр построителя с уже настроенными surface, trajectory,
            rhs_calculator и solver.
        """
        self._builder = inverse_builder
        self._last_success = False
        self._diagnostics = {}

    def build(
        self,
        initial_point,
        initial_tangent=None,
        end_param=None,
        eval_points=None,
        **kwargs
    ):
        """
        Запускает построение линии укладки.

        Параметры
        ---------
        initial_point : tuple (u0, v0)
            Начальные криволинейные координаты на поверхности.
        initial_tangent : tuple, optional
            В обратной задаче начальное направление определяется траекторией,
            поэтому этот параметр игнорируется (совместимость с прямой задачей).
        end_param : float, optional
            Конечное значение z. Если None — используется полная длина.
        eval_points : np.ndarray, optional
            Явный массив z для вывода решения.
        **kwargs
            Дополнительные аргументы, передаются в compute().

        Returns
        -------
        z_vals : np.ndarray
        points : np.ndarray, shape (N, 3)
            3D-точки линии укладки.
        """
        u0, v0 = initial_point

        # initial_tangent не используется в обратной задаче:
        # направление нити в начальной точке определяется автоматически
        # из геометрии (R(0) - r(u0,v0)).
        if initial_tangent is not None:
            pass  # совместимость с интерфейсом прямой задачи

        z_vals, points = self._builder.compute(
            u0, v0,
            z_end=end_param,
            z_eval=eval_points,
            **kwargs
        )

        self._diagnostics = self._builder._diagnostics
        self._last_success = self._diagnostics.get('success', False)
        return z_vals, points

    def get_residuals(self):
        """Возвращает невязки δ(z) для диагностики."""
        return self._builder.get_residuals()

    def get_uv_states(self):
        """Возвращает криволинейные координаты (u, v)."""
        return self._builder.get_uv_states()

    def get_tangents(self):
        """
        Возвращает касательные векторы к ЛУ.

        В обратной задаче касательные можно вычислить из внутренних данных,
        но по умолчанию возвращается None (если не требуется).
        """
        return None

    def get_3d_points(self):
        """Возвращает 3D-точки линии укладки."""
        return self._builder.get_3d_points()

    @property
    def last_run_successful(self):
        """Флаг успешности последнего расчёта."""
        return self._last_success

    def get_diagnostics(self) -> dict:
        """Возвращает словарь с диагностической информацией."""
        return self._diagnostics
