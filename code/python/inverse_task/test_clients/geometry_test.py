import numpy as np
import sys
from pathlib import Path
# Добавляем корень проекта в пути поиска модулей, чтобы работали импорты типа core.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.tsurfaces import *
from geometry.ellipsoid import EllipsoidWithDerivatives
# # =====================================================================
# # НОВЫЙ КЛАСС-АДАПТЕР (полностью самодостаточный)
# # =====================================================================
# class EllipsoidWithDerivatives(AnalyticalSurface):
#     """
#     Эллипсоид, расширенный методами position, normal и derivatives.
#     Использует композицию с EllipsoidAnalytical для переиспользования
#     формул квадратичных форм.
#     """
#     def __init__(self, a=2.0, b=1.5, c=1.0):
#         # Внутренний объект для делегирования вычисления метрики и кривизн
#         self._ellipsoid = EllipsoidAnalytical(a, b, c)
#         self.a, self.b, self.c = a, b, c

#     # -----------------------------------------------------------------
#     # ГЕОМЕТРИЧЕСКИЕ МЕТОДЫ (реализованы здесь)
#     # -----------------------------------------------------------------
#     def position(self, u, v):
#         """Точка на поверхности эллипсоида."""
#         a, b, c = self.a, self.b, self.c
#         cos_u, sin_u = jnp.cos(u), jnp.sin(u)
#         cos_v, sin_v = jnp.cos(v), jnp.sin(v)
#         return jnp.array([
#             a * cos_u * cos_v,
#             b * sin_u * cos_v,
#             c * sin_v
#         ])

#     def normal(self, u, v):
#         """Единичная нормаль к эллипсоиду."""
#         a, b, c = self.a, self.b, self.c
#         cos_u, sin_u = jnp.cos(u), jnp.sin(u)
#         cos_v, sin_v = jnp.cos(v), jnp.sin(v)
#         # Ненормированная нормаль (градиент неявной функции x²/a² + y²/b² + z²/c² = 1)
#         nx = cos_u * cos_v / a
#         ny = sin_u * cos_v / b
#         nz = sin_v / c
#         n = jnp.array([nx, ny, nz])
#         return n / jnp.linalg.norm(n)

#     # -----------------------------------------------------------------
#     # ДЕЛЕГИРОВАНИЕ ВНУТРЕННЕМУ ОБЪЕКТУ
#     # -----------------------------------------------------------------
#     def first_fundamental_form(self, u, v):
#         return self._ellipsoid.first_fundamental_form(u, v)

#     def second_fundamental_form(self, u, v):
#         return self._ellipsoid.second_fundamental_form(u, v)

#     def metric_derivatives(self, u, v):
#         """Делегируем вызов metric_derivatives, если он есть."""
#         if hasattr(self._ellipsoid, 'metric_derivatives'):
#             return self._ellipsoid.metric_derivatives(u, v)
#         else:
#             raise NotImplementedError("metric_derivatives не реализован в EllipsoidAnalytical")

#     # -----------------------------------------------------------------
#     # НОВЫЙ МЕТОД, ТРЕБУЕМЫЙ AnalyticalSurface
#     # -----------------------------------------------------------------
#     def derivatives(self, u, v) -> Dict[str, jnp.ndarray]:
#         """
#         Возвращает точку, частные производные и нормаль.
#         Необходим для системы (3.41).
#         """
#         a, b, c = self.a, self.b, self.c
#         cos_u, sin_u = jnp.cos(u), jnp.sin(u)
#         cos_v, sin_v = jnp.cos(v), jnp.sin(v)
        
#         r = self.position(u, v)
#         ru = jnp.array([
#             -a * sin_u * cos_v,
#              b * cos_u * cos_v,
#              0.0
#         ])
#         rv = jnp.array([
#             -a * cos_u * sin_v,
#             -b * sin_u * sin_v,
#              c * cos_v
#         ])
#         normal = self.normal(u, v)
        
#         return {'r': r, 'ru': ru, 'rv': rv, 'normal': normal}

# =====================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =====================================================================
if __name__ == "__main__":
    # Создаем новый эллипсоид с поддержкой derivatives
    ellipsoid = EllipsoidWithDerivatives(a=3.0, b=2.0, c=1.5)
    
    u, v = 0.5, 0.3
    geom = ellipsoid.derivatives(u, v)
    print(f"Point: {geom['r']}")
    print(f"ru: {geom['ru']}")
    print(f"rv: {geom['rv']}")
    print(f"Normal: {geom['normal']}")
    
    # Проверяем, что делегирование работает
    E, F, G = ellipsoid.first_fundamental_form(u, v)
    print(f"First form: E={E:.3f}, F={F:.3f}, G={G:.3f}")
    
    L, M, N = ellipsoid.second_fundamental_form(u, v)
    print(f"Second form: L={L:.3f}, M={M:.3f}, N={N:.3f}")