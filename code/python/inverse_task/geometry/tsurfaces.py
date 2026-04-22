# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import jax
import jax.numpy as jnp

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import abc
import numpy as np
from typing import Tuple,Dict

class SurfaceBase(abc.ABC):
    @abc.abstractmethod
    def first_fundamental_form(self, u, v) -> Tuple[float, float, float]:
        pass

    @abc.abstractmethod
    def second_fundamental_form(self, u, v) -> Tuple[float, float, float]:
        pass
    
    @staticmethod
    def compute_christoffel_symbols(E, F, G, E_u, E_v, F_u, F_v, G_u, G_v):
        det = E * G - F**2
        inv_det = 1.0 / det
        
        # Обратная метрика
        g11 = G * inv_det
        g12 = -F * inv_det
        g22 = E * inv_det

        Gamma = jnp.zeros((2, 2, 2))
        
        # Строгая формула: Gamma^k_ij = 0.5 * g^{km} (dj g_{mi} + di g_{mj} - dm g_{ij})
        # где u^1=u, u^2=v. g_{11}=E, g_{12}=F, g_{22}=G
        
        # k=1 (Gamma^1_ij)
        Gamma = Gamma.at[0, 0, 0].set(0.5 * (g11 * E_u + g12 * (2.0 * F_u - E_v)))
        Gamma = Gamma.at[0, 0, 1].set(0.5 * (g11 * E_v + g12 * G_u))
        Gamma = Gamma.at[0, 1, 0].set(Gamma[0, 0, 1]) # Симметрия Gamma^1_12 == Gamma^1_21
        Gamma = Gamma.at[0, 1, 1].set(0.5 * (g11 * (2.0 * F_v - G_u) + g12 * G_v))
        
        # k=2 (Gamma^2_ij)
        Gamma = Gamma.at[1, 0, 0].set(0.5 * (g12 * E_u + g22 * (2.0 * F_u - E_v)))
        Gamma = Gamma.at[1, 0, 1].set(0.5 * (g12 * E_v + g22 * G_u))
        Gamma = Gamma.at[1, 1, 0].set(Gamma[1, 0, 1]) # Симметрия Gamma^2_12 == Gamma^2_21
        Gamma = Gamma.at[1, 1, 1].set(0.5 * (g12 * (2.0 * F_v - G_u) + g22 * G_v))
        
        return Gamma

    # @staticmethod
    # def compute_christoffel_symbols(E, F, G, E_u, E_v, F_u, F_v, G_u, G_v):
    #     det = E * G - F**2
    #     inv_det = 1.0 / det
        
    #     # Компоненты обратной метрики (контравариантный тензор)
    #     g11 = G * inv_det
    #     g12 = -F * inv_det
    #     g22 = E * inv_det

    #     # Прямая подстановка в классические формулы
    #     Gamma = jnp.zeros((2, 2, 2))
        
    #     # Gamma^1_{ij}
    #     Gamma = Gamma.at[0, 0, 0].set(0.5 * g11 * E_u + g12 * (F_u - 0.5 * E_v))
    #     Gamma = Gamma.at[0, 0, 1].set(0.5 * g11 * F_u + 0.5 * g12 * (2 * F_v - G_u))
    #     Gamma = Gamma.at[0, 1, 0].set(Gamma[0, 0, 1]) # Симметрия
    #     Gamma = Gamma.at[0, 1, 1].set(0.5 * g11 * (2 * F_v - G_u) + 0.5 * g12 * G_v)
        
    #     # Gamma^2_{ij}
    #     Gamma = Gamma.at[1, 0, 0].set(0.5 * g12 * E_u + 0.5 * g22 * (2 * F_u - E_v))
    #     Gamma = Gamma.at[1, 0, 1].set(g12 * (F_u - 0.5 * E_v) + g22 * (G_u - F_v))
    #     Gamma = Gamma.at[1, 1, 0].set(Gamma[1, 0, 1]) # Симметрия
    #     Gamma = Gamma.at[1, 1, 1].set(0.5 * g12 * (2 * F_v - G_u) + 0.5 * g22 * G_v)
        
    #     return Gamma

# =====================================================================
# ПРОМЕЖУТОЧНЫЙ КЛАСС ДЛЯ ПОВЕРХНОСТЕЙ С АНАЛИТИЧЕСКИМИ ПРОИЗВОДНЫМИ
# =====================================================================
class AnalyticalSurface(SurfaceBase):
    """
    Расширяет базовый интерфейс, требуя явного предоставления векторов 
    производных ru и rv. Это необходимо для расчета правых частей системы (3.41).
    """
    
    @abc.abstractmethod
    def derivatives(self, u: float, v: float) -> Dict[str, np.ndarray]:
        """
        Возвращает словарь с ключами:
            'r'      : точка на поверхности
            'ru'     : частная производная по u
            'rv'     : частная производная по v
            'normal' : единичная нормаль
        """
        pass

    # Методы position и normal можно определить через derivatives по умолчанию,
    # чтобы не дублировать код в наследниках (но наследник может переопределить 
    # для производительности).
    def position(self, u: float, v: float) -> np.ndarray:
        return self.derivatives(u, v)['r']
    
    def normal(self, u: float, v: float) -> np.ndarray:
        return self.derivatives(u, v)['normal']

# =====================================================================
# КОНКРЕТНЫЕ АНАЛИТИЧЕСКИЕ ПОВЕРХНОСТИ
# =====================================================================

class Cylinder(SurfaceBase):
    """Прямой круговой цилиндр радиуса R. Ось Z. u - угол, v - высота."""
    def __init__(self, radius: float = 1.0):
        self.R = radius

    def position(self, u, v):
        return (self.R * np.cos(u), self.R * np.sin(u), v)

    def normal(self, u, v):
        return (np.cos(u), np.sin(u), 0.0)

    def first_fundamental_form(self, u, v):
        R = self.R
        E = R**2
        F = 0.0
        G = 1.0
        return E, F, G

    def second_fundamental_form(self, u, v):
        # ruu = (-R cos u, -R sin u, 0), N = (cos u, sin u, 0)
        # L = ruu . N = -R
        return -self.R, 0.0, 0.0


class Sphere(SurfaceBase):
    """Сфера радиуса R. u - долгота, v - широта."""
    def __init__(self, radius: float = 1.0):
        self.R = radius

    def position(self, u, v):
        R = self.R
        return (R * np.cos(u) * np.cos(v), R * np.sin(u) * np.cos(v), R * np.sin(v))

    def normal(self, u, v):
        # Для сферы нормаль совпадает с радиус-вектором
        r = self.position(u, v)
        norm = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
        return (r[0]/norm, r[1]/norm, r[2]/norm)

    def first_fundamental_form(self, u, v):
        R = self.R
        cos_v = np.cos(v)
        E = R**2 * cos_v**2
        F = 0.0
        G = R**2
        return E, F, G

    def second_fundamental_form(self, u, v):
        R = self.R
        cos_v = np.cos(v)
        # L = r_uu . N = -R cos^2(v)
        # M = 0
        # N_coef = r_vv . N = -R
        return -R * cos_v**2, 0.0, -R


class OffsetSurface(SurfaceBase):
    """
    Смещенная поверхность (эквидистанта).
    Пример комбинации: берет базовую поверхность и сдвигает ее вдоль нормали на d.
    """
    def __init__(self, base_surface: SurfaceBase, offset_distance: float):
        self.base = base_surface
        self.d = offset_distance

    def position(self, u, v):
        bx, by, bz = self.base.position(u, v)
        nx, ny, nz = self.base.normal(u, v)
        return (
            bx + self.d * nx,
            by + self.d * ny,
            bz + self.d * nz
        )

    def normal(self, u, v):
        # У смещенной поверхности нормаль та же, что и у базовой
        return self.base.normal(u, v)

    def first_fundamental_form(self, u, v):
        """
        Здесь происходит магия AD в будущем.
        Мы НЕ выводим формулы Эйлера для смещенной поверхности.
        Мы просто вычисляем производные от новой position() через jax.grad,
        и из них получаем E, F, G.
        
        Пока в numpy это сделать нельзя, поэтому при интеграции с JAX
        этот метод будет выглядеть так:
        
        ru = jax.grad(self.position, argnums=0)(u, v)
        rv = jax.grad(self.position, argnums=1)(u, v)
        E = jnp.sum(ru * ru)
        F = jnp.sum(ru * rv)
        G = jnp.sum(rv * rv)
        return E, F, G
        """
        raise NotImplementedError("Этот класс требует JAX/PyTorch для вычисления метрики через градиенты")
        
    def second_fundamental_form(self, u, v):
        # Аналогично, через вторые производные (jax.hessian) от position()
        raise NotImplementedError("Этот класс требует JAX/PyTorch для вычисления кривизн через гессиан")


class Ellipsoid(SurfaceBase):
    """
    Общий трехосный эллипсоид.
    x = a * cos(u) * cos(v)
    y = b * sin(u) * cos(v)
    z = c * sin(v)
    u - долгота, v - широта
    """
    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        self.a = a
        self.b = b
        self.c = c

    def position(self, u, v):
        a, b, c = self.a, self.b, self.c
        return jnp.array([
            a * jnp.cos(u) * jnp.cos(v),
            b * jnp.sin(u) * jnp.cos(v),
            c * jnp.sin(v)
        ])

    def first_fundamental_form(self, u, v):
        # МАГИЯ AD: Мы не пишем формулы для E, F, G. 
        # Мы берем градиент от position по u и по v.
        ru = jax.grad(lambda u, v: self.position(u, v)[0], argnums=0)(u, v)
        rv = jax.grad(lambda u, v: self.position(u, v)[0], argnums=1)(u, v)
        
        # Так как position возвращает вектор из 3 элементов, берем градиенты от всех 3
        ru = jnp.array([jax.grad(lambda u, v: self.position(u, v)[i], argnums=0)(u, v) for i in range(3)])
        rv = jnp.array([jax.grad(lambda u, v: self.position(u, v)[i], argnums=1)(u, v) for i in range(3)])
        
        E = jnp.sum(ru * ru)
        F = jnp.sum(ru * rv)
        G = jnp.sum(rv * rv)
        return E, F, G

    def second_fundamental_form(self, u, v):
        # Вычисляем первые производные
        ru = jnp.array([jax.grad(lambda u, v: self.position(u, v)[i], argnums=0)(u, v) for i in range(3)])
        rv = jnp.array([jax.grad(lambda u, v: self.position(u, v)[i], argnums=1)(u, v) for i in range(3)])
        
        # Вектор нормали (недоединовичный, но для L,M,N это ок, так как деление на норму сократится)
        # Строго говоря, лучше единичный, чтобы числа были красивее
        N_vec = jnp.cross(ru, rv)
        N_norm = jnp.linalg.norm(N_vec) + 1e-12 # Защита от полюсов
        N_hat = N_vec / N_norm

        # Вторые производные (Гессиан)
        ruu = jnp.array([jax.grad(jax.grad(lambda u, v: self.position(u, v)[i], argnums=0), argnums=0)(u, v) for i in range(3)])
        ruv = jnp.array([jax.grad(jax.grad(lambda u, v: self.position(u, v)[i], argnums=0), argnums=1)(u, v) for i in range(3)])
        rvv = jnp.array([jax.grad(jax.grad(lambda u, v: self.position(u, v)[i], argnums=1), argnums=1)(u, v) for i in range(3)])

        L = jnp.sum(ruu * N_hat)
        M = jnp.sum(ruv * N_hat)
        N_coef = jnp.sum(rvv * N_hat)
        return L, M, N_coef

class EllipsoidAnalytical(SurfaceBase):
    """
    Эллипсоид с ПОЛНОСТЬЮ аналитическими формулами (без вложенных jax.grad).
    Это гарантирует отсутствие проблем с Higher-Order AD.
    """
    def __init__(self, a=2.0, b=1.5, c=1.0):
        self.a, self.b, self.c = a, b, c

    def first_fundamental_form(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = jnp.cos(u), jnp.sin(u)
        cos_v, sin_v = jnp.cos(v), jnp.sin(v)
        
        E = a**2 * sin_u**2 * cos_v**2 + b**2 * cos_u**2 * cos_v**2
        F = (a**2 - b**2) * sin_u * cos_u * sin_v * cos_v
        G = a**2 * cos_u**2 * sin_v**2 + b**2 * sin_u**2 * sin_v**2 + c**2 * cos_v**2
        return E, F, G

    def second_fundamental_form(self, u, v):
        a, b, c = self.a, self.b, self.c
        cos_v, sin_v = jnp.cos(v), jnp.sin(v)
        
        # Считаем определитель первой квадратичной формы (Delta)
        E, F, G = self.first_fundamental_form(u, v)
        Delta = E * G - F**2
        sqrt_Delta = jnp.sqrt(Delta + 1e-16) # Защита от деления на 0 на полюсах
        
        # Формулы для L, M, N (выведены вручную)
        L = -a * b * c * cos_v**3 / sqrt_Delta
        M = 0.0
        N_coef = -a * b * c * cos_v / sqrt_Delta
        return L, M, N_coef

    def metric_derivatives(self, u, v):
        """Аналитические производные E, F, G по u и v для символов Кристоффеля"""
        a, b, c = self.a, self.b, self.c
        cos_u, sin_u = jnp.cos(u), jnp.sin(u)
        cos_v, sin_v = jnp.cos(v), jnp.sin(v)
        
        E_u = 2 * (a**2 - b**2) * sin_u * cos_u * cos_v**2
        E_v = -2 * cos_v * sin_v * (a**2 * sin_u**2 + b**2 * cos_u**2)
        F_u = (a**2 - b**2) * jnp.cos(2*u) * sin_v * cos_v
        F_v = (a**2 - b**2) * sin_u * cos_u * jnp.cos(2*v)
        G_u = 2 * (b**2 - a**2) * sin_u * cos_u * sin_v**2
        G_v = 2 * sin_v * cos_v * (a**2 * cos_u**2 + b**2 * sin_u**2 - c**2)
        
        return E_u, E_v, F_u, F_v, G_u, G_v

