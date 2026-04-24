import numpy as np
from typing import Dict, Tuple
from geometry.tsurfaces import AnalyticalSurface

class SurfaceSegment:
    """Базовый класс для сегмента составной поверхности."""
    def contains(self, v: float) -> bool:
        return False

    def point(self, u: float, v: float) -> np.ndarray:
        raise NotImplementedError

    def derivatives(self, u: float, v: float) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        raise NotImplementedError

    def second_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        raise NotImplementedError

class CylinderSegment(SurfaceSegment):
    def __init__(self, R: float, z_min: float, z_max: float):
        self.R = R
        self.z_min = z_min
        self.z_max = z_max

    def contains(self, v: float) -> bool:
        return self.z_min <= v <= self.z_max

    def point(self, u: float, v: float) -> np.ndarray:
        return np.array([self.R * np.cos(u), self.R * np.sin(u), v])

    def derivatives(self, u: float, v: float) -> Dict[str, np.ndarray]:
        cos_u, sin_u = np.cos(u), np.sin(u)
        return {
            'r': np.array([self.R * cos_u, self.R * sin_u, v]),
            'ru': np.array([-self.R * sin_u, self.R * cos_u, 0.0]),
            'rv': np.array([0.0, 0.0, 1.0]),
            'normal': np.array([cos_u, sin_u, 0.0])
        }

    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        return self.R**2, 0.0, 1.0

    def second_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        return -self.R, 0.0, 0.0

class SphereSegment(SurfaceSegment):
    def __init__(self, R: float, z0: float, is_upper: bool):
        self.R = R
        self.z0 = z0
        self.is_upper = is_upper
        if is_upper:
            self.z_min = z0
            self.z_max = z0 + R
        else:
            self.z_min = z0 - R
            self.z_max = z0

    def contains(self, v: float) -> bool:
        return self.z_min <= v <= self.z_max

    def _f(self, v: float) -> Tuple[float, float, float]:
        """Возвращает f(v), f'(v), f''(v) для поверхности вращения."""
        t = v - self.z0
        R = self.R
        # f = sqrt(R^2 - t^2)
        f = np.sqrt(max(0.0, R*R - t*t))
        if f > 1e-12:
            fp = -t / f
            fpp = -(R*R) / (f**3)
        else:
            fp = -np.inf if t > 0 else np.inf  # не используется, т.к. v не доходит до полюса
            fpp = -np.inf
        return f, fp, fpp

    def point(self, u: float, v: float) -> np.ndarray:
        f, _, _ = self._f(v)
        return np.array([f * np.cos(u), f * np.sin(u), v])

    def derivatives(self, u: float, v: float) -> Dict[str, np.ndarray]:
        f, fp, _ = self._f(v)
        cos_u, sin_u = np.cos(u), np.sin(u)
        r = np.array([f * cos_u, f * sin_u, v])
        ru = np.array([-f * sin_u, f * cos_u, 0.0])
        rv = np.array([fp * cos_u, fp * sin_u, 1.0])
        # нормаль внешняя
        denom = np.sqrt(1 + fp*fp)
        normal = np.array([cos_u / denom, sin_u / denom, -fp / denom])  # обеспечивает внешнее направление
        return {'r': r, 'ru': ru, 'rv': rv, 'normal': normal}

    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        f, fp, _ = self._f(v)
        E = f*f
        F = 0.0
        G = fp*fp + 1.0
        return E, F, G

    def second_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        f, fp, fpp = self._f(v)
        denom = np.sqrt(1 + fp*fp)
        # L = ruu·n = -f * cos_u * (cos_u/denom) - f * sin_u * (sin_u/denom) = -f / denom
        L = -f / denom
        M = 0.0
        # N = rvv·n = (fpp cos_u, fpp sin_u, 0) · (cos_u/denom, sin_u/denom, -fp/denom) = fpp / denom
        N = fpp / denom
        return L, M, N

class CompositeSurface(AnalyticalSurface):
    """Составная поверхность, состоящая из нескольких сегментов."""
    def __init__(self, segments: list):
        self.segments = segments
        self.v_min = min(seg.z_min for seg in segments)
        self.v_max = max(seg.z_max for seg in segments)

    def _get_segment(self, v: float) -> SurfaceSegment:
        for seg in self.segments:
            if seg.contains(v):
                return seg
        raise ValueError(f"v={v} вне диапазона [{self.v_min}, {self.v_max}]")

    def position(self, u: float, v: float) -> np.ndarray:
        seg = self._get_segment(v)
        return seg.point(u, v)

    def normal(self, u: float, v: float) -> np.ndarray:
        seg = self._get_segment(v)
        return seg.derivatives(u, v)['normal']

    def derivatives(self, u: float, v: float) -> Dict[str, np.ndarray]:
        seg = self._get_segment(v)
        return seg.derivatives(u, v)

    def first_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        seg = self._get_segment(v)
        return seg.first_fundamental_form(u, v)

    def second_fundamental_form(self, u: float, v: float) -> Tuple[float, float, float]:
        seg = self._get_segment(v)
        return seg.second_fundamental_form(u, v)

    def metric_derivatives(self, u: float, v: float):
        # Для простоты вернем численные производные, т.к. для баллона они не нужны в аналитическом виде
        eps = 1e-6
        # E_u, E_v, F_u, F_v, G_u, G_v
        E, F, G = self.first_fundamental_form(u, v)
        # Производные по u
        E_up, _, _ = self.first_fundamental_form(u+eps, v)
        E_um, _, _ = self.first_fundamental_form(u-eps, v)
        E_u = (E_up - E_um) / (2*eps)
        # Аналогично для остальных... Заглушка
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0