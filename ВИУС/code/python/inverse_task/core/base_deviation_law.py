from abc import ABC, abstractmethod
import numpy as np

class DeviationLaw(ABC):
    """Абстрактный закон изменения угла геодезического отклонения."""

    @abstractmethod
    def tan_theta(self, s: float) -> float:
        """Возвращает tgθ(s) в точке с натуральным параметром s."""
        pass

    @abstractmethod
    def d_tan_theta_ds(self, s: float) -> float:
        """Производная d(tgθ)/ds в точке s (может быть приближённой)."""
        pass

    # Опционально: значение самого угла (удобно для отладки)
    def theta(self, s: float) -> float:
        return np.arctan(self.tan_theta(s))