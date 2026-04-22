# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from abc import ABC, abstractmethod
import numpy as np

class Curve1D(ABC):
    """Одномерная параметрическая кривая f(u)."""
    
    @abstractmethod
    def evaluate(self, u: float, der: int = 0) -> float:
        """Возвращает значение функции или её производной в точке u."""
        pass
    
    @property
    @abstractmethod
    def domain(self) -> tuple[float, float]:
        """Область определения параметра u: (u_min, u_max)."""
        pass