# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# cubic_spline_curve.py
from scipy.interpolate import CubicSpline
from .curve_interface import Curve1D

class CubicSplineCurve(Curve1D):
    def __init__(self, u: np.ndarray, values: np.ndarray, bc_type='natural'):
        self._u = np.asarray(u)
        self._values = np.asarray(values)
        self._spline = CubicSpline(self._u, self._values, bc_type=bc_type)

    def evaluate(self, u: float, der: int = 0) -> float:
        if der > 2:
            raise ValueError("CubicSpline supports derivatives up to order 2")
        return float(self._spline(u, nu=der))

    @property
    def domain(self) -> tuple[float, float]:
        return (self._u[0], self._u[-1])