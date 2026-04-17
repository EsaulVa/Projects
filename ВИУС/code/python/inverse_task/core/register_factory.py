# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# curve_builders.py
from .curve_factory import CurveFactory
from .cubic_spline_curve import CubicSplineCurve
# from smoothing_quintic_curve import SmoothingQuinticCurve   # ваша реализация
# from geomdl_curve_adapter import GeomdlCurveAdapter         # обёртка NURBS

@CurveFactory.register('cubic')
def build_cubic(u, values, **kwargs):
    bc_type = kwargs.get('bc_type', 'natural')
    return CubicSplineCurve(u, values, bc_type)

# @CurveFactory.register('quintic_smooth')
# def build_quintic_smooth(u, values, **kwargs):
#     alpha = kwargs.get('alpha', 0.9)
#     bc_left = kwargs.get('bc_left', {'m': 0.0, 'M': 0.0})
#     bc_right = kwargs.get('bc_right', {'m': 0.0, 'M': 0.0})
#     return SmoothingQuinticCurve(u, values, alpha, bc_left, bc_right)

# @CurveFactory.register('nurbs')
# def build_nurbs(u, values, **kwargs):
#     degree = kwargs.get('degree', 3)
#     # предполагается, что GeomdlCurveAdapter принимает точки и степень
#     return GeomdlCurveAdapter(values, degree)  # параметр u внутри вычисляется автоматически