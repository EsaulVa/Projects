# from trajectory import *
# from register_factory import *
import numpy as np
import sys
from pathlib import Path
# Добавляем корень проекта в пути поиска модулей, чтобы работали импорты типа core.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.trajectory import Trajectory
# from core import register_factory   # чтобы выполнилась регистрация
# Генерируем зашумлённые точки вдоль винтовой линии
t = np.linspace(0, 4*np.pi, 300)
points = np.column_stack([np.cos(t), np.sin(t), t]) + 0.0 * np.random.randn(len(t), 3)

# Траектория с кубическими сплайнами (быстро, но менее гладко)
traj_cubic = Trajectory.from_points(points, method='cubic', bc_type='natural')

# # Траектория со сглаживающими квинтиками (медленнее, но C² и устойчиво к шуму)
# traj_smooth = Trajectory.from_points(points, method='quintic_smooth', alpha=0.95)

# Теперь обе траектории предоставляют единый интерфейс R(z) и R_deriv(z)
z = 2.5*np.sqrt(2)
print(traj_cubic.R(z))
print(traj_cubic.R_deriv(z))