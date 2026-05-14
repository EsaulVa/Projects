"""
Контрольный пример построения линии укладки на эллипсоиде (прямая задача).

На поверхности эллипсоида строится линия укладки с заданным постоянным
углом геодезического отклонения θ. Начальная точка задана на экваторе,
направление определяется углом намотки α.

Результат визуализируется в интерактивном 3D-графике (Plotly).
"""

import numpy as np
import plotly.graph_objects as go

# Импорт компонентов фреймворка
from geometry.ellipsoid import EllipsoidWithDerivatives
from geometry.cylinder import CylinderWithDerivatives,CylinderAnalytical
from core.const_dev_law import ConstantDeviation
# from forward_winding.forward_rhs_calculator import ForwardRHS
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from solvers.scipy_solver import SciPySolver

# ----------------------------------------------------------------------
# 1. Параметры поверхности
# ----------------------------------------------------------------------
# Эллипсоид с полуосями a, b, c
# a, b, c = 3.0, 2.0, 1.5
# surface = EllipsoidWithDerivatives(a, b, c)

surface = CylinderAnalytical(radius=4.0)
deviation_law = ConstantDeviation(tan_theta=0.0)
u0, v0 = 0.0, 0.0
alpha = np.pi / 6        # 30 градусов
s_end =30.0
# ----------------------------------------------------------------------
# 2. Параметры закона отклонения и численного интегрирования
# ----------------------------------------------------------------------
# Постоянный угол геодезического отклонения (tgθ = 0.15)
# deviation_law = ConstantDeviation(tan_theta=0.1)

# Решатель ОДУ (SciPy DOP853 для высокой точности)
solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)

# Строитель линии укладки (прямая задача)
builder = ForwardWindingBuilder(
    surface=surface,
    deviation_law=deviation_law,
    solver=solver,
    normalize_tangent=True,
    eps=1e-12
)

# # ----------------------------------------------------------------------
# # 3. Начальные условия
# # ----------------------------------------------------------------------
# # Начальная точка на поверхности (u = долгота, v = широта)
# u0, v0 = 0.0, 0.0           # экватор, пересечение с положительной осью X

# # Начальное направление: задаём углом намотки α (отсчитывается от ru)
# alpha = 0*np.pi / 2            # 30 градусов

# # Длина линии укладки (натуральный параметр s)
# s_end = 12.0

# Точки вывода – равномерная сетка по s
s_eval = np.linspace(0, s_end, 100)

# ----------------------------------------------------------------------
# 4. Запуск построения
# ----------------------------------------------------------------------
print("Построение линии укладки (прямая задача)...")
s_vals, line_3d = builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),   # задаём угол
    eval_points=s_eval
)

if builder.last_run_successful:
    print(f"Рассчитано {len(s_vals)} точек линии укладки.")
    uv = builder.get_uv_states()
    tang = builder.get_tangents()
    print(f"Диапазон u: [{np.min(uv[:,0]):.3f}, {np.max(uv[:,0]):.3f}]")
    print(f"Диапазон v: [{np.min(uv[:,1]):.3f}, {np.max(uv[:,1]):.3f}]")
else:
    print("Расчёт завершился с ошибкой.")
    exit(1)

# ----------------------------------------------------------------------
# 5. Визуализация с Plotly
# ----------------------------------------------------------------------
print("Построение интерактивного графика...")

fig = go.Figure()

# 5.1. Поверхность эллипсоида (полупрозрачная)
# u_grid = np.linspace(0, 2*np.pi, 80)
# v_grid = np.linspace(-np.pi/2, np.pi/2, 50)
# U, V = np.meshgrid(u_grid, v_grid)
# X = np.zeros_like(U)
# Y = np.zeros_like(U)
# Z = np.zeros_like(U)
# for i in range(U.shape[0]):
#     for j in range(U.shape[1]):
#         p = surface.position(U[i,j], V[i,j])
#         X[i,j], Y[i,j], Z[i,j] = p

# fig.add_trace(go.Surface(
#     x=X, y=Y, z=Z,
#     opacity=0.4,
#     colorscale='Viridis',
#     showscale=False,
#     name='Эллипсоид'
# ))
# 5.1. Поверхность цилиндра (полупрозрачная)
u_grid = np.linspace(0, 2*np.pi, 80)          # полный оборот по углу
v_grid = np.linspace(-1, s_end + 1, 100)      # достаточный диапазон высот
U, V = np.meshgrid(u_grid, v_grid)
X = np.zeros_like(U)
Y = np.zeros_like(U)
Z = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = surface.position(U[i,j], V[i,j])
        X[i,j], Y[i,j], Z[i,j] = p
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    opacity=0.4,
    colorscale='Blues',
    showscale=False,
    name='Цилиндр'
))
# 5.2. Линия укладки
fig.add_trace(go.Scatter3d(
    x=line_3d[:,0], y=line_3d[:,1], z=line_3d[:,2],
    mode='lines',
    line=dict(color='red', width=6),
    name='Линия укладки'
))

# 5.3. Начальная точка
start_point = surface.position(u0, v0)
fig.add_trace(go.Scatter3d(
    x=[start_point[0]], y=[start_point[1]], z=[start_point[2]],
    mode='markers',
    marker=dict(color='green', size=8, symbol='circle'),
    name='Начало'
))

# 5.4. Конечная точка
end_point = line_3d[-1]
fig.add_trace(go.Scatter3d(
    x=[end_point[0]], y=[end_point[1]], z=[end_point[2]],
    mode='markers',
    marker=dict(color='blue', size=8, symbol='circle'),
    name='Конец'
))

# 5.5. Касательные в некоторых точках для иллюстрации направления
step = len(s_vals) // 10
for i in range(0, len(s_vals), step):
    pt = line_3d[i]
    u_i, v_i = uv[i]
    geom = surface.derivatives(u_i, v_i)
    ru = geom['ru']
    rv = geom['rv']
    u_prime, v_prime = tang[i]
    tau = ru * u_prime + rv * v_prime
    tau_len = np.linalg.norm(tau)
    if tau_len > 1e-6:
        tau = tau / tau_len * 0.5  # масштаб для отображения
    fig.add_trace(go.Scatter3d(
        x=[pt[0], pt[0] + tau[0]],
        y=[pt[1], pt[1] + tau[1]],
        z=[pt[2], pt[2] + tau[2]],
        mode='lines',
        line=dict(color='orange', width=3),
        showlegend=False
    ))

# Настройка макета
fig.update_layout(
    title=f'Линия укладки на эллипсоиде (tgθ = {deviation_law.tan_theta(0):.2f})',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    width=1000,
    height=800,
    hovermode='closest'
)

fig.show()
print("Готово.")