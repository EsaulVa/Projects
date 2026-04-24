"""
Прямая задача построения линии укладки на баллоне со сферическими днищами.

Используется постоянный угол геодезического отклонения θ.
Начальная точка задаётся на нижнем днище, направление — таким образом,
чтобы линия укладки прошла через цилиндрическую часть и верхнее днище.
"""

import numpy as np
import plotly.graph_objects as go

# Импорт компонентов фреймворка
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_rhs_calculator import ForwardRHS
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from solvers.scipy_solver import SciPySolver

# ----------------------------------------------------------------------
# 1. Создание составной поверхности баллона
# ----------------------------------------------------------------------
R = 1.0                # радиус цилиндра и сфер
L = 4.0                # длина цилиндрической части
z_cyl_min = -L/2
z_cyl_max =  L/2

# Сегменты
cyl_segment = CylinderSegment(R, z_cyl_min, z_cyl_max)
upper_sphere = SphereSegment(R, z_cyl_max, is_upper=True)
lower_sphere = SphereSegment(R, z_cyl_min, is_upper=False)

# Порядок сегментов: от нижней границы к верхней
balloon = CompositeSurface([lower_sphere, cyl_segment, upper_sphere])

# ----------------------------------------------------------------------
# 2. Параметры намотки
# ----------------------------------------------------------------------
# Закон отклонения: постоянный tgθ = 0 означает геодезическую (θ=0),
# однако для наглядности можно задать маленький угол, например 0.1.
tan_theta = 0.15       # для геодезической (можно 0.1 для винтового эффекта)
deviation_law = ConstantDeviation(tan_theta=tan_theta)

# Начальная точка: на нижнем днище, немного отступив от полюса.
# Возьмём u0 = 0 (нулевая долгота), v0 = v_min + 0.1, где v_min – нижняя граница баллона.
v_start = balloon.v_min +0.1  # отступим от самого низа, чтобы избежать полюса
u0 = 0.0
v0 = v_start

# Угол намотки α (угол между направлением нити и координатной линией u=const).
# Для цилиндра при α=0 нить идёт по окружности (v=const), при α=90° – вертикально вверх.
# Чтобы линия прошла через весь баллон, нужен небольшой угол наклона, например 10°.
alpha = 90*np.pi / 180    # 10 градусов

# Длина линии укладки (натуральный параметр s).
# Подберите так, чтобы нить прошла от нижнего днища до верхнего.
s_end = 13   # можно подкорректировать по результатам первого запуска

# ----------------------------------------------------------------------
# 3. Решатель и строитель прямой задачи
# ----------------------------------------------------------------------
solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)

builder = ForwardWindingBuilder(
    surface=balloon,
    deviation_law=deviation_law,
    solver=solver,
    normalize_tangent=True,
    eps=1e-12
)

# ----------------------------------------------------------------------
# 4. Запуск построения
# ----------------------------------------------------------------------
s_eval = np.linspace(0, s_end, 20)   # точки для сохранения (можно реже для скорости)
print("Построение линии укладки (прямая задача на баллоне)...")
s_vals, line_3d = builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
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
# 5. Визуализация
# ----------------------------------------------------------------------
print("Построение интерактивного графика...")

# # Сетка для всей поверхности (используем параметризацию CompositeSurface)
# u_grid = np.linspace(0, 2*np.pi, 80)
# v_grid = np.linspace(balloon.v_min, balloon.v_max, 120)
# U, V = np.meshgrid(u_grid, v_grid)
# X, Y, Z = np.vectorize(lambda u,v: balloon.position(u,v))(U, V)

# 5.1 Сетка поверхности (вместо np.vectorize)
u_grid = np.linspace(0, 2*np.pi, 80)
v_grid = np.linspace(balloon.v_min, balloon.v_max, 120)
U, V = np.meshgrid(u_grid, v_grid)
X = np.zeros_like(U, dtype=float)
Y = np.zeros_like(U, dtype=float)
Z = np.zeros_like(U, dtype=float)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = balloon.position(U[i, j], V[i, j])
        X[i, j], Y[i, j], Z[i, j] = np.array(p)

fig = go.Figure()

# Поверхность баллона
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    opacity=0.4,
    colorscale='Viridis',
    showscale=False,
    name='Баллон'
))

# Линия укладки
fig.add_trace(go.Scatter3d(
    x=line_3d[:,0], y=line_3d[:,1], z=line_3d[:,2],
    mode='lines+markers',
    line=dict(color='red', width=6),
    marker=dict(size=3),
    name='Линия укладки'
))

# Начальная точка
start_pt = balloon.position(u0, v0)
fig.add_trace(go.Scatter3d(
    x=[start_pt[0]], y=[start_pt[1]], z=[start_pt[2]],
    mode='markers',
    marker=dict(color='green', size=10, symbol='circle'),
    name='Старт'
))

# Конечная точка
end_pt = line_3d[-1]
fig.add_trace(go.Scatter3d(
    x=[end_pt[0]], y=[end_pt[1]], z=[end_pt[2]],
    mode='markers',
    marker=dict(color='blue', size=10, symbol='circle'),
    name='Финиш'
))

# Касательные в некоторых точках (опционально)
step = max(1, len(s_vals) // 10)
for i in range(0, len(s_vals), step):
    pt = line_3d[i]
    u_i, v_i = uv[i]
    geom = balloon.derivatives(u_i, v_i)
    ru = np.array(geom['ru'])
    rv = np.array(geom['rv'])
    u_prime, v_prime = tang[i]
    tau = ru * u_prime + rv * v_prime
    tau_norm = np.linalg.norm(tau)
    if tau_norm > 1e-6:
        tau = tau / tau_norm * 0.5   # масштаб для видимости
    fig.add_trace(go.Scatter3d(
        x=[pt[0], pt[0] + tau[0]],
        y=[pt[1], pt[1] + tau[1]],
        z=[pt[2], pt[2] + tau[2]],
        mode='lines',
        line=dict(color='orange', width=3),
        showlegend=False
    ))

fig.update_layout(
    title=f'Линия укладки на баллоне (прямая задача, tgθ = {tan_theta})',
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