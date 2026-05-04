"""
Комбинированный пример: прямая задача → траектория → обратная задача.

1. На внешнем эллипсоиде E1 строится линия укладки с постоянным
   углом геодезического отклонения θ.
2. Полученная 3D-линия объявляется траекторией точки схода (R(z)).
3. Для внутреннего эллипсоида E2 решается обратная задача: по траектории R(z)
   восстанавливается линия укладки на E2.
4. Результаты визуализируются: E1, E2, исходная линия, восстановленная линия,
   а также соединительные отрезки между соответствующими точками.

Ожидаемый эффект: исходная линия (на большом эллипсоиде) и восстановленная
(на меньшем) геометрически похожи, но отличаются из-за разной кривизны.
"""

import numpy as np
import plotly.graph_objects as go

from geometry.tsurfaces import EllipsoidAnalytical
from geometry.ellipsoid import EllipsoidWithDerivatives
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_rhs_calculator import ForwardRHS
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.scipy_solver import SciPySolver
from inverse_winding.inverse_winding_builder import InvWindingLineBuilder, InverseWindingLineBuilder

# ----------------------------------------------------------------------
# 1. Параметры эллипсоидов
# ----------------------------------------------------------------------
# Внешний эллипсоид E1 (поверхность безопасности)
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)

# Внутренний эллипсоид E2 (оправка) — соосный и уменьшенный
scale = 0.8
a2, b2, c2 = a1 * scale, b1 * scale, c1 * scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)

# ----------------------------------------------------------------------
# 2. Прямая задача: построение линии укладки на E1
# ----------------------------------------------------------------------
print("===== Прямая задача: построение линии укладки на E1 =====")

# Закон отклонения (постоянный малый угол)
deviation_law = ConstantDeviation(tan_theta=0.1)

# Решатель ОДУ
solver_forward = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)

# Строитель прямой задачи
forward_builder = ForwardWindingBuilder(
    surface=E1,
    deviation_law=deviation_law,
    solver=solver_forward,
    normalize_tangent=True,
    eps=1e-12
)

# Начальные условия на E1
u0, v0 = 0.0, 0.0          # экватор
alpha = np.pi / 6          # угол намотки 30°
s_end = 30.0                # длина линии
count_points=100
s_eval = np.linspace(0, s_end, count_points)

# Запуск прямой задачи
s_vals, line_E1 = forward_builder.build(
    initial_point=(u0, v0),
    initial_tangent=(alpha,),
    eval_points=s_eval
)

if not forward_builder.last_run_successful:
    raise RuntimeError("Прямая задача завершилась с ошибкой")

uv_E1 = forward_builder.get_uv_states()
print(f"Прямая задача: построено {len(s_vals)} точек на E1")

# ----------------------------------------------------------------------
# 3. Создание траектории точки схода из линии укладки на E1
# ----------------------------------------------------------------------
print("\n===== Построение траектории точки схода по точкам линии =====")

# Метод кубического сплайна по 3D-точкам
traj = Trajectory.from_points(line_E1, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 4. Обратная задача: восстановление линии укладки на E2
# ----------------------------------------------------------------------
print("\n===== Обратная задача: восстановление линии укладки на E2 =====")
# import numpy as np

def compute_k_from_decay(omega_percent: float, delta_s: float) -> float:
    """
    Вычисляет коэффициент k по желаемому проценту затухания ошибки.
    
    Аргументы:
        omega_percent : процент, на который должна уменьшиться ошибка за delta_s.
                        Например, omega=50 означает, что ошибка уменьшится вдвое.
        delta_s : длина дуги, на которой достигается заданное затухание (в тех же единицах, что и z).
    
    Возвращает:
        k : коэффициент обратной связи.
    """
    if not 0 < omega_percent < 100:
        raise ValueError("omega_percent должно быть в интервале (0, 100)")
    return -np.log(1 - omega_percent / 100.0) / delta_s

# Пример использования
omega = 50.0   # уменьшение невязки на 50%
delta_s = 2.0  # на отрезке длиной 2 единицы
k = compute_k_from_decay(omega, delta_s)
print(f"Вычисленный k = {k:.4f}")
# Вычислитель правых частей обратной задачи
rhs_calc = RightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=1,
    max_ds_dz=5,
    delta_clip=0.999,
    eps=1e-12
)

# Решатель для обратной задачи
solver_inverse = SciPySolver(method='RK45', rtol=1e-8, atol=1e-10)

# Строитель обратной задачи (старый класс + адаптер)
builder_inv_raw = InvWindingLineBuilder(E2, traj, rhs_calc, solver_inverse)
inverse_builder = InverseWindingLineBuilder(builder_inv_raw)

# Точки вывода – равномерная сетка по длине траектории
z_eval = np.linspace(0, traj.total_length, count_points)

# Начальные координаты на E2 те же (долгота/широта), что и на E1
# (так как E2 соосен и подобен E1, соответствующая точка лежит на экваторе)
z_vals, line_E2 = inverse_builder.build(
    initial_point=(u0, v0),
    eval_points=z_eval
)
diag = inverse_builder.get_diagnostics()

if not diag['success']:
    print("=== Построение не удалось ===")
    print(f"Причина: {diag['message']}")
    print(f"Количество успешно вычисленных точек: {diag['num_points']}")
    print(f"Достигнутое значение параметра: {diag['final_param']:.6f}")
    if 'solver_message' in diag:
        print(f"Сообщение решателя: {diag['solver_message']}")
    
    # Анализируем и даём рекомендации
    if 'шаг стал слишком мал' in diag['message']:
        print("Рекомендация: возможно, система стала жёсткой. Попробуйте:")
        print(" - уменьшить max_step")
        print(" - использовать метод 'Radau' или 'BDF' для жёстких систем")
    elif 'терминальному событию' in diag['message']:
        print("Рекомендация: линия вышла за допустимые границы (u, v).")
        print(" Проверьте диапазон параметризации поверхности.")
    elif 'деление на ноль' in diag['message']:
        print("Рекомендация: возможно, поверхность вырождена вблизи достигнутой точки.")
else:
    print(f"Успех! Построено {diag['num_points']} точек.")
if not inverse_builder.last_run_successful:
    raise RuntimeError("Обратная задача завершилась с ошибкой")

z_vals, deltas = inverse_builder.get_residuals()
print("Максимальная невязка:", np.max(np.abs(deltas)))
import matplotlib.pyplot as plt

# z_vals, deltas = штмукыу_builder.get_residuals()

plt.figure(figsize=(10, 4))
plt.plot(z_vals, deltas, 'b-', linewidth=1.5)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('z (длина дуги)')
plt.ylabel('δ')
plt.title('Невязка δ вдоль траектории')
plt.grid(True)
plt.show()


uv_E2 = inverse_builder.get_uv_states()
print(f"Обратная задача: построено {len(z_vals)} точек на E2")

# ----------------------------------------------------------------------
# 5. Визуализация
# ----------------------------------------------------------------------
print("\n===== Построение 3D-графика =====")

fig = go.Figure()

# 5.1. Поверхность E1 (полупрозрачная)
u_grid = np.linspace(0, 2*np.pi, 80)
v_grid = np.linspace(-np.pi/2, np.pi/2, 50)
U, V = np.meshgrid(u_grid, v_grid)
X1 = np.zeros_like(U)
Y1 = np.zeros_like(U)
Z1 = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = E1.position(U[i,j], V[i,j])
        X1[i,j], Y1[i,j], Z1[i,j] = p

fig.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1,
    opacity=0.2,
    colorscale='Blues',
    showscale=False,
    name='E1 (внешний эллипсоид)'
))

# 5.2. Поверхность E2 (полупрозрачная)
X2 = np.zeros_like(U)
Y2 = np.zeros_like(U)
Z2 = np.zeros_like(U)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        p = E2.position(U[i,j], V[i,j])
        X2[i,j], Y2[i,j], Z2[i,j] = p

fig.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2,
    opacity=0.3,
    colorscale='Reds',
    showscale=False,
    name='E2 (внутренний эллипсоид)'
))

# 5.3. Исходная линия укладки на E1 (она же траектория R(z))
fig.add_trace(go.Scatter3d(
    x=line_E1[:,0], y=line_E1[:,1], z=line_E1[:,2],
    mode='lines',
    line=dict(color='blue', width=5),
    name='Линия укладки на E1 (траектория)'
))

# 5.4. Восстановленная линия укладки на E2
fig.add_trace(go.Scatter3d(
    x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
    mode='lines',
    line=dict(color='red', width=5),
    name='Линия укладки на E2'
))

# 5.5. Соединительные отрезки между точками схода (на E1) и точками на E2
step = 2
for i in range(0, len(z_vals), step):
    Rz = line_E1[i]   # точка траектории (на E1)
    Pz = line_E2[i]   # точка линии укладки на E2
    fig.add_trace(go.Scatter3d(
        x=[Rz[0], Pz[0]],
        y=[Rz[1], Pz[1]],
        z=[Rz[2], Pz[2]],
        mode='lines',
        line=dict(color='green', width=2, dash='solid'),
        showlegend=False
    ))

# 5.6. Стартовые точки
fig.add_trace(go.Scatter3d(
    x=[line_E1[0,0], line_E2[0,0]],
    y=[line_E1[0,1], line_E2[0,1]],
    z=[line_E1[0,2], line_E2[0,2]],
    mode='markers',
    marker=dict(color='black', size=5),
    name='Начальные точки'
))

# Настройка сцены
fig.update_layout(
    title='Прямая + обратная задача: линия укладки на E1 -> траектория -> линия на E2',
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

# fig.show()
fig.write_html('winding_plot.html')
print("Готово.")