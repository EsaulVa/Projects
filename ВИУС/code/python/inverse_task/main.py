"""
Контрольный пример восстановления линии укладки на эллипсоидальной оправке.

Генерируется геодезическая траектория на внешнем эллипсоиде E1 (поверхность безопасности).
На внутреннем соосном эллипсоиде E2 (оправка) вычисляется соответствующая линия укладки.
Результат визуализируется в интерактивном 3D-графике (Plotly).
"""

import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# Импорт компонентов нашего фреймворка
from geometry.ellipsoid import EllipsoidWithDerivatives
from core.trajectory import Trajectory
# from trajectory import register_builders  # регистрирует кубический сплайн
from inverse_winding.rhs_calculator import RightHandSideCalculator
from solvers.scipy_solver import SciPySolver
from inverse_winding.inverse_winding_builder import *

# ----------------------------------------------------------------------
# 1. Генерация геодезической линии на внешнем эллипсоиде E1
# ----------------------------------------------------------------------
def generate_geodesic_trajectory(
    surface: EllipsoidWithDerivatives,
    u0: float, v0: float,
    du0: float, dv0: float,
    s_max: float,
    num_points: int = 500
) -> np.ndarray:
    """
    Численное построение геодезической линии на поверхности.

    Решается система ОДУ геодезических:
        d²u/ds² + Γ¹_ij (du^i/ds)(du^j/ds) = 0
        d²v/ds² + Γ²_ij (du^i/ds)(du^j/ds) = 0

    Используется решатель SciPy (solve_ivp) с теми же обёртками,
    что и в основном расчёте.

    Возвращает массив точек (N, 3) вдоль геодезической.
    """
    # Символы Кристоффеля для поверхности
    def christoffel(u, v):
        # Получаем метрику и её производные
        E, F, G = surface.first_fundamental_form(u, v)
        E_u, E_v, F_u, F_v, G_u, G_v = surface.metric_derivatives(u, v)
        det = E*G - F*F
        inv_det = 1.0 / det
        g11 = G * inv_det
        g12 = -F * inv_det
        g22 = E * inv_det

        # Γ¹_uu, Γ¹_uv, Γ¹_vv
        Gamma1_uu = 0.5 * (g11*E_u + g12*(2*F_u - E_v))
        Gamma1_uv = 0.5 * (g11*E_v + g12*G_u)
        Gamma1_vv = 0.5 * (g11*(2*F_v - G_u) + g12*G_v)

        # Γ²_uu, Γ²_uv, Γ²_vv
        Gamma2_uu = 0.5 * (g12*E_u + g22*(2*F_u - E_v))
        Gamma2_uv = 0.5 * (g12*E_v + g22*G_u)
        Gamma2_vv = 0.5 * (g12*(2*F_v - G_u) + g22*G_v)

        return (Gamma1_uu, Gamma1_uv, Gamma1_vv,
                Gamma2_uu, Gamma2_uv, Gamma2_vv)

    def geodesic_ode(s, state):
        u, v, u_prime, v_prime = state
        G1_uu, G1_uv, G1_vv, G2_uu, G2_uv, G2_vv = christoffel(u, v)

        u_2prime = -(G1_uu * u_prime**2 + 2*G1_uv * u_prime*v_prime + G1_vv * v_prime**2)
        v_2prime = -(G2_uu * u_prime**2 + 2*G2_uv * u_prime*v_prime + G2_vv * v_prime**2)

        return [u_prime, v_prime, u_2prime, v_2prime]

    # Начальные условия
    y0 = [u0, v0, du0, dv0]
    # Нормируем начальную скорость, чтобы она была единичной
    geom = surface.derivatives(u0, v0)
    ru, rv = geom['ru'], geom['rv']
    vel = ru * du0 + rv * dv0
    speed = np.linalg.norm(vel)
    y0[2] /= speed
    y0[3] /= speed

    # Интегрирование
    s_eval = np.linspace(0, s_max, num_points)
    sol = solve_ivp(geodesic_ode, (0, s_max), y0, t_eval=s_eval,
                    method='DOP853', rtol=1e-8, atol=1e-10)

    # Преобразование в 3D точки
    points = np.array([surface.position(u, v) for u, v in sol.y[:2].T])
    return points

# ----------------------------------------------------------------------
# 2. Параметры эллипсоидов и траектории
# ----------------------------------------------------------------------
# Внешний эллипсоид E1 (поверхность безопасности)
a1, b1, c1 = 3.0, 2.5, 2.0
E1 = EllipsoidWithDerivatives(a1, b1, c1)

# Внутренний эллипсоид E2 (оправка) – соосный и подобный
scale = 0.6
a2, b2, c2 = a1*scale, b1*scale, c1*scale
E2 = EllipsoidWithDerivatives(a2, b2, c2)

# Параметры геодезической на E1
u0_geo, v0_geo = 0.0, 0.0               # старт на экваторе (x = a1)
du0_geo, dv0_geo = 0.0, 1.0             # направление вдоль меридиана?
# Для наглядности сделаем наклонное направление:
# du0_geo, dv0_geo = 0.6, 0.8

s_max_geo = 20.0  # длина геодезической
num_geo_points = 200

print("Генерация геодезической траектории на E1...")
geo_points = generate_geodesic_trajectory(E1, u0_geo, v0_geo,
                                          du0_geo, dv0_geo,
                                          s_max_geo, num_geo_points)
print(f"Сгенерировано {len(geo_points)} точек.")

# ----------------------------------------------------------------------
# 3. Создание траектории R(z) из точек геодезической
# ----------------------------------------------------------------------
print("Построение сплайна траектории...")
traj = Trajectory.from_points(geo_points, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# ----------------------------------------------------------------------
# 4. Настройка вычислителя правых частей и решателя
# ----------------------------------------------------------------------
rhs_calc = RightHandSideCalculator(
    surface=E2,
    trajectory=traj,
    k=1.5,
    max_ds_dz=0.2,
    delta_clip=0.999,
    eps=1e-12
)

# Решатель ОДУ (SciPy DOP853 для высокой точности)
solver = SciPySolver(method='DOP853', rtol=1e-8, atol=1e-10)

# ----------------------------------------------------------------------
# 5. Строитель линии укладки и запуск расчёта
# ----------------------------------------------------------------------
builder = InvWindingLineBuilder(E2, traj, rhs_calc, solver)
# 2. Оборачиваем его в адаптер
builder = InverseWindingLineBuilder(builder)

# Начальная точка на оправке (подбирается вручную)
# Для простоты выберем точку на E2, соответствующую той же долготе/широте,
# что и первая точка геодезической (но это не обязательно точно)
u0 = u0_geo
v0 = v0_geo

# Точки вывода – равномерная сетка по длине траектории
z_eval = np.linspace(0, traj.total_length, 400)

print("Расчёт линии укладки на E2...")
z_vals, line_3d = builder.build(initial_point=(u0, v0),
    eval_points=z_eval)
uv_states = builder.get_uv_states()
print(f"Рассчитано {len(z_vals)} точек линии укладки.")

# ----------------------------------------------------------------------
# 6. Визуализация с Plotly
# ----------------------------------------------------------------------
print("Построение интерактивного графика...")

# Создание фигуры
fig = go.Figure()

# 6.1. Поверхность E1 (полупрозрачная)
u_grid = np.linspace(0, 2*np.pi, 60)
v_grid = np.linspace(-np.pi/2, np.pi/2, 40)
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
    name='E1 (поверхность безопасности)'
))

# 6.2. Поверхность E2 (полупрозрачная, другого цвета)
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
    name='E2 (оправка)'
))

# 6.3. Траектория R(z) на E1
fig.add_trace(go.Scatter3d(
    x=geo_points[:,0], y=geo_points[:,1], z=geo_points[:,2],
    mode='lines',
    line=dict(color='red', width=4),
    name='R(z) на E1'
))

# 6.4. Линия укладки на E2
fig.add_trace(go.Scatter3d(
    x=line_3d[:,0], y=line_3d[:,1], z=line_3d[:,2],
    mode='lines',
    line=dict(color='blue', width=4),
    name='Линия укладки на E2'
))

# 6.5. Соединительные отрезки между R(z) и линией укладки
# Берём каждую 20-ю точку, чтобы не загромождать
step = 20
for i in range(0, len(z_vals), step):
    z = z_vals[i]
    Rz = traj.R(z)
    Pz = line_3d[i]
    fig.add_trace(go.Scatter3d(
        x=[Rz[0], Pz[0]],
        y=[Rz[1], Pz[1]],
        z=[Rz[2], Pz[2]],
        mode='lines',
        line=dict(color='green', width=2, dash='dot'),
        showlegend=False
    ))

# Настройка макета
fig.update_layout(
    title='Восстановление линии укладки на эллипсоидальной оправке',
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

# Показываем график
fig.show()

print("Готово.")