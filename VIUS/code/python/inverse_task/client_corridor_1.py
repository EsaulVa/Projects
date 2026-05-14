import numpy as np
import plotly.graph_objects as go
import scipy.io


from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.intersection import RayTracer, IntersectionAlgorithm,PiecewisePolynomialIntersection

# ======================================================================
# 1. АРХИТЕКТУРА РАСЧЕТА (Без изменений, идеальная абстракция)
# ======================================================================

from dataclasses import dataclass
from typing import Optional
from constraints.corridor_max_calculator import *


# ======================================================================
# 2. ЧИСТАЯ АЛЬТЕРНАТИВА: Численный алгоритм-стратегия
# ======================================================================

# class PiecewisePolynomialIntersection(IntersectionAlgorithm):
#     """
#     Чистая стратегия пересечения для поверхностей вида PiecewisePolynomialRevolution.
#     Не вторгается в класс поверхности. Использует численный поиск корня уравнения
#     R_ray(z) - R_surf(z) = 0 со сканированием интервала для robustness.
#     """
#     def intersect(self, surface, origin, direction, t_min, t_max, sweep_steps=200):
#         ro = np.asarray(origin, dtype=float)
#         rd = np.asarray(direction, dtype=float)
        
#         def get_R_surf(z):
#             # Уважаем границы поверхности
#             if hasattr(surface, 'u_min') and (z < surface.u_min or z > surface.u_max):
#                 return None
#             # Берем радиус на высоте z при угле v=0
#             pt = surface.position(z, 0.0)
#             return np.hypot(pt[0], pt[1])
                
#         def objective(t):
#             pt = ro + t * rd
#             R_surf = get_R_surf(pt[2])
#             if R_surf is None: 
#                 return 1e9 # Штраф за выход за пределы высоты
#             R_ray = np.hypot(pt[0], pt[1])
#             return R_ray - R_surf

#         # Сканирование: ищем интервал, где функция меняет знак
#         dt = (t_max - t_min) / sweep_steps
#         t_prev, f_prev = t_min, objective(t_min)
        
#         for i in range(1, sweep_steps + 1):
#             t_curr = t_min + i * dt
#             f_curr = objective(t_curr)
            
#             if abs(f_curr) < 1e-8: # Прямое попадание
#                 return t_curr, ro + t_curr * rd
                
#             if f_prev * f_curr < 0: # Нашли смену знака - там есть корень!
#                 try:
#                     # Используем брентq только на найденном узком отрезке
#                     t_hit = brentq(objective, t_prev, t_curr, xtol=1e-6)
#                     return t_hit, ro + t_hit * rd
#                 except ValueError:
#                     pass # Если брентq не сошелся (редко), идем дальше
                    
#             t_prev, f_prev = t_curr, f_curr
            
#         return None, None # Пересечение не найдено

# ======================================================================
# 3. ИСХОДНЫЕ ДАННЫЕ
# ======================================================================

phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705
E2_opravka = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366, -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387
E1_safety = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)

z_offset = (bound_safe[3] - bound_opravka[3]) / 2

try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']
    print(f"Эталонная ЛУ загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    raise RuntimeError("Файл LU_data.mat не найден!")

lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')

# ======================================================================
# 4. РАСЧЕТ КОРИДОРА
# ======================================================================

print("\n===== Запуск расчета коридора (Чистая архитектура) =====")

tracer = RayTracer()
# Регистрируем ЧИСТЫЙ численный алгоритм под конкретный тип поверхности
tracer.register(PiecewisePolynomialRevolution, PiecewisePolynomialIntersection())
# МАГИЯ ЗДЕСЬ: Регистрируем ВАШ универсальный алгоритм под ВАШУ поверхность!
# tracer.register(PiecewisePolynomialRevolution, RobustRevolutionIntersection())

calc = CorridorMaxCalculator(
    lu_trajectory=lu_trajectory,
    safety_surface=E1_safety,
    ray_tracer=tracer,
    safe_distance=15.0
)

result = calc.calculate(num_points=200)
valid_lambda = result.lambda_max[result.valid_mask]
print(f"Успешно прострелено: {np.sum(result.valid_mask)} из {len(result.s_array)}")
if len(valid_lambda) > 0:
    print(f"Мин. lambda_max: {np.min(valid_lambda):.2f} мм | Макс. lambda_max: {np.max(valid_lambda):.2f} мм")


# ======================================================================
# 4.1 РАСЧЕТ НИЖНЕЙ ГРАНИЦЫ КОРИДОРА (lambda_min)
# ======================================================================
from constraints.corridor_min_calculator import CorridorMinCalculator

print("\n===== Запуск расчета нижней границы коридора (Кривизна оправки) =====")

calc_min = CorridorMinCalculator(
    lu_trajectory=lu_trajectory,
    mandrel_surface=E2_opravka,  # ВНИМАНИЕ: именно оправка!
    safe_margin=10.0             # Минимальный зазор 10 мм даже на прямой трубе
)

result_min = calc_min.calculate(num_points=200)

print(f"Расчет завершен для {len(result_min.s_array)} точек.")
valid_min = result_min.lambda_min[result_min.valid_mask]
print(f"Минимальный зазор (lambda_min): {np.min(valid_min):.2f} мм")
print(f"Максимальный зазор (lambda_min): {np.max(valid_min):.2f} мм")

# Диагностика: где кривизна сработала?
# Если lambda_min > safe_margin, значит алгоритм обнаружил вогнутость
concave_mask = result_min.lambda_min > (calc_min.safe_margin + 1.0)
if np.any(concave_mask):
    print(f"\nОбнаружено {np.sum(concave_mask)} точек с повышенным риском отрыва нити (вогнутости).")
    # Выведем номера этих точек для сверки с графиком
    indices = np.where(concave_mask)[0]
    print(f"Индексы точек (по массиву s): {indices[0:min(10, len(indices))]}...")
else:
    print("\nВНИМАНИЕ: Вогнутостей не обнаружено. lambda_min равен безопасному зазору везде.")
    print("Возможно, нормальная кривизна в _estimate_normal_curvature имеет отрицательный знак на вогнутостях.")

# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ======================================================================

print("\n===== Построение 3D-графика =====")
fig = go.Figure()

# Оправка (E2) в глобальной системе
u_opr = np.linspace(0, 768.54, 60)
v_opr = np.linspace(0, 2*np.pi, 40)
Uo, Vo = np.meshgrid(u_opr, v_opr)
Zo = Uo.copy()
Xo, Yo = np.zeros_like(Uo), np.zeros_like(Uo)
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = E2_opravka.position(Uo[i,j], Vo[i,j])
        Xo[i,j], Yo[i,j] = p[0], p[1]
fig.add_trace(go.Surface(x=Xo, y=Yo, z=Zo + z_offset, opacity=0.5, colorscale='Blues', name='Оправка'))

# Безопасность (E1)
u_safe = np.linspace(0, 955.956, 80)
v_safe = np.linspace(0, 2*np.pi, 40)
Us, Vs = np.meshgrid(u_safe, v_safe)
Zs = Us.copy()
Xs, Ys = np.zeros_like(Us), np.zeros_like(Us)
for i in range(Us.shape[0]):
    for j in range(Us.shape[1]):
        p = E1_safety.position(Us[i,j], Vs[i,j])
        Xs[i,j], Ys[i,j] = p[0], p[1]
fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds', name='Безопасность'))

# ЛУ на оправке (смещенная)
lu_global = result.lu_points.copy()
lu_global[:, 2] += z_offset 
fig.add_trace(go.Scatter3d(x=lu_global[:, 0], y=lu_global[:, 1], z=lu_global[:, 2],
                           mode='lines+markers', line=dict(color='blue', width=4), marker=dict(size=3), name='ЛУ на оправке'))

# Траектория на стене (Только валидные точки, без выбросов)
valid_indices = np.where(result.valid_mask)[0]
if len(valid_indices) > 0:
    valid_safety_pts = result.safety_points[valid_indices]
    fig.add_trace(go.Scatter3d(x=valid_safety_pts[:, 0], y=valid_safety_pts[:, 1], z=valid_safety_pts[:, 2],
                               mode='lines+markers', line=dict(color='red', width=3), marker=dict(size=3), name='ТСН на безопасности'))

# Лучи (Только валидные)
for i in valid_indices:
    p1 = lu_global[i]
    p2 = result.safety_points[i]
    fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                               mode='lines', line=dict(color='green', width=1.5), showlegend=False))

# ======================================================================
# 5.1 ВИЗУАЛИЗАЦИЯ lambda_min (Желтые/Красные лучи внутрь)
# ======================================================================

for i in range(len(result_min.s_array)):
    if result_min.valid_mask[i]:
        p1 = lu_global[i] # Точка на оправке
        
        # Направление "внутрь" к оси
        inward_dir = np.array([-p1[0], -p1[1], 0.0])
        norm_len = np.linalg.norm(inward_dir)
        if norm_len > 1e-6:
            inward_dir /= norm_len
            
            p2 = p1 + inward_dir * result_min.lambda_min[i]
            
            # Если длина больше безопасного зазора + 1 мм, значит это зона отрыва (вогнутость)
            is_concave = result_min.lambda_min[i] > (calc_min.safe_margin + 1.0)
            
            if is_concave:
                color = 'red'      # Зона риска отрыва
                width = 3.0        # Толстая линия
            else:
                color = 'yellow'   # Безопасный минимальный зазор
                width = 2.0
            
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], 
                y=[p1[1], p2[1]], 
                z=[p1[2], p2[2]],
                mode='lines', 
                line=dict(color=color, width=width),
                showlegend=False
            ))

# Легенда
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                           marker=dict(color='green', size=6), name='Лучи lambda_max (к стене)'))
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                           marker=dict(color='yellow', size=6), name='Зона lambda_min (безопасная)'))
fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                           marker=dict(color='red', size=6), name='Зона lambda_min (риск отрыва)'))
fig.update_layout(title='Чистая архитектура: Численное пересечение (Scipy + Брент)',
                  scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z (глобальная)', aspectmode='data'),
                  width=1200, height=900, margin=dict(l=0, r=0, b=0, t=40))
fig.write_html('corridor_clean_architecture.html')
print("График сохранен в corridor_clean_architecture.html")