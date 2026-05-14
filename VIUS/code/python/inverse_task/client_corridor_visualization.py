import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import plotly.graph_objects as go
import scipy.io
from scipy.optimize import brentq

from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.intersection import RayTracer, IntersectionAlgorithm

# ======================================================================
# 1. НАШИ НОВЫЕ КЛАССЫ ДЛЯ РАСЧЕТА КОРИДОРА
# ======================================================================

from dataclasses import dataclass
from typing import Optional

@dataclass
class CorridorMaxResult:
    """Контейнер для результатов расчета верхней границы коридора."""
    s_array: np.ndarray               
    lu_points: np.ndarray             
    safety_points: np.ndarray         
    lambda_max: np.ndarray            
    safety_trajectory: Optional[Trajectory] 
    valid_mask: np.ndarray            

class CorridorMaxCalculator:
    def __init__(self, lu_trajectory: Trajectory, safety_surface, ray_tracer: RayTracer, safe_distance: float = 10.0):
        self.traj = lu_trajectory
        self.surface = safety_surface
        self.tracer = ray_tracer
        self.t_min = safe_distance

    def calculate(self, num_points: int = 100, t_max: float = 1500.0) -> CorridorMaxResult:
        s_array = np.linspace(0, self.traj.total_length, num_points)
        lu_points = np.zeros((num_points, 3))
        safety_points = np.zeros((num_points, 3))
        lambda_max = np.zeros(num_points)
        valid_mask = np.zeros(num_points, dtype=bool)
        
        for i, s in enumerate(s_array):
            r = self.traj.R(s)
            tau = self.traj.R_deriv(s)
            lu_points[i] = r
            
            try:
                t, pt = self.tracer.trace(self.surface, r, tau, self.t_min, t_max)
                if t is not None:
                    safety_points[i] = pt
                    lambda_max[i] = t
                    valid_mask[i] = True
                else:
                    safety_points[i] = r + t_max * tau
                    lambda_max[i] = np.inf
            except Exception:
                safety_points[i] = r + t_max * tau
                lambda_max[i] = np.inf

        valid_pts = safety_points[valid_mask]
        safety_trajectory = Trajectory.from_points(valid_pts, method='cubic') if len(valid_pts) > 4 else None

        return CorridorMaxResult(
            s_array=s_array, lu_points=lu_points, safety_points=safety_points,
            lambda_max=lambda_max, safety_trajectory=safety_trajectory, valid_mask=valid_mask
        )

# ======================================================================
# 2. СПЕЦИАЛИЗИРОВАННЫЙ АЛГОРИТМ ПЕРЕСЕЧЕНИЯ ДЛЯ ВАШЕЙ ПОВЕРХНОСТИ
# ======================================================================

class PiecewiseSafeIntersection(IntersectionAlgorithm):
    """
    Аналог fzero из MATLAB. Ищет пересечение луча с поверхностью вращения,
    заданной кусочным полиномом, путем поиска корня уравнения R_ray(z) - R_surf(z) = 0.
    """
    def intersect(self, surface, origin, direction, t_min, t_max):
        ro, rd = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        
        def get_R_surf(z):
            if hasattr(surface, 'u_min') and (z < surface.u_min or z > surface.u_max): return None
            pt = surface.position(z, 0.0)
            return np.hypot(pt[0], pt[1])
            
        def objective(t):
            pt = ro + t * rd
            R_surf = get_R_surf(pt[2])
            if R_surf is None: return 1e9 
            return np.hypot(pt[0], pt[1]) - R_surf
            
        f_min = objective(t_min)
        f_max = objective(t_max)
        
        if f_min * f_max > 0: return None, None
            
        try:
            t_hit = brentq(objective, t_min, t_max, xtol=1e-6)
            return t_hit, ro + t_hit * rd
        except ValueError:
            return None, None

# ======================================================================
# 3. ИСХОДНЫЕ ДАННЫЕ И ИНИЦИАЛИЗАЦИЯ (из вашего кода)
# ======================================================================

# Поверхности
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

# Траектория ЛУ (Линии Укладки на оправке)
try:
    data_l = scipy.io.loadmat('.\\LU_data.mat')
    r_etalon = data_l['r']  # Это точки НА ОПРАВКЕ в ЛОКАЛЬНОЙ системе координат
    print(f"Эталонная ЛУ загружена: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    raise RuntimeError("Файл LU_data.mat не найден!")

# Создаем объект Trajectory из этих точек (он сам посчитает касательные)
lu_trajectory = Trajectory.from_points(r_etalon, method='cubic')

# ======================================================================
# 4. РАСЧЕТ КОРИДОРА (ЛУЧИ)
# ======================================================================

print("\n===== Запуск расчета коридора (трассировка лучей) =====")

# Настраиваем трассировщик
tracer = RayTracer()
tracer.register(PiecewisePolynomialRevolution, PiecewiseSafeIntersection())

# Создаем калькулятор
calc = CorridorMaxCalculator(
    lu_trajectory=lu_trajectory,
    safety_surface=E1_safety,
    ray_tracer=tracer,
    safe_distance=15.0 # Не ищем пересечения ближе 15 мм от оправки
)

# Запускаем расчет (80 точек оптимально для визуализации лучей)
result = calc.calculate(num_points=200)

valid_lambda = result.lambda_max[result.valid_mask]
print(f"Успешно прострелено лучей: {np.sum(result.valid_mask)} из {len(result.s_array)}")
print(f"Минимальная длина нити (lambda_max): {np.min(valid_lambda):.2f} мм")
print(f"Максимальная длина нити (lambda_max): {np.max(valid_lambda):.2f} мм")

# ======================================================================
# 5. ВИЗУАЛИЗАЦИЯ (PLOTLY)
# ======================================================================

print("\n===== Построение 3D-графика =====")

fig = go.Figure()

# --- 1. Рисуем поверхности ---
# Оправка (E2) - рисуем в глобальной системе (прибавляем z_offset)
u_opr = np.linspace(0, 768.54, 60)
v_opr = np.linspace(0, 2*np.pi, 40)
Uo, Vo = np.meshgrid(u_opr, v_opr)
Zo = Uo.copy()
Xo, Yo = np.zeros_like(Uo), np.zeros_like(Uo)
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = E2_opravka.position(Uo[i,j], Vo[i,j])
        Xo[i,j], Yo[i,j] = p[0], p[1]

fig.add_trace(go.Surface(x=Xo, y=Yo, z=Zo + z_offset, opacity=0.5, colorscale='Blues', name='Оправка (E2)'))

# Безопасность (E1) - уже в глобальной системе
u_safe = np.linspace(0, 955.956, 80)
v_safe = np.linspace(0, 2*np.pi, 40)
Us, Vs = np.meshgrid(u_safe, v_safe)
Zs = Us.copy()
Xs, Ys = np.zeros_like(Us), np.zeros_like(Us)
for i in range(Us.shape[0]):
    for j in range(Us.shape[1]):
        p = E1_safety.position(Us[i,j], Vs[i,j])
        Xs[i,j], Ys[i,j] = p[0], p[1]

fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds', name='Безопасность (E1)'))

# --- 2. Рисуем Линию Укладки (на оправке) ---
# Помним, что r_etalon в локальных координатах, для графика прибавляем z_offset к Z
lu_global = result.lu_points.copy()
lu_global[:, 2] += z_offset 

fig.add_trace(go.Scatter3d(
    x=lu_global[:, 0], y=lu_global[:, 1], z=lu_global[:, 2],
    mode='lines+markers', 
    line=dict(color='blue', width=4),
    marker=dict(size=3, color='blue'),
    name='ЛУ на оправке'
))

# --- 2. Рисуем Рассчитанную Траекторию на Стене ---
# Берем ТОЛЬКО валидные точки (там, где луч реально попал в стену)
valid_indices = np.where(result.valid_mask)[0]

if len(valid_indices) > 0:
    valid_safety_pts = result.safety_points[valid_indices]
    
    # Рисуем только успешную часть траектории
    fig.add_trace(go.Scatter3d(
        x=valid_safety_pts[:, 0], 
        y=valid_safety_pts[:, 1], 
        z=valid_safety_pts[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=3, color='red'),
        name='ТСН на безопасности (Только удачные лучи)'
    ))
else:
    print("ВНИМАНИЕ: Ни один луч не попал в поверхность безопасности! Проверьте t_max или геометрию.")

# --- 3. Рисуем ЛУЧИ (касательные) ---
# Цикл ниже остался таким же, но я добавил в него проверку на всякий случай
for i in range(len(result.s_array)):
    if result.valid_mask[i]:
        p1 = lu_global[i] # Точка на оправке (смещенная)
        p2 = result.safety_points[i] # Точка на стене
        
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], 
            y=[p1[1], p2[1]], 
            z=[p1[2], p2[2]],
            mode='lines', 
            line=dict(color='green', width=1.5),
            showlegend=False
        ))
    else:
        # Если хотите визуально видеть, куда улетели "плохие" лучи, 
        # можно раскомментировать код ниже (они будут серыми и пунктирными):
        pass
        # p1 = lu_global[i]
        # p2_bad = p1 + result.lambda_max[i] * lu_trajectory.R_deriv(result.s_array[i])
        # p2_bad[2] += z_offset
        # fig.add_trace(go.Scatter3d(
        #     x=[p1[0], p2_bad[0]], y=[p1[1], p2_bad[1]], z=[p1[2], p2_bad[2]],
        #     mode='lines', line=dict(color='gray', width=1, dash='dash'), showlegend=False
        # ))

# --- 5. Оформление графика ---
fig.update_layout(
    title='Расчет lambda_max (Трассировка лучей от ЛУ к поверхности безопасности)',
    scene=dict(
        xaxis_title='X, мм', 
        yaxis_title='Y, мм', 
        zaxis_title='Z (глобальная), мм', 
        aspectmode='data'
    ),
    width=1200, 
    height=900,
    margin=dict(l=0, r=0, b=0, t=40)
)

# Сохраняем в HTML
output_file = 'corridor_max_visualization.html'
fig.write_html(output_file)
print(f"\nГрафик успешно сохранен в файл: {output_file}")