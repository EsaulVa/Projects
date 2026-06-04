#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fnc_ellipsoid_hybrid_threshold.py
=================================
Гибридная обратная задача намотки (эллипсоид E1 + оправка E2)
с пороговой визуализацией невязки Phi.

На базе fnc_ellipsoid_hybrid.py.
Точки ТСН, в которых |Phi| > THRESHOLD, окрашиваются в отдельный цвет
для визуальной локализации проблемных участков.
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.io
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method_fixed import newton_corrector
# from helpers.inverse_winding_intermediate import inverse_winding_hybrid
from helpers.inverse_winding_intermediate_adaptive import inverse_winding_hybrid
from helpers.dae_predictor import DAEPredictor
from helpers.optical_predictor import OpticalPredictor, RayTracer
from geometry.tsurfaces import FixedPointTrajectory


# ============================================================================
# 1. ПАРАМЕТРЫ
# ============================================================================

# --- Порог невязки ---
THRESHOLD = 1e-3  # точки ТСН с |Phi| > THRESHOLD выделяются на 3D-сцене

# --- Оправка E2 (баллон) ---
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                 -0.0099656628535, 2.9503573330764]
R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
               39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka,
                                        bound_opravka, cyl_r_opravka)

# --- Эллипсоид E1 (поверхность безопасности) ---
#    Полуоси подобраны так, чтобы оправка строго внутри.
#    a = b = R_cyl + 80, c = H*0.6, центр в середине оправки.
R_max = cyl_r_opravka
H = bound_opravka[-1]
a = b = R_max + 80.0      # 331.7 → округлим до 350 для запаса
c = H * 0.6               # 461.1
z_center = H / 2          # 384.27

# ============================================================================
# 2. АНАЛИТИЧЕСКИЙ ЭЛЛИПСОИД (E1)
# ============================================================================

class EllipsoidSurface:
    """Аналитический эллипсоид для использования как E1."""
    def __init__(self, a, b, c, z_center=0.0):
        self.a = a; self.b = b; self.c = c; self.z_center = z_center

    def position(self, u, v):
        # u ∈ [−π/2, π/2] — "широта" (от полюса к полюсу)
        # v ∈ [0, 2π]     — азимут
        x = self.a * np.cos(u) * np.cos(v)
        y = self.b * np.cos(u) * np.sin(v)
        z = self.c * np.sin(u) + self.z_center
        return np.array([x, y, z])

    def derivatives(self, u, v):
        ru = np.array([-self.a * np.sin(u) * np.cos(v),
                       -self.b * np.sin(u) * np.sin(v),
                        self.c * np.cos(u)])
        rv = np.array([-self.a * np.cos(u) * np.sin(v),
                        self.b * np.cos(u) * np.cos(v),
                        0.0])
        return ru, rv

    def normal(self, u, v):
        ru, rv = self.derivatives(u, v)
        n0 = np.cross(ru, rv)
        return n0 / np.linalg.norm(n0)

    def first_fundamental_form(self, u, v):
        ru, rv = self.derivatives(u, v)
        E = np.dot(ru, ru)
        F = np.dot(ru, rv)
        G = np.dot(rv, rv)
        return E, F, G

    def second_fundamental_form(self, u, v):
        ru, rv = self.derivatives(u, v)
        n = self.normal(u, v)
        # Вторая производная по u
        ruu = np.array([-self.a * np.cos(u) * np.cos(v),
                        -self.b * np.cos(u) * np.sin(v),
                        -self.c * np.sin(u)])
        ruv = np.array([ self.a * np.sin(u) * np.sin(v),
                        -self.b * np.sin(u) * np.cos(v),
                         0.0])
        rvv = np.array([-self.a * np.cos(u) * np.cos(v),
                         -self.b * np.cos(u) * np.sin(v),
                          0.0])
        L = np.dot(ruu, n)
        M = np.dot(ruv, n)
        N = np.dot(rvv, n)
        return L, M, N

    def uv_from_point(self, point):
        x, y, z = point
        z_loc = z - self.z_center
        v = np.arctan2(self.a * y, self.b * x)
        # Клиппинг для численной стабильности arcsin
        sin_u = np.clip(z_loc / self.c, -1.0, 1.0)
        u = np.arcsin(sin_u)
        return u, v

    def radius(self, u):
        # Радиус в поперечном сечении на высоте z = c*sin(u) + z_center
        return self.a * np.cos(u)

    @property
    def u_min(self): return -np.pi/2
    @property
    def u_max(self): return  np.pi/2
    @property
    def v_min(self): return 0.0
    @property
    def v_max(self): return 2*np.pi


class EllipsoidRayTracer:
    """Аналитическая трассировка луча к эллипсоиду."""
    def __init__(self, ellipsoid):
        self.E = ellipsoid

    def intersect(self, origin, direction, t_min=1e-6):
        """Решает квадратное уравнение |origin + t*dir|^2_эллипс = 1."""
        ox, oy, oz = origin
        dx, dy, dz = direction
        a2, b2, c2 = self.E.a**2, self.E.b**2, self.E.c**2
        zc = self.E.z_center

        # Квадратичная форма эллипсоида: (x/a)^2 + (y/b)^2 + ((z-zc)/c)^2 = 1
        A = dx*dx/a2 + dy*dy/b2 + dz*dz/c2
        B = 2*(ox*dx/a2 + oy*dy/b2 + (oz-zc)*dz/c2)
        C = ox*ox/a2 + oy*oy/b2 + (oz-zc)*(oz-zc)/c2 - 1.0

        disc = B*B - 4*A*C
        if disc < 0 or abs(A) < 1e-14:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-B - sqrt_disc) / (2*A)
        t2 = (-B + sqrt_disc) / (2*A)
        # Берем ближайший положительный корень
        ts = [t for t in (t1, t2) if t > t_min]
        if not ts:
            return None
        t = min(ts)
        return origin + t * np.array(direction)


E1 = EllipsoidSurface(a, b, c, z_center)

# Проверка охвата
print("=== Проверка охвата эллипсоида ===")
for z in [0, H/2, H]:
    u_t, _ = E1.uv_from_point(np.array([0, 0, z]))
    r_ell = E1.radius(u_t)
    # Радиус оправки в той же z (приближенно через E2)
    try:
        r_opr = E2.radius(z)
    except Exception:
        r_opr = cyl_r_opravka if bound_opravka[0] < z < bound_opravka[-1] else 0
    print(f" z={z:7.2f}: r_ell={r_ell:7.2f}, r_opr={r_opr:7.2f}, delta={r_ell-r_opr:7.2f}")
print()

# ============================================================================
# 3. ТРАЕКТОРИЯ ТСН
# ============================================================================

# Загружаем ТСН из .mat (построенную прямой задачей на эллипсоиде)
try:
    data = scipy.io.loadmat('winding_trajectory_result.mat')
    X, Y, Z = data['X_tsn'].flatten(), data['Y_tsn'].flatten(), data['Z_tsn'].flatten()
    points_tsn = np.column_stack([X, Y, Z])
    print(f"ТСН загружена из winding_trajectory_result.mat: {len(points_tsn)} точек")
except FileNotFoundError:
    # Fallback: синтетическая винтовая линия на эллипсоиде
    print("winding_trajectory_result.mat не найден — используем синтетическую ТСН")
    t = np.linspace(0, 4*np.pi, 200)
    x = (a*0.95) * np.cos(t) * np.cos(t*3)
    y = (b*0.95) * np.cos(t) * np.sin(t*3)
    z = z_center + (c*0.8) * np.sin(t)
    points_tsn = np.column_stack([x, y, z])

# Для эллипсоида z_offset = 0 (коаксиальные поверхности)
z_offset = 0.0
points_tsn[:, 2] -= z_offset

traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"Z начала: {points_tsn[0,2]:.3f}, Z конца: {points_tsn[-1,2]:.3f}")
print(f"R'(0) = {traj.R_deriv(0.0)}")

if points_tsn[0, 2] > points_tsn[-1, 2]:
    points_tsn = points_tsn[::-1].copy()
    print("Инвертировано: ТСН шла сверху вниз")

# ============================================================================
# 4. НАЧАЛЬНАЯ ТОЧКА (fallback: перебор сетки)
# ============================================================================

R0 = traj.R(0.0)
u_guess, v_guess = safe_initial_point(E2, R0)

# Пробуем корректор от guessed точки
u0, v0, Phi0, _, conv = newton_corrector(
    E2, traj, u_guess, v_guess, 0.0, eps_Phi=1e-12, max_iter=50
)

if not conv or abs(Phi0) > 1e-8:
    print(f"Начальная точка не скорректировалась: u={u0:.4f}, Phi={Phi0:.2e}")
    print("Пробуем fallback: поиск в окрестности...")
    best = None
    for u_t in np.linspace(E2.u_min, E2.u_max, 40):
        for v_t in np.linspace(0, 2*np.pi, 20):
            r_t = E2.position(u_t, v_t)
            m_t = E2.normal(u_t, v_t)
            Phi_t = np.dot(R0 - r_t, m_t)
            if best is None or abs(Phi_t) < abs(best[2]):
                best = (u_t, v_t, Phi_t)
    u0, v0, Phi0 = best[0], best[1], best[2]
    print(f"Fallback: u0={u0:.4f}, v0={v0:.4f}, |Phi0|={abs(Phi0):.2e}")
else:
    print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Phi={Phi0:.2e}")

# ============================================================================
# 5. ПРЕДИКТОРЫ
# ============================================================================

solver_dae = SciPySolver(method='BDF', rtol=1e-8, atol=1e-10)
dae_predictor = DAEPredictor(solver_dae)

ray_tracer = RayTracer()
ray_tracer.register(EllipsoidSurface, EllipsoidRayTracer(E1))
optical_predictor = OpticalPredictor(ray_tracer)

# ============================================================================
# 6. ОБРАТНАЯ ЗАДАЧА
# ============================================================================

print(f"\n===== Обратная задача (эллипсоид) | THRESHOLD = {THRESHOLD:.1e} =====")

result = inverse_winding_hybrid(
    E2, traj, u0, v0,
    count_points=200,
    eps_Phi=1e-10,
    max_newton=50,
    max_bisect=6,
    jump_threshold=3.0,
    predictor_dae=dae_predictor,
    predictor_optical=optical_predictor,
    eps_kappa=1e-2,
    u_margin=20.0,
    force_optical_after_fail=True
)

z_vals = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
line_E2 = result['points_3d']
z_eval=z_vals.copy()

# Точки ТСН для каждого z
points_tsn_eval = np.array([traj.R(z) for z in z_vals])

# Метрики
abs_Phi = np.abs(Phi_hist)
mask_bad = abs_Phi > THRESHOLD
n_bad = np.sum(mask_bad)

print(f"Всего шагов: {len(z_vals)-1}")
print(f"Точек с |Phi| > {THRESHOLD:.1e}: {n_bad} ({100*n_bad/(len(z_vals)-1):.1f}%)")
print(f"Максимальная |Phi| = {np.max(abs_Phi):.2e}")
print(f"Средняя |Phi|      = {np.mean(abs_Phi):.2e}")
print(f"Медианная |Phi|    = {np.median(abs_Phi):.2e}")

# Диагностика: для точек с kappa_n = 0 и Phi > THRESHOLD
for idx in np.where((np.abs(kappa_n_hist) < 1e-1) & (abs_Phi > THRESHOLD))[0]:
    u_t, v_t = u_hist[idx], v_hist[idx]
    if E2.u_min <= u_t <= E2.u_max:
        r_t = E2.position(u_t, v_t)
        m_t = E2.normal(u_t, v_t)
        R_t = traj.R(z_eval[idx])
        delta = R_t - r_t
        lam = np.linalg.norm(delta)
        if lam > 1e-9:
            tau = delta / lam
            cos_theta = np.dot(tau, m_t)
            print(f"z={z_eval[idx]:.1f}: cos(tau,m)={cos_theta:.4f}, "
                  f"Phi={abs_Phi[idx]:.2e}, lam={lam:.2f}")

# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ 3D: пороговая подсветка ТСН
# ============================================================================

fig = go.Figure()

# --- Оправка E2 (сетка) ---
v_grid = np.linspace(0, 2*np.pi, 80)
u_grid_E2 = np.linspace(E2.u_min, E2.u_max, 60)
X2 = np.zeros((len(v_grid), len(u_grid_E2)))
Y2, Z2 = np.zeros_like(X2), np.zeros_like(X2)
for i, v in enumerate(v_grid):
    for j, u in enumerate(u_grid_E2):
        p = E2.position(u, v)
        X2[i,j], Y2[i,j], Z2[i,j] = p

fig.add_trace(go.Surface(
    x=X2, y=Y2, z=Z2,
    opacity=0.35, colorscale='Reds', showscale=False, name='Оправка E2'
))

# --- Эллипсоид E1 (сетка, полупрозрачный) ---
u_grid_E1 = np.linspace(-np.pi/2, np.pi/2, 40)
v_grid_E1 = np.linspace(0, 2*np.pi, 60)
X1 = np.zeros((len(v_grid_E1), len(u_grid_E1)))
Y1, Z1 = np.zeros_like(X1), np.zeros_like(X1)
for i, v in enumerate(v_grid_E1):
    for j, u in enumerate(u_grid_E1):
        p = E1.position(u, v)
        X1[i,j], Y1[i,j], Z1[i,j] = p

fig.add_trace(go.Surface(
    x=X1, y=Y1, z=Z1,
    opacity=0.25, colorscale='Blues', showscale=False, name='Эллипсоид E1'
))

# --- ТСН: вся (синяя линия) ---
fig.add_trace(go.Scatter3d(
    x=points_tsn_eval[:,0], y=points_tsn_eval[:,1], z=points_tsn_eval[:,2],
    mode='lines', line=dict(color='blue', width=3),
    name='ТСН (вся)'
))

# --- ТСН: точки с |Phi| > THRESHOLD (красные маркеры) ---
if n_bad > 0:
    fig.add_trace(go.Scatter3d(
        x=points_tsn_eval[mask_bad, 0],
        y=points_tsn_eval[mask_bad, 1],
        z=points_tsn_eval[mask_bad, 2],
        mode='markers',
        marker=dict(color='orangered', size=6, symbol='diamond',
                    line=dict(color='darkred', width=1)),
        name=f'ТСН |Phi| > {THRESHOLD:.0e} ({n_bad} точек)'
    ))

# --- Восстановленная ЛУ (красная линия) ---
fig.add_trace(go.Scatter3d(
    x=line_E2[:,0], y=line_E2[:,1], z=line_E2[:,2],
    mode='lines', line=dict(color='crimson', width=4),
    name='Восстановленная ЛУ'
))

fig.update_layout(
    title=f'Обратная задача | Порог |Phi| = {THRESHOLD:.0e} (красные маркеры)',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
               aspectmode='data'),
    width=1100, height=800,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
fig.write_html('fnc_ellipsoid_threshold_3d.html')
print("3D-сцена сохранена: fnc_ellipsoid_threshold_3d.html")

# ============================================================================
# 8. ВИЗУАЛИЗАЦИЯ 2D: график |Phi(z)| с линией порога
# ============================================================================

fig2, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(z_vals[1:], abs_Phi[1:] + 1e-16, color='steelblue', lw=1.2, label='|Phi(z)|')
ax.axhline(THRESHOLD, color='orangered', ls='--', lw=2,
           label=f'Порог = {THRESHOLD:.0e}')

# Закрашиваем область выше порога
ax.fill_between(z_vals[1:], THRESHOLD, np.maximum(abs_Phi[1:], THRESHOLD),
                where=(abs_Phi[1:] > THRESHOLD),
                color='orangered', alpha=0.25, label='Зона превышения')

ax.set_xlabel('z (параметр длины ТСН)', fontsize=12)
ax.set_ylabel('|Phi| (невязка связи)', fontsize=12)
ax.set_title(f'Невязка связи |Phi(z)| | Порог = {THRESHOLD:.0e}', fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, which='both', ls=':', alpha=0.5)
plt.tight_layout()
plt.savefig('fnc_ellipsoid_threshold_phi.png', dpi=150)
print("2D-график сохранен: fnc_ellipsoid_threshold_phi.png")
plt.show()

# ============================================================================
# 9. ДОПОЛНИТЕЛЬНЫЕ ДИАГНОСТИЧЕСКИЕ ГРАФИКИ
# ============================================================================

fig3, axes = plt.subplots(2, 2, figsize=(12, 9))

# A. u(z), v(z)
axes[0,0].plot(z_vals, u_hist, label='u(z)')
axes[0,0].plot(z_vals, v_hist, label='v(z)')
axes[0,0].set_title('Параметры поверхности E2')
axes[0,0].legend(); axes[0,0].grid(True)

# B. Итерации Ньютона
axes[0,1].plot(z_vals[1:], result['newton_iters'][1:], '.', ms=3)
axes[0,1].set_title('Итерации Ньютона на шаге')
axes[0,1].grid(True)

# C. Кривизна kappa_n
axes[1,0].plot(z_vals, result['kappa_n'])
axes[1,0].set_title('Нормальная кривизна κ_n')
axes[1,0].grid(True)

# D. Длина нити
axes[1,1].plot(z_vals, result['lam'])
axes[1,1].set_title('Длина свободного участка λ(z)')
axes[1,1].grid(True)

plt.suptitle(f'Диагностика | threshold = {THRESHOLD:.0e}', fontsize=13)
plt.tight_layout()
plt.savefig('fnc_ellipsoid_threshold_diagnostics.png', dpi=150)
print("Диагностика сохранена: fnc_ellipsoid_threshold_diagnostics.png")
plt.show()