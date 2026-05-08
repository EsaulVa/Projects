from matplotlib import pyplot as plt
import numpy as np
from geometry.cylinder import CylinderAnalytical  # или тот класс, которым вы пользуетесь
from helpers.inverse_method import *
from core.trajectory import Trajectory

# Генерируем винтовую траекторию на внутреннем цилиндре R=3
def helical(R, height_start, height_end, turns, n):
    total_h = height_end - height_start
    pitch = total_h / turns
    theta = np.linspace(0, 2*np.pi*turns, n)
    z = height_start + pitch * theta / (2*np.pi)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    return np.column_stack([x, y, z])

R_int = 3.0
R_ext = 2.0
height_start = 1.0
height_end = 12.0
turns = 3
traj_pts = helical(R_int, 1.0, 12.0, 3, 300)
traj_points=traj_pts
traj = Trajectory.from_points(traj_pts, method='cubic')

opravka = CylinderAnalytical(radius=R_ext)

# Начальное приближение
R0 = traj.R(0.0)
u_guess = np.arctan2(R0[1], R0[0])+1e-2
v_guess = R0[2]
u0, v0, Phi0, conv = opravka.project_point(R0, u_guess, v_guess)
print(f"Начальная точка: u={u0:.4f}, v={v0:.4f}, Phi={Phi0:.2e}, сходимость={conv}")

# Обратная задача
result = inverse_winding_v3(opravka, traj, u0, v0, count_points=200,
                         eps_Phi=1e-10, max_newton=7)
z_eval = result['z_eval']
u_hist = result['u']
v_hist = result['v']
Phi_hist = result['Phi']
kappa_n_hist = result['kappa_n']
newton_iters_hist = result['newton_iters']
lam_hist = result['lam']
flags = result['flags']
line_E2 = result['points_3d']

# ----------------------------------------------------------------------
# 7. ВИЗУАЛИЗАЦИЯ
# ----------------------------------------------------------------------
R1=R_int
R2=R_ext
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
# поверхность внешнего цилиндра (полупрозрачная)
u_grid = np.linspace(0, 2*np.pi, 60)
v_grid = np.linspace(height_start, height_end, 60)
U, V = np.meshgrid(u_grid, v_grid)
X1 = R1 * np.cos(U)
Y1 = R1 * np.sin(U)
Z1 = V
ax.plot_surface(X1, Y1, Z1, alpha=0.1, color='lightblue', edgecolor='gray', linewidth=0.2)

# поверхность внутреннего цилиндра
X2 = R2 * np.cos(U)
Y2 = R2 * np.sin(U)
Z2 = V
ax.plot_surface(X2, Y2, Z2, alpha=0.2, color='salmon', edgecolor='gray', linewidth=0.2)

# траектория точки схода
ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
        'b-', linewidth=2, label='Траектория R(z)')

# восстановленная линия укладки
ax.plot(line_E2[:, 0], line_E2[:, 1], line_E2[:, 2],
        'r-', linewidth=2, label='Линия укладки')

# несколько нитей
for i in range(0, len(z_eval), len(z_eval)//10):
    R_pt = traj.R(z_eval[i])
    r_pt = line_E2[i]
    ax.plot([R_pt[0], r_pt[0]], [R_pt[1], r_pt[1]], [R_pt[2], r_pt[2]],
            'g-', linewidth=0.8, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Обратная задача намотки: цилиндр → цилиндр')
ax.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 8. ДИАГНОСТИЧЕСКИЕ ГРАФИКИ
# ----------------------------------------------------------------------
fig_diag, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].semilogy(z_eval[1:], np.abs(Phi_hist[1:])+1e-16, 'b-', lw=1.2)
axes[0,0].set_title('Невязка |Φ|')
axes[0,1].bar(z_eval[1:], newton_iters_hist[1:], width=0.8*(z_eval[1]-z_eval[0]),
              color='steelblue', alpha=0.7)
axes[0,1].set_title('Итерации Ньютона')
axes[0,2].bar(z_eval[1:], flags[1:], width=0.8*(z_eval[1]-z_eval[0]),
              color='orange', alpha=0.7)
axes[0,2].set_title('Бисекция')
axes[1,0].plot(z_eval, kappa_n_hist, 'g-', lw=1.2)
axes[1,0].set_title('Нормальная кривизна κ_n')
axes[1,1].plot(z_eval, lam_hist, 'm-', lw=1.2)
axes[1,1].set_title('Длина нити λ')
axes[1,2].plot(z_eval, u_hist, 'r-', label='u(z)')
axes[1,2].plot(z_eval, v_hist, 'b-', label='v(z)')
axes[1,2].set_title('Координаты u(z), v(z)')
axes[1,2].legend()
for ax in axes.flat:
    ax.grid(True)
fig_diag.suptitle('Диагностика обратной задачи (цилиндр)', fontsize=14)
plt.tight_layout()
plt.show()

print("Готово.")