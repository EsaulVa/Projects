import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === ПУТЬ К ПРОЕКТУ ===
PROJECT_ROOT = r'C:\Vius_projects\Projects\VIUS\code\python\inverse_task'
sys.path.insert(0, PROJECT_ROOT)

# === ИМПОРТЫ ===
from helpers.inverse_collocation_scaled_snapshot import solve_collocation_scaled

# ============================================================
# ВЫБЕРИТЕ ВАРИАНТ: тестовый или реальный
# ============================================================

# --- ВАРИАНТ А: Тестовый эллипсоид (работает сразу) ---
class TestBalloon:
    def __init__(self, a=300, b=400):
        self.a = a
        self.b = b
        self.u_min = -b
        self.u_max = b

    def radius(self, u):
        return self.a * np.sqrt(max(1.0 - (u / self.b) ** 2, 0.0))

    def position(self, u, v):
        r = self.radius(u)
        return np.array([r * np.cos(v), r * np.sin(v), u])

    def normal(self, u, v):
        r = self.radius(u)
        if r < 1e-6:
            return np.array([0, 0, 1])
        dr = -self.a ** 2 * u / (self.b ** 2 * r) if r > 0 else 0
        n = np.array([np.cos(v), np.sin(v), -dr])
        return n / np.linalg.norm(n)

    def first_fundamental_form(self, u, v):
        r = self.radius(u)
        dr = -self.a ** 2 * u / (self.b ** 2 * r) if r > 1e-6 else 0
        return 1.0 + dr ** 2, 0.0, r ** 2


class TestTraj:
    def __init__(self, R0, total_length):
        self.R0 = R0
        self.total_length = total_length

    def R(self, z):
        t = z / self.total_length * 4 * np.pi
        r = self.R0 + 50 * np.sin(t)
        return np.array([r * np.cos(t), r * np.sin(t), z])


E2 = TestBalloon(a=300, b=400)
traj = TestTraj(R0=350, total_length=2848.79)
u0, v0 = 0.0, 0.01

# --- ВАРИАНТ Б: Ваш реальный баллон (раскомментируйте) ---
# from geometry.piecewise_polynomial_revolution_fixed import FixedPiecewisePolynomialRevolution
# from core.trajectory import Trajectory
# E2 = FixedPiecewisePolynomialRevolution(...)
# traj = Trajectory.from_points(...)
# u0, v0 = 0.0, 0.0101

# ============================================================
# ЗАПУСК
# ============================================================

print(f"Surface: u∈[{E2.u_min:.2f}, {E2.u_max:.2f}]")
print(f"Trajectory: length={traj.total_length:.2f}")
print(f"Start: u={u0}, v={v0}")

result = solve_collocation_scaled(
    E2, traj, u0, v0,
    count_points=200,
    w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
    init_method='radial',
    max_nfev=50000,
    tol=1e-8,
    verbose=True,
    snapshot_file='balloon_best.npz',
    snapshot_interval=1,
    restore_from_snapshot=True
)

# ============================================================
# ВИЗУАЛИЗАЦИЯ + ВОССТАНОВЛЕНИЕ ИЗ СНАПШОТА
# ============================================================

if result is not None:
    # Демасштабирование
    u_min, u_max = E2.u_min, E2.u_max
    scale_u = u_max - u_min
    scale_v = 2.0 * np.pi

    u_opt = result.x[0::2] * scale_u + u_min
    v_opt = result.x[1::2] * scale_v
    z_eval = np.linspace(0, traj.total_length, len(u_opt))

    r_pts = np.array([E2.position(u_opt[k], v_opt[k]) for k in range(len(u_opt))])
    R_pts = np.array([traj.R(z_eval[k]) for k in range(len(u_opt))])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_pts[:, 0], r_pts[:, 1], r_pts[:, 2], 'g-', lw=2, label='ЛУ (коллокация)')
    ax.plot(R_pts[:, 0], R_pts[:, 1], R_pts[:, 2], 'b--', lw=1, label='ТСН')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Обратная задача: коллокация со снапшотом')
    plt.show()

    # Проверка: можно ли загрузить снапшот отдельно?
    print("\n--- Проверка снапшота ---")
    from helpers.inverse_collocation_scaled_snapshot import load_collocation_snapshot
    X_snap, meta = load_collocation_snapshot('balloon_best.npz')
    if X_snap is not None:
        print(f"Снапшот загружен: max|Phi|={meta.get('best_Phi', '?'):.3e}")