import sys
import numpy as np
import torch

# === ПУТЬ К ПРОЕКТУ ===
PROJECT_ROOT = r'C:\Vius_projects\Projects\VIUS\code\python\inverse_task'
sys.path.insert(0, PROJECT_ROOT)

from helpers.pytorch_winding_net import WindingNet, JLoss, train, generate_batch

# ============================================================
# ВЫБЕРИТЕ ВАРИАНТ: тестовый или ваш реальный баллон
# ============================================================

# --- ВАРИАНТ А: Тестовый эллипсоид (проверка без вашей геометрии) ---
class TestSurface:
    def __init__(self, a=300, b=400):
        self.a = a
        self.b = b
        self.u_min = -b
        self.u_max = b

    def position(self, u, v):
        r = self.a * np.sqrt(max(1.0 - (u / self.b) ** 2, 0.0))
        return np.array([r * np.cos(v), r * np.sin(v), u])

    def normal(self, u, v):
        r = self.a * np.sqrt(max(1.0 - (u / self.b) ** 2, 0.0))
        if r < 1e-6:
            return np.array([0, 0, 1])
        dr = -self.a ** 2 * u / (self.b ** 2 * r)
        n = np.array([np.cos(v), np.sin(v), -dr])
        return n / np.linalg.norm(n)

    def first_fundamental_form(self, u, v):
        r = self.a * np.sqrt(max(1.0 - (u / self.b) ** 2, 0.0))
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


surface = TestSurface(a=300, b=400)
traj = TestTraj(R0=350, total_length=2848.79)

# --- ВАРИАНТ Б: Ваш реальный баллон (раскомментируйте) ---
# from geometry.piecewise_polynomial_revolution_fixed import FixedPiecewisePolynomialRevolution
# from core.trajectory import Trajectory
# surface = FixedPiecewisePolynomialRevolution(...)  # ваш конструктор
# traj = Trajectory.from_points(...)                 # ваша траектория

# ============================================================
# ПАРАМЕТРЫ СЕТКИ
# ============================================================

N = 100  # сетка для обучения (можно 50, 100, 200; GPU потянет и 300)
z_eval = np.linspace(0, traj.total_length, N)

# Базовое начальное приближение (etalon или radial init)
# Если есть эталон:
# base_u = r_etalon[:, ...]  # ваши данные
# base_v = np.arctan2(r_etalon[:, 1], r_etalon[:, 0])

# Если нет эталона — линейная заглушка по высоте:
base_u = np.linspace(surface.u_min, surface.u_max, N)
base_v = np.linspace(0, 2 * np.pi, N)

# ============================================================
# СОЗДАНИЕ СЕТИ И ОБУЧЕНИЕ
# ============================================================

net = WindingNet(hidden=64)
jloss = JLoss(surface, traj, z_eval,
              w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
              spsa_delta=1e-3, spsa_samples=2)

print(f"Обучение на сетке N={N}, batch=16")
print(f"Surface: u∈[{surface.u_min:.1f}, {surface.u_max:.1f}]")

net = train(net, jloss, base_u, base_v,
            epochs=200, batch_size=16, lr=1e-3,
            noise_u=20.0, noise_v=0.2,
            verbose=True)

# ============================================================
# ПРИМЕНЕНИЕ: один прогон от случайного init
# ============================================================

X_test = generate_batch(base_u, base_v, batch_size=1, noise_u=30, noise_v=0.3)
X_test = X_test.permute(0, 2, 1)  # (1, 2, N)

with torch.no_grad():
    Y_test = net(X_test)

Y_np = Y_test.permute(0, 2, 1).cpu().numpy()[0]  # (N, 2)
u_opt = Y_np[:, 0]
v_opt = Y_np[:, 1]

# Проверка Phi
phi_vals = []
for k in range(N):
    r = surface.position(u_opt[k], v_opt[k])
    m = surface.normal(u_opt[k], v_opt[k])
    phi = np.dot(traj.R(z_eval[k]) - r, m)
    phi_vals.append(abs(phi))

print(f"\nПосле сети: max|Phi|={max(phi_vals):.3e}, mean|Phi|={np.mean(phi_vals):.3e}")

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r_pts = np.array([surface.position(u_opt[k], v_opt[k]) for k in range(N)])
R_pts = np.array([traj.R(z_eval[k]) for k in range(N)])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(r_pts[:, 0], r_pts[:, 1], r_pts[:, 2], 'g-', lw=2, label='ЛУ (нейросеть)')
ax.plot(R_pts[:, 0], R_pts[:, 1], R_pts[:, 2], 'b--', lw=1, label='ТСН')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title(f'PyTorch WindingNet | N={N} | max|Phi|={max(phi_vals):.2e}')
plt.show()

# Сохранение
torch.save(net.state_dict(), 'winding_net.pt')
print("\nСеть сохранена в winding_net.pt")