# clients/client_collocation_balloon.py
"""
Клиент прямой коллокации для баллона (PiecewisePolynomialRevolution).
Использует исправленную геометрию (fixed_v2) и ТСН из tsn_shadow.csv.
"""
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy.io
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution, safe_initial_point
from core.trajectory import Trajectory
from helpers.inverse_collocation import solve_collocation
from helpers.inverse_method_fixed import newton_corrector, compute_dr_dz


# ---------- 1. Поверхность E2 (оправка) ----------
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                 -0.0099656628535, 2.9503573330764]
R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
               39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka,
                                          bound_opravka, cyl_r_opravka)
print(f"E2: u∈[{E2.u_min}, {E2.u_max}], R_cyl={cyl_r_opravka}")


# ---------- 2. Загрузка ТСН ----------
df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True].copy()
points_tsn = df_valid[['X', 'Y', 'Z']].values
print(f"ТСН: {len(points_tsn)} валидных точек")

# Траектория ТСН
traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"Траектория: длина = {traj.total_length:.2f}")


# ---------- 3. Начальная точка ----------
R0 = traj.R(0.0)

class DummyTraj:
    def __init__(self, R_fix):
        self._R = np.asarray(R_fix, dtype=float)
        self.total_length = 1.0
    def R(self, z):
        return self._R
    def R_deriv(self, z):
        return np.zeros(3)

dummy = DummyTraj(R0)
u_guess = float(R0[2])
v_guess = float(np.arctan2(R0[1], R0[0]))

u0, v0, Phi0, _, conv = newton_corrector(
    E2, dummy, u_guess, v_guess, 0.0,
    eps_Phi=1e-10, max_iter=20
)
print(f"Старт: u={u0:.3f}, v={v0:.4f}, Φ={Phi0:.2e}, conv={conv}")

# Проверка compute_dr_dz
du, dv = compute_dr_dz(E2, traj, u0, v0, 0.0)
print(f"compute_dr_dz: du={du:.6f}, dv={dv:.6f}")


# ---------- 4. Прямой метод коллокации ----------
print("\n===== Коллокация на баллоне =====")

# Начнём с N=50 для скорости, потом можно N=100
result = solve_collocation(
    E2, traj, u0, v0,
    count_points=50,
    w_Phi=1.0, w_diff=1.0, w_smooth=0.0,
    init_method='dae',
    max_nfev=10000, tol=1e-8,
    verbose=True
)

print(f"\n>>> |F|={result['res_norm']:.3e}, success={result['success']}")
print(f">>> Макс |Φ| = {np.max(np.abs(result['Phi'])):.2e}")
print(f">>> Сред |Φ| = {np.mean(np.abs(result['Phi'])):.2e}")


# ---------- 5. Загрузка эталона для сравнения ----------
try:
    data_l = scipy.io.loadmat('LU_data.mat')
    r_etalon = data_l['r']
    print(f"Эталон: {r_etalon.shape[0]} точек")
except FileNotFoundError:
    r_etalon = None
    print("Эталон не найден")


# ---------- 6. Визуализация ----------
line_E2 = result['points_3d']
tsn_pts = np.array([traj.R(z) for z in result['z_eval']])

fig = go.Figure()

# Поверхность E2
u_grid = np.linspace(E2.u_min, E2.u_max, 50)
v_grid = np.linspace(0, 2 * np.pi, 40)
Um, Vm = np.meshgrid(u_grid, v_grid)
X2 = np.zeros_like(Um)
Y2 = np.zeros_like(Um)
Z2 = np.zeros_like(Um)
for i in range(Um.shape[0]):
    for j in range(Um.shape[1]):
        p = E2.position(Um[i, j], Vm[i, j])
        X2[i, j], Y2[i, j], Z2[i, j] = p
fig.add_trace(go.Surface(x=X2, y=Y2, z=Z2, opacity=0.4, colorscale='Reds',
                         showscale=False, name='Оправка E2'))

# ТСН
fig.add_trace(go.Scatter3d(x=tsn_pts[:, 0], y=tsn_pts[:, 1], z=tsn_pts[:, 2],
                             mode='lines', line=dict(color='blue', width=4),
                             name='ТСН'))

# Восстановленная ЛУ (коллокация)
fig.add_trace(go.Scatter3d(x=line_E2[:, 0], y=line_E2[:, 1], z=line_E2[:, 2],
                             mode='lines+markers', line=dict(color='green', width=3),
                             marker=dict(size=3), name='ЛУ (коллокация)'))

# Эталон
if r_etalon is not None:
    fig.add_trace(go.Scatter3d(x=r_etalon[:, 0], y=r_etalon[:, 1], z=r_etalon[:, 2],
                                 mode='lines', line=dict(color='orange', width=2, dash='dot'),
                                 name='Эталон ЛУ'))

fig.update_layout(
    title='Обратная задача: Коллокация на баллоне',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
    width=1200, height=900
)
fig.write_html('collocation_balloon.html')
print("\nГрафик: collocation_balloon.html")

# Диагностика
z_eval = result['z_eval']
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].semilogy(z_eval, np.abs(result['Phi']) + 1e-16)
axes[0, 0].set_title('Невязка |Φ|')
axes[0, 1].plot(z_eval, result['u'], label='u(z)')
axes[0, 1].plot(z_eval, result['v'], label='v(z)')
axes[0, 1].set_title('Координаты')
axes[0, 1].legend()
axes[1, 0].plot(z_eval[1:], np.diff(result['u']), label='Δu')
axes[1, 0].plot(z_eval[1:], np.diff(result['v']), label='Δv')
axes[1, 0].set_title('Приращения')
axes[1, 0].legend()
plt.tight_layout()
plt.savefig('diagnostics_balloon.png', dpi=150)
plt.show()
print("Диагностика: diagnostics_balloon.png")