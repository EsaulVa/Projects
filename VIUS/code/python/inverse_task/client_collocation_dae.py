# client_collocation_dae.py
"""
Клиент для альтернативных модулей обратной задачи намотки:

    helpers/dae_corrector_predictor.py   — DAE предиктор-корректор (индекс 1)
    helpers/inverse_collocation_dae.py    — прямая коллокация с DAE-инициализацией

Что делает клиент:
  1. Строит оправку E2 (FixedPiecewisePolynomialRevolution) и ТСН-траекторию.
  2. Находит стартовую точку (u0, v0) корректором Ньютона.
  3. Прогон 1: чистый марш DAE предиктором-корректором по сетке узлов.
  4. Прогон 2: прямая коллокация с DAE-инициализацией.
  5. Сравнивает невязку связи Phi на обоих решениях, пишет CSV и HTML.

Запускать из корня проекта (inverse_task/), как остальные client_*.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from geometry.fixed_surfaces_fixed import FixedPiecewisePolynomialRevolution
from core.trajectory import Trajectory
from helpers.inverse_method_fixed import newton_corrector
from helpers.dae_corrector_predictor import DAECorrectorPredictor
from helpers.inverse_collocation_dae import solve_collocation_dae


# ---------------------------------------------------------------------
# 1. Поверхность-оправка E2
# ---------------------------------------------------------------------
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383,
                 -0.0099656628535, 2.9503573330764]
R_c_opravka = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525,
               39582.6812110246392, -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

E2 = FixedPiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka,
                                        bound_opravka, cyl_r_opravka)
print(f"E2: u in [{E2.u_min}, {E2.u_max}], R_cyl={cyl_r_opravka}")


# ---------------------------------------------------------------------
# 2. ТСН-траектория
# ---------------------------------------------------------------------
df = pd.read_csv('tsn_shadow.csv')
df_valid = df[df['valid'] == True].copy()
points_tsn = df_valid[['X', 'Y', 'Z']].values
print(f"ТСН: {len(points_tsn)} валидных точек")

traj = Trajectory.from_points(points_tsn, method='cubic')
print(f"Траектория: длина = {traj.total_length:.2f}")


# ---------------------------------------------------------------------
# 3. Стартовая точка (u0, v0)
# ---------------------------------------------------------------------
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
    E2, dummy, u_guess, v_guess, 0.0, eps_Phi=1e-10, max_iter=20
)
print(f"Старт: u0={u0:.4f}, v0={v0:.4f}, Phi0={Phi0:.2e}, conv={conv}")


# ---------------------------------------------------------------------
# 4. Прогон 1: чистый марш DAE предиктором-корректором
# ---------------------------------------------------------------------
print("\n===== DAE предиктор-корректор (марш) =====")
N_MARCH = 100
z_eval = np.linspace(0.0, traj.total_length, N_MARCH)

predictor = DAECorrectorPredictor(n_substeps=4, method='rk4',
                                  project_every_substep=True,
                                  eps_Phi=1e-10, max_newton_iter=20)

u_dae = np.empty(N_MARCH)
v_dae = np.empty(N_MARCH)
u_dae[0], v_dae[0] = u0, v0
fails = 0
for k in range(N_MARCH - 1):
    out = predictor.predict(z_eval[k], z_eval[k + 1], u_dae[k], v_dae[k], E2, traj)
    if out is None:
        fails += 1
        u_dae[k + 1], v_dae[k + 1] = u_dae[k], v_dae[k]
    else:
        u_dae[k + 1], v_dae[k + 1] = out


def phi_residual(surface, traj, u, v, z):
    out = np.empty(len(z))
    for i in range(len(z)):
        r = surface.position(u[i], v[i])
        m = surface.normal(u[i], v[i])
        out[i] = float(np.dot(traj.R(z[i]) - r, m))
    return out


phi_dae = phi_residual(E2, traj, u_dae, v_dae, z_eval)
print(f"Марш: отказов предиктора = {fails}/{N_MARCH-1}, "
      f"max|Phi|={np.max(np.abs(phi_dae)):.2e}")


# ---------------------------------------------------------------------
# 5. Прогон 2: коллокация с DAE-инициализацией
# ---------------------------------------------------------------------
print("\n===== Коллокация с DAE-инициализацией =====")
result = solve_collocation_dae(
    E2, traj, u0, v0,
    count_points=100,
    w_Phi=10.0, w_diff=1.0,
    init_method='dae', dae_substeps=4, dae_method='rk4',
    max_nfev=10000, tol=1e-10, verbose=True,
)

z_c = result['z']
u_c = result['u']
v_c = result['v']
phi_c = phi_residual(E2, traj, u_c, v_c, z_c)
print(f"Коллокация: max|Phi|={np.max(np.abs(phi_c)):.2e}")


# ---------------------------------------------------------------------
# 6. Сохранение результатов
# ---------------------------------------------------------------------
def to_xyz(surface, u, v):
    P = np.array([surface.position(u[i], v[i]) for i in range(len(u))])
    return P[:, 0], P[:, 1], P[:, 2]


Xd, Yd, Zd = to_xyz(E2, u_dae, v_dae)
Xc, Yc, Zc = to_xyz(E2, u_c, v_c)

pd.DataFrame({
    'z': z_eval, 'u': u_dae, 'v': v_dae,
    'X': Xd, 'Y': Yd, 'Z': Zd, 'Phi': phi_dae,
}).to_csv('inverse_winding_dae_march.csv', index=False)

pd.DataFrame({
    'z': z_c, 'u': u_c, 'v': v_c,
    'X': Xc, 'Y': Yc, 'Z': Zc, 'Phi': phi_c,
}).to_csv('inverse_winding_dae_collocation.csv', index=False)

print("\nСохранено: inverse_winding_dae_march.csv, "
      "inverse_winding_dae_collocation.csv")

# ---- HTML-визуализация (если установлен plotly) ----
try:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=Xd, y=Yd, z=Zd, mode='lines',
                               name='DAE марш', line=dict(width=4)))
    fig.add_trace(go.Scatter3d(x=Xc, y=Yc, z=Zc, mode='lines',
                               name='DAE коллокация', line=dict(width=4)))
    fig.update_layout(title='Обратная задача: DAE предиктор-корректор vs коллокация',
                      scene=dict(aspectmode='data'))
    fig.write_html('inverse_winding_dae.html')
    print("Сохранено: inverse_winding_dae.html")
except Exception as e:
    print(f"plotly недоступен, HTML пропущен: {e}")
