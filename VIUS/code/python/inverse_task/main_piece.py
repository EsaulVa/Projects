import numpy as np
import plotly.graph_objects as go
from geometry.piecewise_polynomial_revolution import PiecewisePolynomialRevolution

# ---------- Коэффициенты из surface_r.m (оправка) ----------
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764]
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, 39582.6812110246392,
                 -43518.6731429065403, 19122.1758646943599]
bound_opravka = [0, 234.27, 534.27, 768.54]
cyl_r_opravka = 251.705

opravka = PiecewisePolynomialRevolution(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka)

# ---------- Коэффициенты из surface_r_b.m (безопасность) ----------
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379]
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, 27152.4105364360366,
              -31195.5446114188999, 14397.6607910855146]
bound_safe = [0, 327.978, 627.978, 955.956]
cyl_r_safe = 352.387

safe_surf = PiecewisePolynomialRevolution(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe)

# Смещение центров (как в align_surface_centers.m)
z_offset = (bound_safe[3] - bound_opravka[3]) / 2  # (955.956 - 768.54)/2 ≈ 93.708
print(f"z_offset = {z_offset:.3f} мм")

# Построение сеток
u_opr = np.linspace(0, 768.54, 80)
v_opr = np.linspace(0, 2*np.pi, 60)
Uo, Vo = np.meshgrid(u_opr, v_opr)
Xo, Yo, Zo = np.zeros_like(Uo), np.zeros_like(Uo), np.zeros_like(Uo)
for i in range(Uo.shape[0]):
    for j in range(Uo.shape[1]):
        p = opravka.position(Uo[i,j], Vo[i,j])
        Xo[i,j] = p[0]
        Yo[i,j] = p[1]
        Zo[i,j] = p[2] + z_offset   # переход в глобальную систему

u_safe = np.linspace(0, 955.956, 100)
v_safe = np.linspace(0, 2*np.pi, 60)
Us, Vs = np.meshgrid(u_safe, v_safe)
Xs, Ys, Zs = np.zeros_like(Us), np.zeros_like(Us), np.zeros_like(Us)
for i in range(Us.shape[0]):
    for j in range(Us.shape[1]):
        p = safe_surf.position(Us[i,j], Vs[i,j])
        Xs[i,j] = p[0]
        Ys[i,j] = p[1]
        Zs[i,j] = p[2]

fig = go.Figure()
fig.add_trace(go.Surface(x=Xo, y=Yo, z=Zo, opacity=0.4, colorscale='Blues', name='Оправка'))
fig.add_trace(go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.2, colorscale='Reds', name='Безопасность'))

# Попытка загрузить ТСН, если есть файл
try:
    import scipy.io
    data = scipy.io.loadmat('winding_trajectory_result.mat')
    X_tsn = data['X_tsn'].flatten()
    Y_tsn = data['Y_tsn'].flatten()
    Z_tsn = data['Z_tsn'].flatten()
    fig.add_trace(go.Scatter3d(x=X_tsn, y=Y_tsn, z=Z_tsn,
                               mode='lines', line=dict(color='green', width=4),
                               name='ТСН (из .mat)'))
except FileNotFoundError:
    print("Файл winding_trajectory_result.mat не найден – ТСН не отображена.")

fig.update_layout(title='Поверхности баллона и безопасности',
                  scene=dict(xaxis_title='X, мм', yaxis_title='Y, мм', zaxis_title='Z, мм',
                             aspectmode='data'),
                  width=1000, height=800)
fig.show()