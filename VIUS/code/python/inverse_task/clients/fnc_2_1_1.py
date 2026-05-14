import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

# ======================================================================
# 1. КЛАСС ЦИЛИНДРА (аналитический, без JAX)
# ======================================================================
class Cylinder:
    """
    Прямой круговой цилиндр радиуса R, ось Z.
    Параметризация: u ∈ [0, 2π) – угол, v ∈ ℝ – высота.
    r(u, v) = (R cos u, R sin u, v)
    """
    def __init__(self, radius=1.0):
        self.R = radius

    def position(self, u, v):
        return np.array([self.R * np.cos(u), self.R * np.sin(u), v])

    def derivatives(self, u, v):
        ru = np.array([-self.R * np.sin(u), self.R * np.cos(u), 0.0])
        rv = np.array([0.0, 0.0, 1.0])
        return {'ru': ru, 'rv': rv}

    def normal(self, u, v):
        return np.array([np.cos(u), np.sin(u), 0.0])

    def first_fundamental_form(self, u, v):
        return self.R**2, 0.0, 1.0

    def second_fundamental_form(self, u, v):
        # L, M, N
        return -self.R, 0.0, 0.0


# ======================================================================
# 2. КЛАСС ТРАЕКТОРИИ (интерполяция 3D-кривой по длине дуги)
# ======================================================================
class Trajectory3D:
    def __init__(self, points):
        self.points = np.array(points, dtype=float)
        N = len(self.points)
        diffs = np.diff(self.points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.s = np.zeros(N)
        self.s[1:] = np.cumsum(seg_lengths)
        self.total_length = self.s[-1]
        self._interp_x = CubicSpline(self.s, self.points[:, 0])
        self._interp_y = CubicSpline(self.s, self.points[:, 1])
        self._interp_z = CubicSpline(self.s, self.points[:, 2])

    def R(self, z):
        z = np.clip(z, 0, self.total_length)
        return np.array([self._interp_x(z), self._interp_y(z), self._interp_z(z)])

    def R_deriv(self, z):
        z = np.clip(z, 0, self.total_length)
        return np.array([self._interp_x(z, 1), self._interp_y(z, 1), self._interp_z(z, 1)])


# ======================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАТНОЙ ЗАДАЧИ
# ======================================================================
def compute_tangent_components(surface, u, v, tau_3d):
    geom = surface.derivatives(u, v)
    ru, rv = geom['ru'], geom['rv']
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    b1 = np.dot(tau_3d, ru)
    b2 = np.dot(tau_3d, rv)
    u_prime = (G * b1 - F * b2) / det
    v_prime = (-F * b1 + E * b2) / det
    return u_prime, v_prime

def normal_curvature(surface, u, v, u_prime, v_prime):
    L, M, N_ff = surface.second_fundamental_form(u, v)
    E, F, G = surface.first_fundamental_form(u, v)
    II_val = L * u_prime**2 + 2.0 * M * u_prime * v_prime + N_ff * v_prime**2
    I_val = E * u_prime**2 + 2.0 * F * u_prime * v_prime + G * v_prime**2
    if abs(I_val) < 1e-15:
        return 0.0
    return II_val / I_val

def compute_grad_Phi(surface, u, v, u_prime, v_prime, lam):
    E, F, G = surface.first_fundamental_form(u, v)
    L, M, N_ff = surface.second_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика в grad_Phi")
    guu = G / det
    guv = -F / det
    gvv = E / det
    tau_u = E * u_prime + F * v_prime
    tau_v = F * u_prime + G * v_prime
    b_u_u = guu * L + guv * M
    b_u_v = guv * L + gvv * M
    b_v_u = guu * M + guv * N_ff
    b_v_v = guv * M + gvv * N_ff
    dPhidu = -lam * (b_u_u * tau_u + b_u_v * tau_v)
    dPhidv = -lam * (b_v_u * tau_u + b_v_v * tau_v)
    return dPhidu, dPhidv

def inverse_metric(surface, u, v):
    E, F, G = surface.first_fundamental_form(u, v)
    det = E * G - F * F
    if abs(det) < 1e-14:
        raise ValueError("Вырожденная метрика")
    return G / det, -F / det, E / det

def recompute_thread_geometry(surface, traj, u, v, z):
    r = surface.position(u, v)
    R = traj.R(z)
    delta = R - r
    lam = np.linalg.norm(delta)
    if lam < 1e-13:
        raise ValueError("Нулевая длина нити")
    tau_3d = delta / lam
    u_p, v_p = compute_tangent_components(surface, u, v, tau_3d)
    return tau_3d, lam, u_p, v_p

def project_to_tangent_plane(surface, u, v, vec_3d):
    m = surface.normal(u, v)
    return vec_3d - np.dot(vec_3d, m) * m

def compute_dr_dz(surface, traj, u, v, z):
    r = surface.position(u, v)
    R = traj.R(z)
    m = surface.normal(u, v)
    delta = R - r
    lam = np.linalg.norm(delta)
    if lam < 1e-13:
        return 0.0, 0.0
    tau = delta / lam
    R_prime = traj.R_deriv(z)
    dPhi_dz = np.dot(R_prime, m)
    up, vp = compute_tangent_components(surface, u, v, tau)
    dPhi_du, dPhi_dv = compute_grad_Phi(surface, u, v, up, vp, lam)
    R_prime_par = project_to_tangent_plane(surface, u, v, R_prime)
    Rp_u, Rp_v = compute_tangent_components(surface, u, v, R_prime_par)
    residual = dPhi_dz + dPhi_du * Rp_u + dPhi_dv * Rp_v
    guu, guv, gvv = inverse_metric(surface, u, v)
    grad_u = guu * dPhi_du + guv * dPhi_dv
    grad_v = guv * dPhi_du + gvv * dPhi_dv
    norm_grad_sq = (guu * dPhi_du**2 + 2 * guv * dPhi_du * dPhi_dv + gvv * dPhi_dv**2)
    if norm_grad_sq < 1e-14:
        return Rp_u, Rp_v
    mu = -residual / norm_grad_sq
    du_dz = Rp_u + mu * grad_u
    dv_dz = Rp_v + mu * grad_v
    return du_dz, dv_dz

def newton_corrector(surface, traj, u_pred, v_pred, z_target,
                     eps_Phi=1e-10, max_iter=7):
    u_c, v_c = u_pred, v_pred
    for nit in range(max_iter):
        r_c = surface.position(u_c, v_c)
        m_c = surface.normal(u_c, v_c)
        R_t = traj.R(z_target)
        delta_c = R_t - r_c
        lam_c = np.linalg.norm(delta_c)
        Phi_c = np.dot(delta_c, m_c)
        if abs(Phi_c) < eps_Phi:
            return u_c, v_c, Phi_c, nit, True
        if lam_c < 1e-13:
            return u_c, v_c, Phi_c, nit, False
        tau_c = delta_c / lam_c
        try:
            up_c, vp_c = compute_tangent_components(surface, u_c, v_c, tau_c)
            dPdu, dPdv = compute_grad_Phi(surface, u_c, v_c, up_c, vp_c, lam_c)
            guu, guv, gvv = inverse_metric(surface, u_c, v_c)
        except ValueError:
            return u_c, v_c, Phi_c, nit, False
        Ng = (guu * dPdu**2 + 2 * guv * dPdu * dPdv + gvv * dPdv**2)
        if Ng < 1e-14:
            return u_c, v_c, Phi_c, nit, False
        u_c -= Phi_c / Ng * (guu * dPdu + guv * dPdv)
        v_c -= Phi_c / Ng * (guv * dPdu + gvv * dPdv)
    r_c = surface.position(u_c, v_c)
    m_c = surface.normal(u_c, v_c)
    Phi_c = np.dot(traj.R(z_target) - r_c, m_c)
    return u_c, v_c, Phi_c, max_iter, abs(Phi_c) < eps_Phi

def inverse_winding_v3(surface, traj, u0, v0, count_points=300,
                       eps_Phi=1e-10, max_newton=7,
                       max_bisect=4, jump_threshold=3.0):
    z_eval = np.linspace(0, traj.total_length, count_points)
    N = len(z_eval)
    u_hist = np.zeros(N)
    v_hist = np.zeros(N)
    Phi_hist = np.zeros(N)
    kappa_n_hist = np.zeros(N)
    newton_iters_hist = np.zeros(N, dtype=int)
    lam_hist = np.zeros(N)
    flags = np.zeros(N, dtype=int)

    u_hist[0], v_hist[0] = u0, v0
    u_cur, v_cur = u0, v0

    for i in range(N - 1):
        z_k = z_eval[i]
        z_next = z_eval[i + 1]
        try:
            tau_k, lam_k, up_k, vp_k = recompute_thread_geometry(
                surface, traj, u_cur, v_cur, z_k)
        except ValueError as e:
            print(f"Шаг {i}: {e}")
            u_hist[i+1:] = u_cur
            v_hist[i+1:] = v_cur
            break
        lam_hist[i] = lam_k
        kappa_n_hist[i] = normal_curvature(surface, u_cur, v_cur, up_k, vp_k)

        try:
            du_dz_k, dv_dz_k = compute_dr_dz(surface, traj, u_cur, v_cur, z_k)
        except ValueError:
            du_dz_k, dv_dz_k = 0.0, 0.0

        E_f, F_f, G_f = surface.first_fundamental_form(u_cur, v_cur)
        speed_expected = np.sqrt(max(E_f * du_dz_k**2
                                     + 2 * F_f * du_dz_k * dv_dz_k
                                     + G_f * dv_dz_k**2, 0.0))

        n_sub = 1
        best_u, best_v = u_cur, v_cur
        best_Phi = 1.0
        best_nit = 0
        bisected = False

        for bisect_level in range(max_bisect + 1):
            sub_z = np.linspace(z_k, z_next, n_sub + 1)
            u_s, v_s = u_cur, v_cur
            total_nit = 0
            jump_detected = False

            for j in range(n_sub):
                z_a = sub_z[j]
                z_b = sub_z[j + 1]
                dz_sub = z_b - z_a
                try:
                    du_s, dv_s = compute_dr_dz(surface, traj, u_s, v_s, z_a)
                except ValueError:
                    du_s, dv_s = du_dz_k, dv_dz_k
                u_p = u_s + du_s * dz_sub
                v_p = v_s + dv_s * dz_sub
                u_c, v_c, Phi_c, nit, conv = newton_corrector(
                    surface, traj, u_p, v_p, z_b,
                    eps_Phi=eps_Phi, max_iter=max_newton)
                total_nit += nit
                du_j = u_c - u_s
                dv_j = v_c - v_s
                Ej, Fj, Gj = surface.first_fundamental_form(u_s, v_s)
                ds_actual = np.sqrt(max(Ej * du_j**2
                                        + 2*Fj * du_j * dv_j
                                        + Gj * dv_j**2, 0.0))
                ds_expect = speed_expected * dz_sub
                if ds_expect > 1e-12:
                    ratio = ds_actual / ds_expect
                else:
                    ratio = 1.0
                if (ratio > jump_threshold or ratio < 1.0/jump_threshold) \
                        and bisect_level < max_bisect:
                    jump_detected = True
                    break
                u_s, v_s = u_c, v_c

            if not jump_detected:
                best_u, best_v = u_s, v_s
                r_f = surface.position(best_u, best_v)
                m_f = surface.normal(best_u, best_v)
                best_Phi = np.dot(traj.R(z_next) - r_f, m_f)
                best_nit = total_nit
                if bisect_level > 0:
                    bisected = True
                break
            else:
                n_sub *= 2

        u_cur, v_cur = best_u, best_v
        u_hist[i + 1] = u_cur
        v_hist[i + 1] = v_cur
        Phi_hist[i + 1] = best_Phi
        newton_iters_hist[i + 1] = best_nit
        flags[i + 1] = 1 if bisected else 0

    try:
        _, lam_last, up_last, vp_last = recompute_thread_geometry(
            surface, traj, u_hist[-1], v_hist[-1], z_eval[-1])
        lam_hist[-1] = lam_last
        kappa_n_hist[-1] = normal_curvature(
            surface, u_hist[-1], v_hist[-1], up_last, vp_last)
    except ValueError:
        pass

    points_3d = np.array([surface.position(u_hist[k], v_hist[k]) for k in range(N)])
    return {
        'z_eval': z_eval,
        'u': u_hist, 'v': v_hist,
        'Phi': Phi_hist, 'kappa_n': kappa_n_hist,
        'newton_iters': newton_iters_hist,
        'lam': lam_hist, 'flags': flags,
        'points_3d': points_3d
    }


# ======================================================================
# 4. ГЕНЕРАЦИЯ ТРАЕКТОРИИ НА ВНЕШНЕМ ЦИЛИНДРЕ (винтовая линия)
# ======================================================================
def helical_trajectory(R_ext, height_start, height_end, turns, num_points):
    """
    Винтовая линия на цилиндре радиуса R_ext с осью Z.
    Начинается с u=0, v=height_start, поднимается до height_end,
    делая turns полных оборотов.
    Параметризована равномерно по длине дуги.
    """
    total_height = height_end - height_start
    # шаг винта
    pitch = total_height / turns if turns != 0 else 0
    # угловая скорость
    total_angle = 2 * np.pi * turns
    # длина винтовой линии: ∫ sqrt(R² + (dz/dθ)²) dθ
    # dz/dθ = pitch / (2π)
    dz_dtheta = pitch / (2 * np.pi) if turns != 0 else total_height / total_angle if total_angle != 0 else 0
    # Элемент дуги ds = sqrt(R² + (dz/dθ)²) dθ
    ds_dtheta = np.sqrt(R_ext**2 + dz_dtheta**2)
    total_length = ds_dtheta * abs(total_angle) if turns != 0 else total_height

    # массив длин дуг s от 0 до total_length
    s_vals = np.linspace(0, total_length, num_points)
    theta = s_vals / ds_dtheta
    z_vals = height_start + dz_dtheta * theta
    x = R_ext * np.cos(theta)
    y = R_ext * np.sin(theta)
    return np.column_stack([x, y, z_vals])


# ======================================================================
# 5. ПАРАМЕТРЫ ЗАДАЧИ
# ======================================================================
R1 = 3.0  # радиус внешнего цилиндра (траектория раскладчика)
R2 = 2.0  # радиус внутреннего цилиндра (оправка)

height_start = 1.0
height_end = 12.0
turns = 3

# Генерируем траекторию точки схода (винтовая линия на цилиндре R1)
traj_points = helical_trajectory(R1, height_start, height_end, turns, num_points=300)
traj = Trajectory3D(traj_points)

print(f"Длина траектории: {traj.total_length:.3f}")

# Начальная точка на оправке E2: проекция R(0) на цилиндр R2.
R0 = traj.R(0.0)
# Простейшая проекция: та же угловая координата u=0, а высоту v ставим как у R0.
# Для цилиндра R2 это точка (R2, 0, R0[2]), т.е. u=0, v=R0[2].
u0_est = np.pi/10
v0_est = R0[2]

# Создаем поверхность E2
E2 = Cylinder(radius=R2)

# Проверка начальной невязки
r0 = E2.position(u0_est, v0_est)
m0 = E2.normal(u0_est, v0_est)
Phi0 = np.dot(R0 - r0, m0)
print(f"Начальная невязка Φ₀ = {Phi0:.6e}")

if abs(Phi0) > 1e-8:
    print("Корректировка начальной точки...")
    u0_est, v0_est, Phi0_corr, _, conv = newton_corrector(
        E2, traj, u0_est, v0_est, 0.0, eps_Phi=1e-12, max_iter=20)
    print(f"  После коррекции: Φ = {Phi0_corr:.6e}, сошёлся: {conv}")

# ======================================================================
# 6. ЗАПУСК ОБРАТНОЙ ЗАДАЧИ
# ======================================================================
count_points = 300
result = inverse_winding_v3(
    E2, traj, u0_est, v0_est,
    count_points=count_points,
    eps_Phi=1e-10, max_newton=7, max_bisect=4, jump_threshold=3.0
)

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