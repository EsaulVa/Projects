import numpy as np
from geometry.composite_surface import CompositeSurface, CylinderSegment, SphereSegment
from core.const_dev_law import ConstantDeviation
from forward_winding.forward_winding_builder import ForwardWindingBuilder
from core.trajectory import Trajectory
from solvers.scipy_solver import SciPySolver
from helpers.inverse_method import inverse_winding_v3, newton_corrector,inverse_winding_v4

# 1. Создаём внутренний баллон E_int (оправка) и внешний E_ext (безопасность)
R_int, L_int = 2, 6
z_min_int, z_max_int = -L_int/2, L_int/2
cyl_int = CylinderSegment(R_int, z_min_int, z_max_int)
lower_sph_int = SphereSegment(R_int, z_min_int, is_upper=False)
upper_sph_int = SphereSegment(R_int, z_max_int, is_upper=True)
E_int = CompositeSurface([lower_sph_int, cyl_int, upper_sph_int])

R_ext, L_ext = 4, 12
z_min_ext, z_max_ext = -L_ext/2, L_ext/2
cyl_ext = CylinderSegment(R_ext, z_min_ext, z_max_ext)
lower_sph_ext = SphereSegment(R_ext, z_min_ext, is_upper=False)
upper_sph_ext = SphereSegment(R_ext, z_max_ext, is_upper=True)
E_ext = CompositeSurface([lower_sph_ext, cyl_ext, upper_sph_ext])

# 2. Прямая задача на внутреннем баллоне — траектория R(z)
dev_law = ConstantDeviation(tan_theta=0.1)
solver_fwd = SciPySolver(method='BDF', rtol=1e-8, atol=1e-10)
fwd_builder = ForwardWindingBuilder(
    surface=E_int, deviation_law=dev_law,
    solver=solver_fwd, normalize_tangent=True, eps=1e-12
)

u0_int = 0.0
v0_int = E_int.v_min + 0.2   # нижнее днище внутреннего баллона
alpha = np.pi / 6
s_end = 25.0
s_eval = np.linspace(0, s_end, 1200)

print("Прямая задача на внутреннем баллоне...")
s_vals, line_int = fwd_builder.build(
    initial_point=(u0_int, v0_int),
    initial_tangent=(alpha,),
    eval_points=s_eval
)
if not fwd_builder.last_run_successful:
    raise RuntimeError("Прямая задача не завершена")

traj = Trajectory.from_points(line_int, method='cubic', bc_type='natural')
print(f"Длина траектории: {traj.total_length:.3f}")

# 3. Начальная точка на внешнем баллоне для обратной задачи
#    Ищем проекцию первой точки траектории на внешнюю поверхность
R0 = traj.R(0.0)
# Простейшее начальное приближение: те же угловые координаты (u0_int, v0_int)
u0_ext_guess = u0_int
v0_ext_guess = v0_int
# Используем метод project_point, если он добавлен в CompositeSurface, иначе newton_corrector
if hasattr(E_ext, 'project_point'):
    u0_ext, v0_ext, Phi0, conv = E_ext.project_point(
        R0, u0_ext_guess, v0_ext_guess, eps_Phi=1e-12, max_iter=20
    )
else:
    # Фиктивная траектория для корректора
    from helpers.inverse_method import FixedPointTrajectory
    dummy_traj = FixedPointTrajectory(R0)
    u0_ext, v0_ext, Phi0, nit, conv = newton_corrector(
        E_ext, dummy_traj, u0_ext_guess, v0_ext_guess, 0.0,
        eps_Phi=1e-12, max_iter=20
    )
print(f"Начальная точка на внешнем баллоне: u={u0_ext:.4f}, v={v0_ext:.4f}, Φ0={Phi0:.2e}")
E2=E_ext
u0_guess=u0_ext
v0_guess=v0_ext
# Коррекция начальной точки методом Ньютона
r0 = E2.position(u0_guess, v0_guess)
R0 = traj.R(0.0)
m0 = E2.normal(u0_guess, v0_guess)
Phi0 = np.dot(R0 - r0, m0)
if abs(Phi0) > 1e-8:
    print("Корректировка начальной точки...")
    u0, v0, Phi0_corr, _, conv0 = newton_corrector(
        E2, traj, u0_guess, v0_guess, 0.0, eps_Phi=1e-12, max_iter=20
    )
    print(f"После коррекции: Φ₀ = {Phi0_corr:.6e}, сошёлся: {conv0}")
else:
    u0, v0 = u0_guess, v0_guess
# 4. Обратная задача на внешнем баллоне
print("Обратная задача на внешнем баллоне...")
result = inverse_winding_v4(
    E_ext, traj, u0_ext, v0_ext,
    count_points=1300,
    eps_Phi=1e-10, max_newton=7, max_bisect=4, jump_threshold=3.0
)

line_ext = result['points_3d']
Phi_hist = result['Phi']
z_vals = result['z_eval']          # ← вот она, недостающая строка
line_E2 = result['points_3d']
Phi_hist = result['Phi']
print(f"Максимальная невязка |Φ| = {np.max(np.abs(Phi_hist)):.2e}")

# Визуализация может быть добавлена аналогично предыдущим примерам.