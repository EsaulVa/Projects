# temporal_deployment_from_kinematics.py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from core.curve_factory import CurveFactory
from core.register_factory import *   # регистрирует 'cubic' и 'nurbs'
from machine.temporal_deployer import TemporalDeployer


# =============================================================================
# Загрузка исходных данных
# =============================================================================
data = scipy.io.loadmat('kinematics_results.mat')
s_array = data['s'].flatten()
theta = data['theta'].flatten()
Z = data['Z'].flatten()
R = data['R'].flatten()
phi = data['phi'].flatten()   # радианы
z_offset = float(data['z_offset'].flatten()[0])

# =============================================================================
# Параметры станка и ограничения (ПОДСТАВЬТЕ РЕАЛЬНЫЕ ЗНАЧЕНИЯ)
# =============================================================================
limits = {
    'theta': {'max_speed': 10.0, 'max_accel': 50.0},   # рад/с, рад/с²
    'Z': {'max_speed': 200.0, 'max_accel': 500.0},     # мм/с, мм/с²
    'R': {'max_speed': 150.0, 'max_accel': 400.0},
    'phi': {'max_speed': 20.0, 'max_accel': 100.0}
}

axes_data = {
    'theta': theta,
    'Z': Z,
    'R': R,
    'phi': phi
}

# =============================================================================
# Временная развертка с использованием NURBS
# =============================================================================
factory = CurveFactory()
# Для NURBS можно задать степень (degree) – 3 или 5
deployer = TemporalDeployer(factory, method='nurbs', degree=5)
result = deployer.deploy(s_array, axes_data, limits,
                         mode='const_speed', speed_param=150.0,
                         n_iter=5, relax=0.3)

t = result['t_array']
total_time = result['total_time']
curves_t = result['curves_t']

# =============================================================================
# Вычисление скоростей и ускорений на равномерной сетке по времени
# =============================================================================
t_uniform = np.linspace(0, total_time, 1000)

q_names = ['theta', 'Z', 'R', 'phi']
q_vals = {name: np.zeros(len(t_uniform)) for name in q_names}
dq_vals = {name: np.zeros(len(t_uniform)) for name in q_names}
ddq_vals = {name: np.zeros(len(t_uniform)) for name in q_names}

for i, ti in enumerate(t_uniform):
    for name in q_names:
        curve = curves_t[name]
        q_vals[name][i] = curve.evaluate(ti, der=0)
        dq_vals[name][i] = curve.evaluate(ti, der=1)
        ddq_vals[name][i] = curve.evaluate(ti, der=2)

# =============================================================================
# Построение графиков
# =============================================================================
fig, axes = plt.subplots(4, 3, figsize=(12, 14))
fig.suptitle('Временная развертка и кинематика приводов (NURBS)', fontsize=14)

for idx, name in enumerate(q_names):
    # Позиция
    ax = axes[idx, 0]
    ax.plot(t_uniform, q_vals[name], 'b-')
    ax.set_ylabel(f'{name} (рад/мм)' if name in ('theta','phi') else f'{name} (мм)')
    ax.set_title(f'Позиция {name}(t)')
    ax.grid(True)
    # Скорость
    ax = axes[idx, 1]
    ax.plot(t_uniform, dq_vals[name], 'g-')
    ax.set_ylabel(f'd{name}/dt' + (' (рад/с)' if name in ('theta','phi') else ' (мм/с)'))
    ax.set_title(f'Скорость {name}(t)')
    ax.grid(True)
    # Ускорение
    ax = axes[idx, 2]
    ax.plot(t_uniform, ddq_vals[name], 'r-')
    ax.set_ylabel(f'd²{name}/dt²' + (' (рад/с²)' if name in ('theta','phi') else ' (мм/с²)'))
    ax.set_title(f'Ускорение {name}(t)')
    ax.grid(True)

for i in range(4):
    axes[i,0].set_xlabel('Время, с')
    axes[i,1].set_xlabel('Время, с')
    axes[i,2].set_xlabel('Время, с')

plt.tight_layout()
plt.savefig('temporal_kinematics.png', dpi=150)
plt.show()

# =============================================================================
# Дополнительно: профиль скорости V(s) и время t(s)
# =============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(s_array, result['V_s'], 'm-')
ax1.set_xlabel('s, мм')
ax1.set_ylabel('V(s), мм/с')
ax1.set_title('Скорость развёртки V(s)')
ax1.grid(True)

ax2.plot(s_array, t, 'c-')
ax2.set_xlabel('s, мм')
ax2.set_ylabel('t, с')
ax2.set_title('Время t(s)')
ax2.grid(True)
plt.tight_layout()
plt.savefig('temporal_profile.png', dpi=150)
plt.show()

# =============================================================================
# Сохранение результатов
# =============================================================================
save_dict = {
    't_uniform': t_uniform,
    'theta': q_vals['theta'],
    'Z': q_vals['Z'],
    'R': q_vals['R'],
    'phi': q_vals['phi'],
    'dtheta_dt': dq_vals['theta'],
    'dZ_dt': dq_vals['Z'],
    'dR_dt': dq_vals['R'],
    'dphi_dt': dq_vals['phi'],
    'ddtheta_dt2': ddq_vals['theta'],
    'ddZ_dt2': ddq_vals['Z'],
    'ddR_dt2': ddq_vals['R'],
    'ddphi_dt2': ddq_vals['phi'],
    'total_time': total_time,
    't_array_original': t,
    'V_s': result['V_s'],
    's_array': s_array
}
scipy.io.savemat('temporal_results.mat', save_dict)

print(f"Полное время цикла: {total_time:.2f} с")
print("Результаты сохранены в temporal_results.mat")