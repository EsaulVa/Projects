% inverse_revolution_trace.m
% Обратная задача: по траектории точки схода R(z) восстановить линию укладки
% на поверхности вращения (оправке E2) и сравнить с эталоном.
clear; clc; close all;

% ======================================================================
% 1. Загрузка данных
% ======================================================================
if ~exist('LU_data.mat', 'file')
    error('LU_data.mat не найден');
end
if ~exist('R_points.mat', 'file')
    error('R_points.mat не найден (запустите сначала main_shadow_trace)');
end

data_lu = load('LU_data.mat');
r_etalon = data_lu.r;          % матрица 3×N (столбцы)
if size(r_etalon,1) ~= 3
    r_etalon = r_etalon';
end
fprintf('Загружена эталонная ЛУ: %d точек\n', size(r_etalon,2));

data_r = load('R_points.mat');
R_valid = data_r.R_valid;      % матрица 3×M (глобальная система)
s_valid = data_r.s_valid;      % соответствующие s (длина дуги ЛУ)
fprintf('Загружена траектория R(z): %d точек\n', size(R_valid,2));

% ======================================================================
% 2. Поверхности
% ======================================================================
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764];
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, ...
                 39582.6812110246392, -43518.6731429065403, 19122.1758646943599];
bound_opravka = [0, 234.27, 534.27, 768.54];
cyl_r_opravka = 251.705;
E2 = RevolutionSurface(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka);

phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;
E1 = RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);
z_offset = (bound_safe(4) - bound_opravka(4)) / 2;   % = 93.708 мм
fprintf('z_offset = %.3f мм\n', z_offset);

% ======================================================================
% 3. Приведение R(z) к локальной системе оправки (ВАЖНО!)
% ======================================================================
R_valid_local = R_valid;
R_valid_local(3,:) = R_valid_local(3,:) - z_offset;   % исправлено: убран *0
fprintf('R(z) переведена в локальную систему (Z -= %.2f мм)\n', z_offset);

% Создаём Trajectory для R(z)
R_traj = Trajectory(R_valid_local);
z_max = R_traj.totalLength();
fprintf('Полная длина траектории R(z) в локальной системе: %.2f мм\n', z_max);

% ======================================================================
% 4. Начальные условия (из эталонной ЛУ) – используем ПЕРВУЮ точку (s=0)
% ======================================================================
lu_traj_ref = Trajectory(r_etalon);
r0 = lu_traj_ref.getPoint(1e-1);                % первая точка
[u0, v0] = E2.uv_from_point(r0');
tau0 = lu_traj_ref.getTangent(1e-2);            % касательная в начале

% Разложение tau0 по базису (ru, rv)
[ru0, rv0] = E2.derivatives(u0, v0);
E0 = dot(ru0, ru0); F0 = dot(ru0, rv0); G0 = dot(rv0, rv0);
M = [E0, F0; F0, G0];
rhs = [dot(tau0, ru0); dot(tau0, rv0)];
uv = M \ rhs;
du0_s = uv(1); dv0_s = uv(2);

fprintf('Начальная точка: u0=%.4f, v0=%.4f\n', u0, v0);
fprintf('Начальные du/ds = %.6f, dv/ds = %.6f\n', du0_s, dv0_s);

% ======================================================================
% 5. Параметры сходимости и интегрирования
% ======================================================================
DeltaZ = 0.5;        % мм
Percentage = 50;     % % (k = -log(1-0.9)/0.5 ≈ 4.605)
Nsteps = 10000;

% ======================================================================
% 6. Восстановление линии укладки (с исправленным знаком ds/dz)
% ======================================================================
[u_rec, v_rec, s_rec, z_rec] = recoverLayer_rev(E2, R_traj, [0, z_max], ...
    u0, v0, du0_s, dv0_s, DeltaZ, Percentage, Nsteps);

% Переводим восстановленную линию в декартовы координаты
Nrec = length(u_rec);
rec_pts = zeros(3, Nrec);
for i = 1:Nrec
    rec_pts(:,i) = E2.position(u_rec(i), v_rec(i));
end

% ======================================================================
% 7. Сравнение с эталонной ЛУ (только если s_rec монотонно возрастает)
% ======================================================================
if any(diff(s_rec) <= 0)
    warning('s_rec не монотонна – восстановление не удалось');
else
    % Вычисляем натуральный параметр эталонной ЛУ
    dist_etalon = sqrt(sum(diff(r_etalon,1,2).^2, 1));
    s_etalon = [0, cumsum(dist_etalon)];
    
    % Обрезаем s_rec, если выходит за пределы
    if s_rec(end) > s_etalon(end)
        idx = s_rec <= s_etalon(end);
        s_rec = s_rec(idx);
        u_rec = u_rec(idx);
        v_rec = v_rec(idx);
        rec_pts = rec_pts(:,idx);
    end
    
    r_etalon_interp = zeros(3, length(s_rec));
    for i = 1:length(s_rec)
        r_etalon_interp(:,i) = interp1(s_etalon, r_etalon', s_rec(i), 'pchip')';
    end
    
    err_xyz = sqrt(sum((rec_pts - r_etalon_interp).^2, 1));
    
    % Визуализация
    figure('Name','Сравнение линий укладки','Color','w');
    plot3(r_etalon(1,:), r_etalon(2,:), r_etalon(3,:), 'b-', 'LineWidth', 1.5); hold on;
    plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'r--', 'LineWidth', 1.5);
    legend('Эталон','Восстановленная'); axis equal; grid on; view(3);
    
    figure('Name','Ошибка','Color','w');
    semilogy(s_rec, err_xyz, 'k.-'); grid on;
    xlabel('s (мм)'); ylabel('Евклидова ошибка (мм)');
    title(sprintf('Ошибка восстановления (k = %.3f)', -log(1-Percentage/100)/DeltaZ));
end