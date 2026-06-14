% main_synthesize_R.m
% Синтез траектории раскладчика R(z) по линии укладки
clear; clc; close all;

%% 1. Загрузка данных
%% 1. Загрузка исходной оправки E2 (без сглаживания)
csv_name = 'meridian_E2_raw_rz.csv';
if ~exist(csv_name, 'file')
    error('Файл %s не найден.', csv_name);
end
data = readtable(csv_name);
z_raw = data.z;
r_raw = data.r;

E2 = Geometry.DiscreteRevolutionSurface(z_raw, r_raw);
fprintf('1. Загружена исходная оправка E2: z ? [%.3f, %.3f] мм\n', E2.u_min, E2.u_max);

%% 2. Загрузка исходной ЛУ (без сглаживания и проекции)
if ~exist('LU_data.mat', 'file')
    error('Файл LU_data.mat не найден.');
end
load('LU_data.mat', 'r');
if size(r,1) ~= 3
    r = r';
end

lu_original = Geometry.Trajectory(r);
lu_on_E3=lu_original;
fprintf('2. Загружена исходная ЛУ: длина дуги = %.3f мм\n', lu_original.totalLength());

%% 3. Создание поверхности безопасности E1 как DiscreteRevolutionSurface
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;

% Создаем аналитическую поверхность для извлечения меридиана
E1_analytic = Geometry.RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);

% Извлекаем меридиан (z, r)
N_meridian = 1000;
z_vals = linspace(E1_analytic.u_min, E1_analytic.u_max, N_meridian)';
r_vals = zeros(size(z_vals));
for i = 1:N_meridian
    pt = E1_analytic.position(z_vals(i), 0.0);
    r_vals(i) = sqrt(pt(1)^2 + pt(2)^2);
end

% Создаем DiscreteRevolutionSurface (имеет унифицированный интерфейс)
E1 = Geometry.DiscreteRevolutionSurface(z_vals, r_vals);
fprintf('3. Создана E1 (DiscreteRevolutionSurface): z ? [%.3f, %.3f] мм\n', ...
    E1.u_min, E1.u_max);
E3=E2;
%% 2. Синтез траектории R(z) в режиме трассировки
synth = Geometry.TrajectorySynthesizer(E3, E1, 'shadow');
[R_traj, lambda_s, s_vals] = synth.synthesize(lu_original, 400);

fprintf('Синтезированная траектория R(z): длина = %.3f мм\n', R_traj.totalLength());

%% 3. Визуализация
figure('Name', 'Синтез траектории R(z)', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; view(35, 25);

% % Каркас E3
% z_vis = linspace(E3.u_min, E3.u_max, 40);
% for v0 = [0, pi/2, pi, 3*pi/2]
%     pts = zeros(3, length(z_vis));
%     for ii = 1:length(z_vis)
%         pts(:, ii) = E3.position_by_z(z_vis(ii), v0);
%     end
%     plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5);
% end
z_vis = linspace(E3.u_min, E3.u_max, 40);
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:, ii) = E3.position(z_vis(ii), v0);  % <-- было position_by_z
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5);
end
% Линия укладки
s_plot = linspace(0, lu_on_E3.totalLength(), 500);
pts_lu = zeros(3, length(s_plot));
for i = 1:length(s_plot)
    pts_lu(:, i) = lu_on_E3.getPoint(s_plot(i));
end
plot3(pts_lu(1,:), pts_lu(2,:), pts_lu(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'ЛУ на E3');

% Траектория раскладчика
z_plot = linspace(0, R_traj.totalLength(), 500);
pts_R = zeros(3, length(z_plot));
for i = 1:length(z_plot)
    pts_R(:, i) = R_traj.getPoint(z_plot(i));
end
plot3(pts_R(1,:), pts_R(2,:), pts_R(3,:), 'r--', 'LineWidth', 2, 'DisplayName', 'R(z)');

% Лучи (каждый 50-й)
skip = 10;
for i = 1:skip:length(s_vals)
    p1 = lu_on_E3.getPoint(s_vals(i));
    p2 = R_traj.getPoint(z_plot(min(i, length(z_plot))));
    plot3([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], 'g-', 'LineWidth', 0.5);
end

xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
title('Синтез траектории раскладчика R(z)');
legend('Location', 'best');
hold off;

%% 4. Сохранение результатов
save('R_trajectory_synthesized.mat', 'R_traj', 'lambda_s', 's_vals');
fprintf('Результаты сохранены в R_trajectory_synthesized.mat\n');