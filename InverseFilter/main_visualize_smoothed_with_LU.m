% main_visualize_smoothed_with_LU.m
% Визуализация сглаженной оправки (E3) и исходной линии укладки (LU) из LU_data.mat.
% Аналог client_corridor_3.py: ЛУ отображается как есть, без проецирования.

clear; clc; close all;

%% 1. Параметры сглаживания
FILTER_N       = 1800;      % число точек дискретизации меридиана
FILTER_TAU     = 5.0;      % шаг диффузии (мм^2)
FILTER_NSTEPS  = 10;        % число шагов => T = tau * nsteps
PRESERVE_Z     = true;     % сохранять параметризацию по z (удобно для визуализации)

%% 2. Загрузка исходного меридиана (оправка E2_raw)
csv_name = 'meridian_E2_raw_rz.csv';
if ~exist(csv_name, 'file')
    error('Файл %s не найден. Создайте его (например, из Python-скрипта).', csv_name);
end
data = readtable(csv_name);
z_raw = data.z;
r_raw = data.r;
fprintf('Загружен меридиан: %d точек, z ∈ [%.3f, %.3f] мм\n', ...
    length(z_raw), min(z_raw), max(z_raw));

%% 3. Построение дискретной поверхности вращения (исходная)
E2_raw = Filter.DiscreteRevolutionSurface(z_raw, r_raw);

%% 4. Диффузионное сглаживание -> E3
fprintf('\nЗапуск диффузии: tau=%.1f, nsteps=%d, N=%d, T=%.1f мм^2\n', ...
    FILTER_TAU, FILTER_NSTEPS, FILTER_N, FILTER_TAU*FILTER_NSTEPS);
E3 = Filter.DiffusedRevolutionSurface(E2_raw, FILTER_N, FILTER_TAU, FILTER_NSTEPS, ...
    Filter.BoundaryCondition.dirichlet(0.0), ...   % левый конец фиксирован
    Filter.BoundaryCondition.dirichlet(0.0), ...   % правый конец фиксирован
    'PreserveZParameter', PRESERVE_Z, ...
    'SaveMeridianPath', 'meridian_E3_smooth.csv');

if PRESERVE_Z
    fprintf('Сглаженная оправка E3: z ∈ [%.3f, %.3f] мм, s ∈ [%.3f, %.3f] мм\n', ...
        E3.z_min, E3.z_max, E3.u_min, E3.u_max);
else
    fprintf('Сглаженная оправка E3: s ∈ [%.3f, %.3f] мм\n', E3.u_min, E3.u_max);
end

%% 5. Загрузка исходной линии укладки (как в client_corridor_3.py)
if ~exist('LU_data.mat', 'file')
    error('Файл LU_data.mat не найден. Ожидается переменная r (3×N точек).');
end
load('LU_data.mat', 'r');   % загружаем матрицу 3×N или N×3
% Приводим к размеру 3×N
if size(r,1) ~= 3
    r = r';
end
fprintf('Загружена исходная ЛУ: %d точек\n', size(r,2));

% Для единообразия создадим объект Trajectory (как в Python)
lu_traj = InverseTask.Trajectory(r);
fprintf('Исходная ЛУ: длина дуги = %.3f мм\n', lu_traj.totalLength());

%% 6. Визуализация: сглаженная оправка + исходная ЛУ
figure('Name', 'Сглаженная оправка и исходная ЛУ', 'Color', 'w', 'Position', [100 100 1200 800]);
hold on; grid on; axis equal; view(45, 25);

% --- Сглаженная поверхность E3 (полупрозрачная) ---
N_phi = 60;   % количество сечений по углу
N_z   = 40;   % количество сечений по высоте
z_vis = linspace(E3.z_min, E3.z_max, N_z);
v_vis = linspace(0, 2*pi, N_phi);
[Zgrid, Vgrid] = meshgrid(z_vis, v_vis);
Xsurf = zeros(size(Zgrid));
Ysurf = zeros(size(Zgrid));
for i = 1:size(Zgrid,1)
    for j = 1:size(Zgrid,2)
        pt = E3.position_by_z(Zgrid(i,j), Vgrid(i,j));
        Xsurf(i,j) = pt(1);
        Ysurf(i,j) = pt(2);
    end
end
surf(Xsurf, Ysurf, Zgrid, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', [0.8 0.5 0.2], ...
    'DisplayName', 'Сглаженная оправка E3');

% --- Каркас (меридианы и параллели) для ориентира ---
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, N_z);
    for i = 1:N_z
        pts(:,i) = E3.position_by_z(z_vis(i), v0);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
for z0 = z_vis(1:5:end)
    pts = zeros(3, N_phi);
    for i = 1:N_phi
        pts(:,i) = E3.position_by_z(z0, v_vis(i));
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% --- Исходная линия укладки (синяя сплошная) ---
plot3(r(1,:), r(2,:), r(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Исходная ЛУ');

xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
title('Сглаженная оправка и исходная линия укладки');
legend('Location', 'best');

% E3 – сглаженная поверхность, lu_traj – исходная линия укладки
curve_proj = Filter.CurveProjector(E3);
lu_on_E3 = curve_proj.project(lu_traj);
% Визуализация:
s_vals = linspace(0, lu_on_E3.totalLength(), 200);
pts = zeros(3,200);
for i=1:200, pts(:,i) = lu_on_E3.getPoint(s_vals(i)); end
plot3(pts(1,:), pts(2,:), pts(3,:), 'r-', 'LineWidth',2);

hold off;

%% 7. Сохранение объектов для дальнейшего использования (опционально)
save('E3_smoothed.mat', 'E3');
save('LU_original_traj.mat', 'lu_traj');
save('lu_on_E3.mat','lu_on_E3');
fprintf('\nОбъекты сохранены: E3 (сглаженная оправка) и lu_traj (исходная ЛУ).\n');