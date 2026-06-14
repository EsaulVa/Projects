% main_trace_on_safety.m
% ЭТАП 3: Трассировка лучей (модель тени) от физически корректной линии укладки 
% на сглаженной оправке E3 до сглаженной поверхности безопасности E1.
%
% ВАЖНО: Перед запуском этого скрипта убедитесь, что вы выполнили 
% main_generate_physically_correct_LU.m, чтобы файл lu_on_E3.mat содержал 
% линию, сгенерированную по закону намотки, а не ортогональной проекцией.

clear; clc; close all;

%% 1. Загрузка сглаженной оправки E3
if ~exist('E3_smoothed.mat', 'file')
    error('Файл E3_smoothed.mat не найден. Сначала выполните main_generate_physically_correct_LU.m');
end
tmp = load('E3_smoothed.mat');
fn = fieldnames(tmp);
E3 = tmp.(fn{1});
fprintf('1. Загружена сглаженная оправка E3: z ? [%.3f, %.3f] мм\n', E3.z_min, E3.z_max);

%% 2. Загрузка физически корректной линии укладки на E3
if ~exist('lu_on_E3.mat', 'file')
    error('Ф??? lu_on_E3.mat не найден. Сначала выполните main_generate_physically_correct_LU.m');
end
tmp2 = load('lu_on_E3.mat');
fn2 = fieldnames(tmp2);
lu_on_E3 = tmp2.(fn2{1});
fprintf('2. Загружена физически корректная ЛУ на E3: длина дуги = %.3f мм\n', lu_on_E3.totalLength());

%% 3. Загрузка сглаженной поверхности безопасности E1
if ~exist('E1_smoothed.mat', 'file')
    error('Файл E1_smoothed.mat не найден. Сначала выполните main_smooth_E1.m');
end
tmp3 = load('E1_smoothed.mat');
fn3 = fieldnames(tmp3);
E1 = tmp3.(fn3{1});
fprintf('3. Загружена сглаженная поверхность безопасности E1: z ? [%.3f, %.3f] мм\n', E1.z_min, E1.z_max);

%% 4. Параметры трассировки
num_points = 2500;          % количество точек трассировки вдоль ЛУ
t_min = 1.0;                % минимальная длина луча (мм)
t_max = 3000.0;             % максимальная длина луча (мм)

s_vals = linspace(0, lu_on_E3.totalLength(), num_points);
R_points = zeros(3, num_points);
lambda_vals = zeros(num_points, 1);
valid_mask = false(num_points, 1);
phi_vals = zeros(num_points, 1); % Невязка контакта <R-r, n>

%% 5. Основной цикл трассировки
fprintf('\n4. Начинаем трассировку %d точек (модель тени)... ', num_points);
tic;

for i = 1:num_points
    s = s_vals(i);
    
    % 5.1. Получаем точку и касательную исходной ЛУ на E3
    r_point = lu_on_E3.getPoint(s);
    r_point = r_point(:);   % столбец 3x1
    tau_lu  = lu_on_E3.getTangent(s);
    tau_lu  = tau_lu(:);
    
    % 5.2. Получаем нормаль к поверхности E3 в этой точке
    z_coord = r_point(3);
    v_coord = atan2(r_point(2), r_point(1));
    s_merid = E3.s_from_z(z_coord);
    n = E3.normal(s_merid, v_coord);
    n = n(:);
    
    % 5.3. Проецируем касательную ЛУ на касательную плоскость E3
    % Это и есть "модель тени": нить лежит в касательной плоскости оправки
    tau_proj = tau_lu - dot(tau_lu, n) * n;
    if norm(tau_proj) < 1e-8
        % Вырожденный случай (нить идет строго по нормали, маловероятно для намотки)
        tau_proj = [1; 0; 0]; 
    else
        tau_proj = tau_proj / norm(tau_proj);
    end
    
    % 5.4. Трассировка луча до поверхности E1
    % Примечание: trace_ray должен принимать строки или столбцы в зависимости от вашей реализации.
    % Здесь передаем строки, как в оригинальном коде.
    [t, pt_row] = InverseTask.trace_ray(E1, r_point(:)', tau_proj(:)', t_min, t_max);
    
    if ~isnan(t)
        pt = pt_row(:);   % обратно в столбец 3x1
        R_points(:, i) = pt;
        lambda_vals(i) = t;
        valid_mask(i) = true;
        % Вычисляем невязку условия контакта (должна быть близка к 0)
        phi_vals(i) = dot(pt - r_point, n);
    else
        R_points(:, i) = NaN(3, 1);
        lambda_vals(i) = Inf;
        phi_vals(i) = NaN;
    end
    
    if mod(i, 500) == 0
        fprintf('.');
    end
end
toc;
fprintf(' Готово!\n');

valid_count = sum(valid_mask);
fprintf('Успешно протрассировано: %d из %d точек (%.1f%%)\n', valid_count, num_points, 100*valid_count/num_points);

%% 6. Сохранение результатов для Этапа 4 (Обратная задача)
valid_idx = find(valid_mask);
R_valid = R_points(:, valid_idx);
s_valid = s_vals(valid_idx);

save('R_trajectory_on_E1.mat', 'R_valid', 's_valid', 'lambda_vals', 'phi_vals');
fprintf('\n5. Результаты сохранены в R_trajectory_on_E1.mat\n');

% Сохранение в CSV для внешнего анализа или построения графиков в Python/Excel
T_out = table(s_valid', R_valid(1,:)', R_valid(2,:)', R_valid(3,:)', ...
    lambda_vals(valid_idx), phi_vals(valid_idx), ...
    'VariableNames', {'s_lu', 'X_R', 'Y_R', 'Z_R', 'lambda', 'phi_contact'});
writetable(T_out, 'tsn_on_E1.csv');
fprintf('   Таблица данных сохранена в tsn_on_E1.csv\n');

%% 7. Визуализация результатов
fprintf('\n6. Построение визуализации...\n');
figure('Name', 'Трассировка на поверхность безопасности E1', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; axis equal; view(35, 25);

% --- 7.1. Сглаженная поверхность E1 (полупрозрачная) ---
u_plot = linspace(E1.z_min, E1.z_max, 60);
v_plot = linspace(0, 2*pi, 40);
[Xs, Ys, Zs] = deal(zeros(length(u_plot), length(v_plot)));
for i = 1:length(u_plot)
    for j = 1:length(v_plot)
        p = E1.position_by_z(u_plot(i), v_plot(j));
        Xs(i,j) = p(1); Ys(i,j) = p(2); Zs(i,j) = p(3);
    end
end
surf(Xs, Ys, Zs, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'FaceColor', [1, 0.6, 0.6], ...
    'DisplayName', 'Поверхность безопасности E1 (сглаж.)');

% --- 7.2. Каркас сглаженной оправки E3 ---
z_vis = linspace(E3.z_min, E3.z_max, 40);
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:,ii) = E3.position_by_z(z_vis(ii), v0);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% --- 7.3. Физически корректная ЛУ на E3 (синяя сплошная) ---
s_lu_plot = linspace(0, lu_on_E3.totalLength(), 500);
pts_lu = zeros(3, length(s_lu_plot));
for i = 1:length(s_lu_plot)
    pts_lu(:,i) = lu_on_E3.getPoint(s_lu_plot(i));
end
plot3(pts_lu(1,:), pts_lu(2,:), pts_lu(3,:), 'b-', 'LineWidth', 2.5, 'DisplayName', 'Физически корректная ЛУ на E3');

% --- 7.4. Траектория точки схода R(z) на E1 (красная пунктирная) ---
plot3(R_valid(1,:), R_valid(2,:), R_valid(3,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Траектория R(z) на E1');

% --- 7.5. Демонстрационные лучи (каждый N-й для наглядности) ---
skip = max(1, floor(valid_count / 30));
ray_idx = valid_idx(1:skip:end);

for k = 1:length(ray_idx)
    idx = ray_idx(k); % idx - это ОРИГИНАЛЬНЫЙ индекс в массивах s_vals и R_points
    
    % 1. Получаем точку на ЛУ напрямую по оригинальному индексу
    current_s = s_vals(idx);
    p1 = lu_on_E3.getPoint(current_s);
    
    % 2. Получаем соответствующую точку на поверхности E1
    p2 = R_points(:, idx);
    
    % 3. Рисуем луч, если трассировка была успешной (нет NaN)
    if ~any(isnan(p2))
        plot3([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], ...
              'g-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
    end
end

%% 8. График невязки условия контакта
figure('Name', 'Невязка условия контакта \Phi', 'Color', 'w', 'Position', [150, 150, 800, 400]);
valid_phi = phi_vals(valid_mask);
plot(s_valid, valid_phi, 'b.-', 'MarkerSize', 6, 'LineWidth', 1);
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('Длина дуги ЛУ s, мм', 'FontWeight', 'bold'); 
ylabel('\Phi = \langle R-r, n \rangle, мм', 'FontWeight', 'bold');
title('Невязка условия касания (должна быть близка к 0)', 'FontWeight', 'bold');
grid on;

fprintf('Все этапы трассировки успешно завершены.\n');
fprintf('Теперь можно запускать main_inverse_task.m для проверки DAE-решателя.\n');