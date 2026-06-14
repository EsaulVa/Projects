% main_generate_physically_correct_LU.m
% Генерация физически корректной линии укладки (ЛУ) на сглаженной поверхности E3
% на основе извлечения и сглаживания закона намотки с исходной ЛУ.
clear; clc; close all;

%% 1. Параметры сглаживания поверхности
FILTER_N       = 1800;      % число точек дискретизации меридиана
FILTER_TAU     = 1e-1;       % шаг диффузии (мм^2)
FILTER_NSTEPS  = 3;        % число шагов => полное время T = tau * nsteps = 50 мм^2
PRESERVE_Z     = true;      % сохранять параметризацию по z (критично для ОДУ по z)

%% 2. Загрузка и сглаживание исходного меридиана (оправка E2_raw -> E3)
csv_name = 'meridian_E2_raw_rz.csv';
if ~exist(csv_name, 'file')
    error('Файл %s не найден. Убедитесь, что он находится в текущей папке.', csv_name);
end
data = readtable(csv_name);
z_raw = data.z;
r_raw = data.r;

fprintf('Загружен исходный меридиан: %d точек, z ? [%.3f, %.3f] мм\n', ...
    length(z_raw), min(z_raw), max(z_raw));

% Создаем исходную дискретную поверхность
E2_raw = Filter.DiscreteRevolutionSurface(z_raw, r_raw);

% Запускаем диффузию Лапласа-Бельтрами
fprintf('\nЗапуск диффузии: tau=%.1f, nsteps=%d, N=%d, T=%.1f мм^2\n', ...
    FILTER_TAU, FILTER_NSTEPS, FILTER_N, FILTER_TAU*FILTER_NSTEPS);

E3 = Filter.DiffusedRevolutionSurface(E2_raw, FILTER_N, FILTER_TAU, FILTER_NSTEPS, ...
    Filter.BoundaryCondition.dirichlet(0.0), ...   % левый торец жестко зафиксирован
    Filter.BoundaryCondition.dirichlet(0.0), ...   % правый торец жестко зафиксирован
    'PreserveZParameter', PRESERVE_Z, ...
    'SaveMeridianPath', 'meridian_E3_smooth.csv');

fprintf('Сглаженная оправка E3 создана: z ? [%.3f, %.3f] мм\n', E3.z_min, E3.z_max);

%% 3. Загрузка и подготовка исходной линии укладки (LU)
if ~exist('LU_data.mat', 'file')
    error('Файл LU_data.mat не найден. Ожидается переменная r (матрица 3?N).');
end
load('LU_data.mat', 'r');

% Приводим к формату 3?N
if size(r, 1) ~= 3
    r = r';
end
N_pts = size(r, 2);
fprintf('Загружена исходная ЛУ: %d точек\n', N_pts);

% Вычисляем длину дуги s и касательные tau для исходной ЛУ (если их нет в файле)
s_vals = zeros(1, N_pts);
tau_s = zeros(3, N_pts);
for i = 2:N_pts
    ds = norm(r(:, i) - r(:, i-1));
    s_vals(i) = s_vals(i-1) + ds;
    tau_s(:, i) = (r(:, i) - r(:, i-1)) / ds;
end
tau_s(:, 1) = tau_s(:, 2); % Граничное условие для первой точки

fprintf('Длина исходной ЛУ: %.3f мм\n', s_vals(end));

%% 4. Генерация физически корректной ЛУ на сглаженной поверхности E3
fprintf('\nГенерация новой ЛУ на основе извлеченного закона намотки...\n');
lu_on_E3 = generatePhysicallyCorrectLU(E2_raw, E3, s_vals, r, tau_s);

%% 5. Визуализация: Сглаженная поверхность + Исходная ЛУ + Новая ЛУ
figure('Name', 'Сравнение линий укладки на сглаженной поверхности E3', ...
       'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; axis equal; view(45, 25);

% --- Сглаженная поверхность E3 (полупрозрачная) ---
N_phi = 60;   
N_z   = 50;   
z_vis = linspace(E3.z_min, E3.z_max, N_z);
v_vis = linspace(0, 2*pi, N_phi);
[Zgrid, Vgrid] = meshgrid(z_vis, v_vis);
Xsurf = zeros(size(Zgrid));
Ysurf = zeros(size(Zgrid));

for i = 1:size(Zgrid, 1)
    for j = 1:size(Zgrid, 2)
        pt = E3.position_by_z(Zgrid(i, j), Vgrid(i, j));
        Xsurf(i, j) = pt(1);
        Ysurf(i, j) = pt(2);
    end
end
surf(Xsurf, Ysurf, Zgrid, 'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
     'FaceColor', [0.7, 0.8, 0.9], 'DisplayName', 'Сглаженная оправка E3');

% --- Каркас поверхности для ориентира ---
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, N_z);
    for i = 1:N_z
        pts(:, i) = E3.position_by_z(z_vis(i), v0);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% % --- Исходная ЛУ (синяя сплошная) ---
% plot3(r(1,:), r(2,:), r(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Исходная ЛУ (сырая)');

% --- Новая физически корректная ЛУ на E3 (красная пунктирная) ---
s_plot_new = linspace(0, lu_on_E3.totalLength(), 500);
pts_new = zeros(3, length(s_plot_new));
for i = 1:length(s_plot_new)
    pts_new(:, i) = lu_on_E3.getPoint(s_plot_new(i));
end
plot3(pts_new(1,:), pts_new(2,:), pts_new(3,:), 'r--', 'LineWidth', 2.5, ...
      'DisplayName', 'Новая ЛУ на E3 (по закону намотки)');

xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
title('Сравнение исходной и физически корректной ЛУ на сглаженной оправке');
legend('Location', 'best');
hold off;

%% 6. Сохранение результатов для следующих этапов пайплайна
save('E3_smoothed.mat', 'E3');
save('lu_on_E3.mat', 'lu_on_E3');
fprintf('\nУспешно! Объекты сохранены:\n');
fprintf('  - E3_smoothed.mat (сглаженная поверхность)\n');
fprintf('  - lu_on_E3.mat (физически корректная линия укладки)\n');
fprintf('Теперь можно запускать main_trace_on_safety.m\n');


%% =========================================================================
% ЛОКАЛЬНАЯ ФУНКЦИЯ: Генерация физически корректной ЛУ
% =========================================================================
function lu_on_E3 = generatePhysicallyCorrectLU(E2_raw, E3, s_vals, r_s, tau_s)
% Генерирует ЛУ на сглаженной поверхности E3, извлекая закон намотки
% (угол alpha) как функцию длины дуги s.

    N = size(r_s, 2);
    alpha_vals = zeros(N, 1);

    % --- ШАГ 1: Извлечение угла намотки alpha(s) ---
    fprintf('  Извлечение закона намотки с исходной поверхности...\n');
    for i = 1:N
        pt = r_s(:, i);
        tau = tau_s(:, i);
        
        z = pt(3);
        v = atan2(pt(2), pt(1));
        
        % Получаем базис исходной поверхности E2_raw в точке (z, v)
        d_E2 = E2_raw.derivatives(z, v);
        e_merid = d_E2.ru / norm(d_E2.ru);
        e_circ  = d_E2.rv / norm(d_E2.rv);
        
        % Угол через atan2 для непрерывности знака
        cos_a = dot(tau, e_merid);
        sin_a = dot(tau, e_circ);
        alpha_vals(i) = atan2(sin_a, cos_a);
    end

    % --- ШАГ 2: Сглаживание 1D-профиля alpha(s) ---
    fprintf('  Сглаживание закона намотки по длине дуги s...\n');
    
    % s_vals уже монотонно возрастает (длина дуги), дубликатов нет
    window = max(5, round(N / 50));
    if mod(window, 2) == 0, window = window + 1; end
    
    alpha_smooth = smoothdata(alpha_vals, 'sgolay', window);
    
    % Интерполятор alpha(s)
    alpha_func = @(s) interp1(s_vals, alpha_smooth, s, 'pchip', 'extrap');

    % --- ШАГ 3: Интеграция системы ОДУ по s на поверхности E3 ---
    fprintf('  Интеграция системы ОДУ по s (ode45)...\n');
    
    s0 = s_vals(1);
    s_end = s_vals(end);
    z0 = r_s(3, 1);  % начальная Z
    v0 = atan2(r_s(2, 1), r_s(1, 1));  % начальный угол V
    
    % Система ОДУ: [dz/ds; dv/ds]
    % dz/ds = cos(alpha) / sqrt(1 + (dr/dz)^2)
    % dv/ds = sin(alpha) / r(z)
    odefun = @(s, y) winding_ode(s, y, E3, alpha_func);
    
    y0 = [z0; v0];
    ode_options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
    
    [s_new, y_new] = ode45(odefun, [s0, s_end], y0, ode_options);
    z_new = y_new(:, 1);
    v_new = y_new(:, 2);
    
    % --- ШАГ 4: Формирование 3D-траектории ---
    N_new = length(z_new);
    pts_new = zeros(3, N_new);
    for i = 1:N_new
        pt = E3.position_by_z(z_new(i), v_new(i));
        pts_new(:, i) = pt(:);
    end
    
    lu_on_E3 = InverseTask.Trajectory(pts_new);
    
    fprintf('  Успешно! Длина новой ЛУ: %.3f мм (исходная: %.3f мм)\n', ...
        lu_on_E3.totalLength(), s_end - s0);
end

% Вспомогательная функция: система ОДУ для намотки
function dyds = winding_ode(s, y, E3, alpha_func)
    z = y(1);
    v = y(2);
    
    try
        % Ограничиваем z в пределах поверхности
        z = max(E3.z_min + 1e-6, min(E3.z_max - 1e-6, z));
        
        r_val = E3.radius(z);
        dr_val = E3.radius_deriv(z);
        alpha_val = alpha_func(s);
        
        % Защита от полюса (r -> 0)
        if r_val < 1e-6
            dv_ds = 0;
        else
            dv_ds = sin(alpha_val) / r_val;
        end
        
        % dz/ds = cos(alpha) / sqrt(1 + (dr/dz)^2)
        dz_ds = cos(alpha_val) / sqrt(1 + dr_val^2);
        
        dyds = [dz_ds; dv_ds];
    catch
        dyds = [0; 0]; % Fallback
    end
end