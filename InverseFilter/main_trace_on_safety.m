% main_trace_on_safety.m
% Трассировка лучей (модель тени) от спроецированной линии укладки (на E3)
% до поверхности безопасности E1, используя готовую функцию trace_ray.

clear; clc; close all;

%% 1. Загрузка сглаженной оправки E3
if ~exist('E3_smoothed.mat', 'file')
    error('Файл E3_smoothed.mat не найден. Сначала выполните main_visualize_smoothed_with_LU.m');
end
tmp = load('E3_smoothed.mat');
fn = fieldnames(tmp);
E3 = tmp.(fn{1});
fprintf('Загружена оправка (%s): z ∈ [%.3f, %.3f] мм\n', fn{1}, E3.z_min, E3.z_max);

%% 2. Загрузка спроецированной линии укладки (на E3)
if ~exist('lu_on_E3.mat', 'file')
    error('Файл lu_on_E3.mat не найден. Сначала выполните проекцию кривой.');
end
tmp2 = load('lu_on_E3.mat');
fn2 = fieldnames(tmp2);
lu_on_E3 = tmp2.(fn2{1});
fprintf('Загружена ЛУ (%s): длина дуги = %.3f мм\n', fn2{1}, lu_on_E3.totalLength());

%% 3. Создание поверхности безопасности E1 (RevolutionSurface)
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;
E1 = InverseTask.RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);
load('E1_smoothed.mat', 'E1_smooth');
% E1 = E1_smooth;   % или прямо используйте E1_smooth
fprintf('Поверхность безопасности E1 создана: z ∈ [%.3f, %.3f] мм\n', E1.u_min, E1.u_max);

%% 4. Параметры трассировки
num_points = 2500;          % количество точек на ЛУ
t_min = 1.0;
t_max = 2500.0;
s_vals = linspace(0, lu_on_E3.totalLength(), num_points);

R_points = zeros(3, num_points);
lambda_vals = zeros(num_points, 1);
valid_mask = false(num_points, 1);
phi_vals = zeros(num_points, 1);

%% 5. Вспомогательные функции
% Нормаль к E3 в точке (x,y,z)
function n = get_normal_E3(E3, x, y, z)
    v = atan2(y, x);
    s = E3.s_from_z(z);
    n = E3.normal(s, v);
    n = n(:);   % принудительно столбец
end

% Проекция вектора на касательную плоскость
function tau_proj = project_to_tangent_plane(tau, n)
    tau_proj = tau - dot(tau, n) * n;
    if norm(tau_proj) < 1e-12
        tau_proj = [1; 0; 0];
    else
        tau_proj = tau_proj / norm(tau_proj);
    end
end

%% 6. Основной цикл трассировки
fprintf('Начинаем трассировку %d точек...\n', num_points);
for i = 1:num_points
    s = s_vals(i);
    r_point = lu_on_E3.getPoint(s);
    r_point = r_point(:);   % столбец 3x1
    tau_lu  = lu_on_E3.getTangent(s);
    tau_lu  = tau_lu(:);
    
    n = get_normal_E3(E3, r_point(1), r_point(2), r_point(3));  % столбец 3x1
    tau_proj = project_to_tangent_plane(tau_lu, n);             % столбец 3x1
    
    % Преобразуем в строки для trace_ray
    r_point_row = r_point(:)';
    tau_proj_row = tau_proj(:)';
    [t, pt_row] = InverseTask.trace_ray(E1, r_point_row, tau_proj_row, t_min, t_max);
    
    if ~isnan(t)
        pt = pt_row(:);   % обратно в столбец
        R_points(:,i) = pt;
        lambda_vals(i) = t;
        valid_mask(i) = true;
        phi_vals(i) = dot(pt - r_point, n);
    else
        R_points(:,i) = NaN(3,1);
        lambda_vals(i) = Inf;
        phi_vals(i) = NaN;
    end
end
% for i = 1:num_points
%     s = s_vals(i);
%     r_point = lu_on_E3.getPoint(s); r_point = r_point(:);
%     tau_lu  = lu_on_E3.getTangent(s); tau_lu = tau_lu(:);
% 
%     n = get_normal_E3(E3, r_point(1), r_point(2), r_point(3));
%     tau_proj = project_to_tangent_plane(tau_lu, n);
% 
%     % Используем готовую функцию trace_ray (как в main_shadow_trace.m)
%     [t, pt] = trace_ray(E1, r_point, tau_proj, t_min, t_max);
% 
%     if ~isnan(t)
%         R_points(:,i) = pt;
%         lambda_vals(i) = t;
%         valid_mask(i) = true;
%         phi_vals(i) = dot(pt - r_point, n);
%     else
%         R_points(:,i) = NaN(3,1);
%         lambda_vals(i) = Inf;
%         phi_vals(i) = NaN;
%     end
% 
%     if mod(i, 50) == 0
%         fprintf('  Обработано %d/%d\n', i, num_points);
%     end
% end
fprintf('Успешно: %d из %d\n', sum(valid_mask), num_points);

%% 7. Сохранение результатов
valid_idx = find(valid_mask);
R_valid = R_points(:, valid_idx);
s_valid = s_vals(valid_idx);
save('R_trajectory_on_E1.mat', 'R_valid', 's_valid', 'lambda_vals', 'phi_vals');
fprintf('Сохранено %d точек траектории R(z) в R_trajectory_on_E1.mat\n', size(R_valid,2));

T_out = table(s_valid', R_valid(1,:)', R_valid(2,:)', R_valid(3,:)', ...
              lambda_vals(valid_idx), phi_vals(valid_idx), ...
              'VariableNames', {'s', 'X', 'Y', 'Z', 'lambda', 'phi'});
writetable(T_out, 'tsn_on_E1.csv');
fprintf('CSV сохранён в tsn_on_E1.csv\n');

%% 8. Визуализация (аналогично main_shadow_trace.m)
figure('Name', 'Трассировка на E1', 'Color', 'w');
hold on; grid on; axis equal; view(3);

% Поверхность E1 (полупрозрачная)
u_plot = linspace(E1.u_min, E1.u_max, 60);
v_plot = linspace(0, 2*pi, 40);
[Xs, Ys, Zs] = meshgrid(0,0,0);
for i = 1:length(u_plot)
    for j = 1:length(v_plot)
        p = E1.position(u_plot(i), v_plot(j));
        Xs(i,j) = p(1); Ys(i,j) = p(2); Zs(i,j) = p(3);
    end
end
surf(Xs, Ys, Zs, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'FaceColor', [1 0.5 0.5]);

% Сглаженная оправка E3 (каркас)
z_vis = linspace(E3.z_min, E3.z_max, 40);
v_vis = linspace(0, 2*pi, 30);
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:,ii) = E3.position_by_z(z_vis(ii), v0);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5);
end

% Линия укладки на E3 (синяя)
pts_lu = zeros(3, num_points);
for i = 1:num_points
    pts_lu(:,i) = lu_on_E3.getPoint(s_vals(i));
end
plot3(pts_lu(1,:), pts_lu(2,:), pts_lu(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'ЛУ на E3');

% Траектория R(z) на E1 (красная)
plot3(R_valid(1,:), R_valid(2,:), R_valid(3,:), 'r-', 'LineWidth', 2, 'DisplayName', 'ТСН на E1');
scatter3(R_valid(1,:), R_valid(2,:), R_valid(3,:), 20, 'r', 'filled');

% Лучи (каждый 10-й)
skip = max(1, floor(num_points/50));
for idx = valid_idx(1:skip:end)
    p1 = pts_lu(:,idx);
    p2 = R_points(:,idx);
    if ~any(isnan(p2))
        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'g-', 'LineWidth', 0.8);
    end
end

xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
title('Трассировка лучей (модель тени) на поверхность безопасности');
legend('Location', 'best');
hold off;

% График невязки
figure('Name', 'Невязка связи');
plot(s_valid, phi_vals(valid_idx), 'b.-', 'MarkerSize', 8);
xlabel('s (длина дуги ЛУ)'); ylabel('\Phi');
title('Невязка \Phi(s) = \langle R-r, n \rangle');
grid on;