% main_inverse_task_case1_no_smoothing.m
% Случай 1: Никакие поверхности и ЛУ не сглаживаются
% Используется аналитическая E1, дискретная E2 и исходная ЛУ
clear; clc; close all;

fprintf('=== СЛУЧАЙ 1: БЕЗ СГЛАЖИВАНИЯ ===\n');
fprintf('E1: аналитическая RevolutionSurface\n');
fprintf('E2: дискретная DiscreteRevolutionSurface (исходная)\n');
fprintf('ЛУ: исходная из LU_data.mat\n\n');

%% 1. Загрузка исходной оправки E2 (без сглаживания)
csv_name = 'meridian_E2_raw_rz.csv';
if ~exist(csv_name, 'file')
    error('Файл %s не найден.', csv_name);
end
data = readtable(csv_name);
z_raw = data.z;
r_raw = data.r;

E2 = Filter.DiscreteRevolutionSurface(z_raw, r_raw);
fprintf('1. Загружена исходная оправка E2: z ? [%.3f, %.3f] мм\n', E2.u_min, E2.u_max);

%% 2. Загрузка исходной ЛУ (без сглаживания и проекции)
if ~exist('LU_data.mat', 'file')
    error('Файл LU_data.mat не найден.');
end
load('LU_data.mat', 'r');
if size(r,1) ~= 3
    r = r';
end

lu_original = InverseTask.Trajectory(r);
fprintf('2. Загружена исходная ЛУ: длина дуги = %.3f мм\n', lu_original.totalLength());

%% 3. Создание аналитической поверхности E1 (без сглаживания)
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;

E1 = InverseTask.RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);
fprintf('3. Создана аналитическая E1: z ? [%.3f, %.3f] мм\n', E1.u_min, E1.u_max);

%% 4. Трассировка лучей (модель тени) от ЛУ к E1
fprintf('\n4. Трассировка лучей от ЛУ к E1...\n');
num_points = 2500;
s_vals = linspace(0, lu_original.totalLength(), num_points);
R_points = zeros(3, num_points);
valid_mask = false(num_points, 1);
phi_vals = zeros(num_points, 1);

for i = 1:num_points
    s = s_vals(i);
    r_point = lu_original.getPoint(s);
    r_point = r_point(:);
    tau_lu = lu_original.getTangent(s);
    tau_lu = tau_lu(:);
    
    % Нормаль к E2 в точке r_point
    z_coord = r_point(3);
    v_coord = atan2(r_point(2), r_point(1));
    n = E2.getNormal(z_coord, v_coord);
    
    % Проекция касательной на касательную плоскость E2
    tau_proj = tau_lu - dot(tau_lu, n) * n;
    if norm(tau_proj) < 1e-8
        tau_proj = [1; 0; 0];
    else
        tau_proj = tau_proj / norm(tau_proj);
    end
    
    % Трассировка луча до E1
    % Используем функцию trace_ray (если она есть в InverseTask)
    % Если нет - используем численный поиск пересечения
    t_min = 1.0;
    t_max = 3000.0;
    
    % Численный поиск пересечения луча с E1
    % Луч: P(t) = r_point + t * tau_proj
    % Ищем t, при котором P(t) лежит на E1
    % Для аналитической E1: точка (x,y,z) лежит на E1, если sqrt(x^2+y^2) = R(z)
    
    t_found = NaN;
    for t = linspace(t_min, t_max, 1000)
        P = r_point + t * tau_proj;
        rho = sqrt(P(1)^2 + P(2)^2);
        z_P = P(3);
        if z_P >= E1.u_min && z_P <= E1.u_max
            R_E1 = E1.radius(z_P);
            if abs(rho - R_E1) < 0.1  % допуск 0.1 мм
                t_found = t;
                break;
            end
        end
    end
    
    if ~isnan(t_found)
        R_points(:, i) = r_point + t_found * tau_proj;
        valid_mask(i) = true;
        phi_vals(i) = dot(R_points(:, i) - r_point, n);
    end
    
    if mod(i, 500) == 0
        fprintf('   Трассировка: %d/%d точек (найдено: %d)\n', i, num_points, sum(valid_mask));
    end
end

valid_idx = find(valid_mask);
R_valid = R_points(:, valid_idx);
s_valid = s_vals(valid_idx);
fprintf('   Успешно протрассировано: %d из %d точек\n', length(valid_idx), num_points);

% Сохраняем для обратной задачи
save('R_trajectory_case1.mat', 'R_valid', 's_valid');

%% 5. Построение траектории R(z) как объекта Trajectory
[s_sorted, idx] = sort(s_valid);
R_sorted = R_valid(:, idx);
R_traj = InverseTask.Trajectory(R_sorted);
z_R_max = R_traj.totalLength();
fprintf('5. Траектория R(z): длина = %.3f мм\n', z_R_max);

%% 6. Начальные условия для обратной задачи
fprintf('\n6. Установка начальных условий...\n');
s_0 = 0;
u0_geo = lu_original.getPoint(s_0);
u0_geo = u0_geo(:);
tau0_geo = lu_original.getTangent(s_0);
tau0_geo = tau0_geo(:);

% Параметры на поверхности E2
z0 = u0_geo(3);
v0 = atan2(u0_geo(2), u0_geo(1));
s0 = z0;  % Для DiscreteRevolutionSurface параметр u = z

[du0_s, dv0_s] = InverseTask.computeTangentCoeffs(E2, s0, v0, tau0_geo);
fprintf('   Начальные условия: z0=%.3f, v0=%.3f\n', z0, v0);
fprintf('   Начальные скорости: du/ds=%.6e, dv/ds=%.6e\n', du0_s, dv0_s);

%% 7. Вызов recoverLayer
fprintf('\n7. Запуск recoverLayer (обратная задача)...\n');
DeltaZ = 100;
Percentage = 0;

R_func = @(z) R_traj.getPoint(z);
Rprime_func = @(z) R_traj.getTangent(z);

z_span = [0, z_R_max];

try
    tic;
    [u_rec, v_rec, s_rec, z_rec] = InverseTask.recoverLayer(E2, R_func, Rprime_func, ...
        z_span, s0, v0, du0_s, dv0_s, DeltaZ, Percentage);
    toc;
    fprintf('   Восстановлено %d точек\n', length(u_rec));
catch ME
    fprintf('   ОШИБКА в recoverLayer: %s\n', ME.message);
    fprintf('   Примечание: recoverLayer использует RK4, который может не работать без сглаживания.\n');
    
    % Заглушка: простой метод Эйлера
    fprintf('\n   Используем заглушку (Euler метод)...\n');
    Nsteps = 1000;
    zeta_hist = linspace(0, z_R_max, Nsteps+1);
    h = zeta_hist(2) - zeta_hist(1);
    
    u_rec = zeros(1, Nsteps+1);
    v_rec = zeros(1, Nsteps+1);
    s_rec = zeros(1, Nsteps+1);
    z_rec = zeros(1, Nsteps+1);
    
    u = s0; v = v0; s = 0;
    u_rec(1) = u; v_rec(1) = v; s_rec(1) = s; z_rec(1) = 0;
    
    for i = 1:Nsteps
        zeta = zeta_hist(i);
        R = R_func(zeta); R = R(:);
        Rprime = Rprime_func(zeta);
        r = E2.getPoint(u, v); r = r(:);
        n = E2.getNormal(u, v); n = n(:);
        
        tau = (R - r) / norm(R - r);
        [du_s, dv_s] = InverseTask.computeTangentCoeffs(E2, u, v, tau);
        
        [L, M, N] = E2.getSecondFundamental(u, v);
        II = L*du_s^2 + 2*M*du_s*dv_s + N*dv_s^2;
        lambda = norm(R - r);
        
        if abs(II) < 1e-10 || lambda < 1e-10
            dsdz = 0;
        else
            dsdz = dot(Rprime, n) / (lambda * II);
        end
        
        dudz = du_s * dsdz;
        dvdz = dv_s * dsdz;
        
        u = u + h * dudz;
        v = v + h * dvdz;
        s = s + h * dsdz;
        
        u_rec(i+1) = u;
        v_rec(i+1) = v;
        s_rec(i+1) = s;
        z_rec(i+1) = zeta + h;
    end
    fprintf('   Заглушка: восстановлено %d точек\n', length(u_rec));
end

%% 8. Построение восстановленной ЛУ
N_rec = length(u_rec);
rec_pts = zeros(3, N_rec);
for i = 1:N_rec
    rec_pts(:, i) = E2.position(u_rec(i), v_rec(i));
end
rec_traj = InverseTask.Trajectory(rec_pts);
fprintf('8. Восстановленная ЛУ: длина дуги = %.3f мм\n', rec_traj.totalLength());

%% 9. Сравнение с исходной ЛУ
fprintf('\n9. Сравнение с исходной ЛУ...\n');
s_orig = linspace(0, lu_original.totalLength(), 3000);
pts_orig = zeros(3, length(s_orig));
for i = 1:length(s_orig)
    pts_orig(:, i) = lu_original.getPoint(s_orig(i));
end
orig_traj = InverseTask.Trajectory(pts_orig);

orig_pts_at_rec = zeros(3, N_rec);
for i = 1:N_rec
    s_interp = min(s_rec(i), lu_original.totalLength());
    s_interp = max(s_interp, 0);
    orig_pts_at_rec(:, i) = orig_traj.getPoint(s_interp);
end

err_xyz = sqrt(sum((rec_pts - orig_pts_at_rec).^2, 1));
fprintf('   Средняя ошибка: %.2e мм\n', mean(err_xyz));
fprintf('   Максимальная ошибка: %.2e мм\n', max(err_xyz));
fprintf('   Медианная ошибка: %.2e мм\n', median(err_xyz));

%% 10. Визуализация
fprintf('\n10. Построение визуализации...\n');

figure('Name', 'Случай 1: Без сглаживания', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; view(35, 25);

% Каркас E2
z_vis = linspace(E2.u_min, E2.u_max, 40);
for v0_vis = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:, ii) = E2.position(z_vis(ii), v0_vis);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% Исходная ЛУ (синяя)
s_plot = linspace(0, lu_original.totalLength(), 500);
pts_orig_plot = zeros(3, length(s_plot));
for i = 1:length(s_plot)
    pts_orig_plot(:, i) = lu_original.getPoint(s_plot(i));
end
plot3(pts_orig_plot(1,:), pts_orig_plot(2,:), pts_orig_plot(3,:), 'b-', 'LineWidth', 2.5, ...
    'DisplayName', 'Исходная ЛУ');

% Восстановленная ЛУ (красная пунктирная)
plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'r--', 'LineWidth', 2, ...
    'DisplayName', 'Восстановленная ЛУ');

% Траектория R(z) (зелёная)
R_plot = zeros(3, length(z_rec));
for i = 1:length(z_rec)
    R_plot(:, i) = R_func(z_rec(i));
end
plot3(R_plot(1,:), R_plot(2,:), R_plot(3,:), 'g-', 'LineWidth', 1.5, ...
    'DisplayName', 'R(z)');

plot3(u0_geo(1), u0_geo(2), u0_geo(3), 'mo', 'MarkerSize', 12, 'MarkerFaceColor', 'm', ...
    'DisplayName', 'Начальная точка');

xlabel('X, мм', 'FontWeight', 'bold'); 
ylabel('Y, мм', 'FontWeight', 'bold'); 
zlabel('Z, мм', 'FontWeight', 'bold');
title('Случай 1: Без сглаживания', 'FontWeight', 'bold');
legend('Location', 'bestoutside');

r_max_E2 = max(E2.radius(linspace(E2.u_min, E2.u_max, 100)));
xlim([-r_max_E2*1.3, r_max_E2*1.3]);
ylim([-r_max_E2*1.3, r_max_E2*1.3]);
zlim([E2.u_min-20, E2.u_max+20]);
daspect([1, 1, 1]);
hold off;

figure('Name', 'Ошибка восстановления (Случай 1)', 'Color', 'w', 'Position', [150, 150, 900, 500]);
semilogy(s_rec, err_xyz, 'r-', 'LineWidth', 1.5);
xlabel('Длина дуги s, мм', 'FontWeight', 'bold'); 
ylabel('Евклидова ошибка, мм', 'FontWeight', 'bold');
title('Ошибка восстановления линии укладки (Случай 1: без сглаживания)', 'FontWeight', 'bold');
grid on;
yline(1e-6, 'g--', 'LineWidth', 1, 'Label', '1 мкм');
yline(1e-3, 'b--', 'LineWidth', 1, 'Label', '1 мм');

fprintf('\n=== Случай 1 завершен ===\n');
save('case1_results.mat', 'rec_pts', 'err_xyz', 'rec_traj', 'u_rec', 'v_rec', 's_rec', 'z_rec');