% main_inverse_task_case3_smooth_E2_LU.m
% Случай 3: Сглажены оправка E2 (-> E3) и линия укладки (-> lu_on_E3).
% E1 - аналитическая (без сглаживания).
clear; clc; close all;

fprintf('=== СЛУЧАЙ 3: СГЛАЖЕНЫ E2 И ЛУ ===\n');
fprintf('E1: аналитическая RevolutionSurface\n');
fprintf('E2: сглаженная DiffusedRevolutionSurface (E3)\n');
fprintf('ЛУ: физически корректная (lu_on_E3)\n\n');

%% 1. Загрузка данных
% Сглаженная оправка E3
if ~exist('E3_smoothed.mat', 'file')
    error('Файл E3_smoothed.mat не найден. Сначала выполните main_generate_physically_correct_LU.m');
end
load('E3_smoothed.mat', 'E3');
fprintf('1. Загружена сглаженная оправка E3: z ? [%.3f, %.3f] мм\n', E3.z_min, E3.z_max);

% Физически корректная ЛУ на E3
if ~exist('lu_on_E3.mat', 'file')
    error('Файл lu_on_E3.mat не найден.');
end
load('lu_on_E3.mat', 'lu_on_E3');
fprintf('2. Загружена физически корректная ЛУ на E3: длина дуги = %.3f мм\n', lu_on_E3.totalLength());

% Исходная ЛУ на E2 (для сравнения)
if ~exist('LU_data.mat', 'file')
    error('Файл LU_data.mat не найден.');
end
load('LU_data.mat', 'r');
if size(r, 1) ~= 3, r = r'; end
lu_orig = InverseTask.Trajectory(r);
fprintf('3. Загружена исходная ЛУ на E2: длина дуги = %.3f мм\n', lu_orig.totalLength());

%% 2. Создание аналитической поверхности E1
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;
E1 = InverseTask.RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);
fprintf('4. Создана аналитическая E1: z ? [%.3f, %.3f] мм\n', E1.u_min, E1.u_max);

%% 3. Трассировка лучей от lu_on_E3 к E1
fprintf('\n5. Трассировка лучей от lu_on_E3 к E1...\n');
num_points = 2500;
s_vals = linspace(0, lu_on_E3.totalLength(), num_points);
R_points = zeros(3, num_points);
valid_mask = false(num_points, 1);

for i = 1:num_points
    s = s_vals(i);
    r_point = lu_on_E3.getPoint(s); r_point = r_point(:);
    tau_lu  = lu_on_E3.getTangent(s); tau_lu = tau_lu(:);
    
    % Нормаль к E3
    z_coord = r_point(3);
    v_coord = atan2(r_point(2), r_point(1));
    s_merid = E3.s_from_z(z_coord);
    n = E3.normal(s_merid, v_coord); n = n(:);
    
    % Проекция касательной
    tau_proj = tau_lu - dot(tau_lu, n) * n;
    if norm(tau_proj) < 1e-8
        tau_proj = [1; 0; 0];
    else
        tau_proj = tau_proj / norm(tau_proj);
    end
    
    % Численный поиск пересечения с E1
    t_min = 1.0; t_max = 3000.0;
    t_found = NaN;
    t_vec = linspace(t_min, t_max, 2000);
    for t = t_vec
        P = r_point + t * tau_proj;
        rho = sqrt(P(1)^2 + P(2)^2);
        z_P = P(3);
        if z_P >= E1.u_min && z_P <= E1.u_max
            R_E1 = E1.radius(z_P);
            if abs(rho - R_E1) < 0.5 
                t_found = t;
                break;
            end
        end
    end
    
    if ~isnan(t_found)
        R_points(:, i) = r_point + t_found * tau_proj;
        valid_mask(i) = true;
    end
    
    if mod(i, 500) == 0
        fprintf('   Трассировка: %d/%d точек\n', i, num_points);
    end
end

valid_idx = find(valid_mask);
R_valid = R_points(:, valid_idx);
s_valid = s_vals(valid_idx);
fprintf('   Успешно протрассировано: %d из %d точек\n', length(valid_idx), num_points);

%% 4. Построение траектории R(z) и вызов recoverLayer
[s_sorted, idx_sort] = sort(s_valid);
R_sorted = R_valid(:, idx_sort);
R_traj = InverseTask.Trajectory(R_sorted);
z_R_max = R_traj.totalLength();
fprintf('\n6. Траектория R(z): длина = %.3f мм\n', z_R_max);

% Начальные условия
s_0 = 0;
u0_geo = lu_on_E3.getPoint(s_0); u0_geo = u0_geo(:);
tau0_geo = lu_on_E3.getTangent(s_0); tau0_geo = tau0_geo(:);
z0 = u0_geo(3);
v0 = atan2(u0_geo(2), u0_geo(1));
s0 = E3.s_from_z(z0);
[du0_s, dv0_s] = InverseTask.computeTangentCoeffs(E3, s0, v0, tau0_geo);

z_span = [0, z_R_max];
R_func = @(z) R_traj.getPoint(z);
Rprime_func = @(z) R_traj.getTangent(z);

fprintf('\n7. Запуск recoverLayer...\n');
try
    tic;
    [u_rec, v_rec, s_rec, z_rec] = InverseTask.recoverLayer(E3, R_func, Rprime_func, ...
        z_span, s0, v0, du0_s, dv0_s, 100, 0);
    toc;
    fprintf('   Восстановлено %d точек\n', length(u_rec));
catch ME
    fprintf('   ОШИБКА в recoverLayer: %s\n', ME.message);
    fprintf('   Используем заглушку (Euler метод)...\n');
    Nsteps = 1000;
    zeta_hist = linspace(0, z_R_max, Nsteps+1);
    h = zeta_hist(2) - zeta_hist(1);
    u_rec = zeros(1, Nsteps+1); v_rec = zeros(1, Nsteps+1);
    s_rec = zeros(1, Nsteps+1); z_rec = zeros(1, Nsteps+1);
    u = s0; v = v0; s = 0;
    u_rec(1) = u; v_rec(1) = v; s_rec(1) = s; z_rec(1) = 0;
    for i = 1:Nsteps
        zeta = zeta_hist(i);
        R = R_func(zeta); R = R(:);
        Rprime = Rprime_func(zeta);
        r = E3.getPoint(u, v); r = r(:);
        n = E3.getNormal(u, v); n = n(:);
        tau = (R - r) / norm(R - r);
        [du_s, dv_s] = InverseTask.computeTangentCoeffs(E3, u, v, tau);
        [L, M, N] = E3.getSecondFundamental(u, v);
        II = L*du_s^2 + 2*M*du_s*dv_s + N*dv_s^2;
        lambda = norm(R - r);
        if abs(II) < 1e-10 || lambda < 1e-10
            dsdz = 0;
        else
            dsdz = dot(Rprime, n) / (lambda * II);
        end
        u = u + h * du_s * dsdz;
        v = v + h * dv_s * dsdz;
        s = s + h * dsdz;
        u_rec(i+1) = u; v_rec(i+1) = v; s_rec(i+1) = s; z_rec(i+1) = zeta + h;
    end
    fprintf('   Заглушка: восстановлено %d точек\n', length(u_rec));
end

%% 5. Построение восстановленной ЛУ
N_rec = length(u_rec);
rec_pts = zeros(3, N_rec);
for i = 1:N_rec
    rec_pts(:, i) = E3.position(u_rec(i), v_rec(i));
end

%% 6. Вычисление погрешностей (через минимальное расстояние до облака точек)
fprintf('\n8. Вычисление погрешностей...\n');

% Плотные сетки для точного сравнения
s_orig_dense = linspace(0, lu_orig.totalLength(), 3000);
pts_orig = zeros(3, length(s_orig_dense));
for i = 1:length(s_orig_dense)
    pts_orig(:, i) = lu_orig.getPoint(s_orig_dense(i));
end

s_lu_dense = linspace(0, lu_on_E3.totalLength(), 3000);
pts_lu_E3 = zeros(3, length(s_lu_dense));
for i = 1:length(s_lu_dense)
    pts_lu_E3(:, i) = lu_on_E3.getPoint(s_lu_dense(i));
end

% Функция для вычисления минимального расстояния от точек A до облака точек B
calc_min_dist = @(A, B) arrayfun(@(i) min(sqrt(sum((B - A(:, i)).^2, 1))), 1:size(A, 2));

% Погрешность 1: lu_on_E3 vs исходная ЛУ
fprintf('   Вычисление погрешности lu_on_E3 vs исходная ЛУ...\n');
err_lu_vs_orig = calc_min_dist(pts_lu_E3, pts_orig);

% Погрешность 2: восстановленная vs lu_on_E3
fprintf('   Вычисление погрешности восстановленной vs lu_on_E3...\n');
err_rec_vs_lu = calc_min_dist(rec_pts, pts_lu_E3);

% Погрешность 3: восстановленная vs исходная ЛУ
fprintf('   Вычисление погрешности восстановленной vs исходная ЛУ...\n');
err_rec_vs_orig = calc_min_dist(rec_pts, pts_orig);

fprintf('\n=== СТАТИСТИКА ПОГРЕШНОСТЕЙ ===\n');
fprintf('lu_on_E3 vs исходная ЛУ:    средняя = %.3f мм, макс = %.3f мм\n', mean(err_lu_vs_orig), max(err_lu_vs_orig));
fprintf('Восстановленная vs lu_on_E3: средняя = %.3f мм, макс = %.3f мм\n', mean(err_rec_vs_lu), max(err_rec_vs_lu));
fprintf('Восстановленная vs исходная: средняя = %.3f мм, макс = %.3f мм\n', mean(err_rec_vs_orig), max(err_rec_vs_orig));

%% 7. Визуализация
fprintf('\n9. Построение визуализации...\n');

% --- 3D Сцена ---
figure('Name', 'Случай 3: 3D Сцена', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; view(35, 25);

% Каркас E3
z_vis = linspace(E3.z_min, E3.z_max, 40);
for v0_vis = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:, ii) = E3.position_by_z(z_vis(ii), v0_vis);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

% Поверхность E1 (полупрозрачная)
u_plot = linspace(E1.u_min, E1.u_max, 60);
v_plot = linspace(0, 2*pi, 40);
[Xs, Ys, Zs] = deal(zeros(length(u_plot), length(v_plot)));
for i = 1:length(u_plot)
    for j = 1:length(v_plot)
        p = E1.position(u_plot(i), v_plot(j));
        Xs(i,j) = p(1); Ys(i,j) = p(2); Zs(i,j) = p(3);
    end
end
surf(Xs, Ys, Zs, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [1 0.5 0.5], ...
    'DisplayName', 'E1 (аналитическая)');

% Исходная ЛУ (синяя)
plot3(pts_orig(1,:), pts_orig(2,:), pts_orig(3,:), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Исходная ЛУ (E2)');

% lu_on_E3 (зеленая)
plot3(pts_lu_E3(1,:), pts_lu_E3(2,:), pts_lu_E3(3,:), 'g-', 'LineWidth', 2, 'DisplayName', 'lu_on_E3 (сглаженная)');

% Восстановленная ЛУ (красная пунктирная)
plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'r--', 'LineWidth', 2.5, 'DisplayName', 'Восстановленная ЛУ');

% R(z) (серая)
R_plot = zeros(3, length(z_rec));
for i = 1:length(z_rec)
    R_plot(:, i) = R_func(z_rec(i));
end
plot3(R_plot(1,:), R_plot(2,:), R_plot(3,:), 'k-', 'LineWidth', 1, 'DisplayName', 'R(z)');

xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
title('Случай 3: Сглажены E2 и ЛУ');
legend('Location', 'bestoutside');
r_max_E3 = max(E3.radius(linspace(E3.z_min, E3.z_max, 100)));
xlim([-r_max_E3*1.3, r_max_E3*1.3]);
ylim([-r_max_E3*1.3, r_max_E3*1.3]);
zlim([E3.z_min-20, E3.z_max+20]);
daspect([1, 1, 1]);
hold off;

% --- Графики погрешностей ---
figure('Name', 'Случай 3: Погрешности', 'Color', 'w', 'Position', [150, 150, 1000, 800]);

subplot(3,1,1);
plot(s_lu_dense, err_lu_vs_orig, 'b-', 'LineWidth', 1.5);
ylabel('Ошибка, мм'); title('Погрешность: lu\_on\_E3 vs Исходная ЛУ (E2)');
grid on; yline(0, 'k--');

subplot(3,1,2);
plot(s_rec, err_rec_vs_lu, 'r-', 'LineWidth', 1.5);
ylabel('Ошибка, мм'); title('Погрешность: Восстановленная vs lu\_on\_E3');
grid on; yline(0, 'k--');

subplot(3,1,3);
plot(s_rec, err_rec_vs_orig, 'm-', 'LineWidth', 1.5);
ylabel('Ошибка, мм'); title('Погрешность: Восстановленная vs Исходная ЛУ (E2)');
xlabel('Длина дуги s, мм');
grid on; yline(0, 'k--');

fprintf('\n=== Случай 3 завершен ===\n');