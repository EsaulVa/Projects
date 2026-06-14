% main_inverse_task_case2_smooth_E1.m
% Случай 2: Сглажена только поверхность безопасности E1
% E2 и ЛУ - исходные (без сглаживания)
clear; clc; close all;

fprintf('=== СЛУЧАЙ 2: СГЛАЖЕНА ТОЛЬКО E1 ===\n');
fprintf('E1: сглаженная DiffusedRevolutionSurface\n');
fprintf('E2: дискретная DiscreteRevolutionSurface (исходная)\n');
fprintf('ЛУ: исходная из LU_data.mat\n\n');

%% 1. Загрузка исходной оправки E2 (без сглаживания)
csv_name = 'meridian_E2_raw_rz.csv';
data = readtable(csv_name);
E2 = Filter.DiscreteRevolutionSurface(data.z, data.r);
fprintf('1. Загружена исходная оправка E2: z ? [%.3f, %.3f] мм\n', E2.u_min, E2.u_max);

%% 2. Загрузка исходной ЛУ
load('LU_data.mat', 'r');
if size(r,1) ~= 3
    r = r';
end
lu_original = InverseTask.Trajectory(r);
fprintf('2. Загружена исходная ЛУ: длина дуги = %.3f мм\n', lu_original.totalLength());

%% 3. Загрузка сглаженной E1
if ~exist('E1_smoothed.mat', 'file')
    error('Файл E1_smoothed.mat не найден. Сначала выполните main_smooth_E1.m');
end
load('E1_smoothed.mat', 'E1_smooth');
fprintf('3. Загружена сглаженная E1: z ? [%.3f, %.3f] мм\n', ...
    E1_smooth.z_min, E1_smooth.z_max);

%% 4. Трассировка на сглаженную E1
fprintf('\n4. Трассировка лучей от ЛУ к сглаженной E1...\n');
num_points = 2500;
s_vals = linspace(0, lu_original.totalLength(), num_points);
R_points = zeros(3, num_points);
valid_mask = false(num_points, 1);

for i = 1:num_points
    s = s_vals(i);
    r_point = lu_original.getPoint(s);
    r_point = r_point(:);
    tau_lu = lu_original.getTangent(s);
    tau_lu = tau_lu(:);
    
    % Нормаль к E2
    z_coord = r_point(3);
    v_coord = atan2(r_point(2), r_point(1));
    n = E2.getNormal(z_coord, v_coord);
    
    % Проекция касательной
    tau_proj = tau_lu - dot(tau_lu, n) * n;
    tau_proj = tau_proj / norm(tau_proj);
    
    % Трассировка на E1_smooth (численный поиск)
    t_min = 1.0;
    t_max = 3000.0;
    
    t_found = NaN;
    for t = linspace(t_min, t_max, 1000)
        P = r_point + t * tau_proj;
        rho = sqrt(P(1)^2 + P(2)^2);
        z_P = P(3);
        if z_P >= E1_smooth.z_min && z_P <= E1_smooth.z_max
            R_E1 = E1_smooth.radius(z_P);
            if abs(rho - R_E1) < 0.1
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

%% 5. Построение траектории R(z)
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

z0 = u0_geo(3);
v0 = atan2(u0_geo(2), u0_geo(1));
s0 = z0;  % Для DiscreteRevolutionSurface параметр u = z

[du0_s, dv0_s] = InverseTask.computeTangentCoeffs(E2, s0, v0, tau0_geo);
fprintf('   Начальные условия: z0=%.3f, v0=%.3f\n', z0, v0);

DeltaZ = 100;
Percentage = 0;
z_span = [0, z_R_max];

%% 7. Вызов recoverLayer
fprintf('\n7. Запуск recoverLayer (обратная задача)...\n');
R_func = @(z) R_traj.getPoint(z);
Rprime_func = @(z) R_traj.getTangent(z);

try
    tic;
    [u_rec, v_rec, s_rec, z_rec] = InverseTask.recoverLayer(E2, R_func, Rprime_func, ...
        z_span, s0, v0, du0_s, dv0_s, DeltaZ, Percentage);
    toc;
    fprintf('   Восстановлено %d точек\n', length(u_rec));
catch ME
    fprintf('   ОШИБКА в recoverLayer: %s\n', ME.message);
    
    % Заглушка - метод Эйлера
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

%% 9. Вычисление погрешности
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
figure('Name', 'Случай 2: Сглажена E1', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; view(35, 25);

% Каркас E2 (исходная)
z_vis = linspace(E2.u_min, E2.u_max, 40);
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:,ii) = E2.position(z_vis(ii), v0);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5);
end

% Исходная ЛУ
s_plot = linspace(0, lu_original.totalLength(), 500);
pts_lu = zeros(3, length(s_plot));
for i = 1:length(s_plot)
    pts_lu(:,i) = lu_original.getPoint(s_plot(i));
end
plot3(pts_lu(1,:), pts_lu(2,:), pts_lu(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Исходная ЛУ');

% Восстановленная ЛУ
plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Восстановленная ЛУ');

title('Случай 2: Сглажена только E1 (поверхность безопасности)');
legend('Location', 'best');
hold off;

figure('Name', 'Ошибка восстановления (Случай 2)', 'Color', 'w', 'Position', [150, 150, 900, 500]);
semilogy(s_rec, err_xyz, 'r-', 'LineWidth', 1.5);
xlabel('Длина дуги s, мм', 'FontWeight', 'bold'); 
ylabel('Евклидова ошибка, мм', 'FontWeight', 'bold');
title('Ошибка восстановления (Случай 2: сглажена E1)', 'FontWeight', 'bold');
grid on;
yline(1e-6, 'g--', 'LineWidth', 1, 'Label', '1 мкм');
yline(1e-3, 'b--', 'LineWidth', 1, 'Label', '1 мм');

fprintf('\n=== Случай 2 завершен ===\n');
save('case2_results.mat', 'rec_pts', 'err_xyz', 'rec_traj', 'u_rec', 'v_rec', 's_rec', 'z_rec');