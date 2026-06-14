% main_inverse_task_2.m
% Обратная задача: восстановление линии укладки на сглаженной оправке E3
% по траектории точки схода R(z) на поверхности безопасности E1.
clear; clc; close all;

%% 1. Загрузка данных
load('E3_smoothed.mat', 'E3');
fprintf('1. E3: z ? [%.3f, %.3f] мм\n', E3.z_min, E3.z_max);

load('lu_on_E3.mat', 'lu_on_E3');
fprintf('2. lu_on_E3: длина дуги = %.3f мм\n', lu_on_E3.totalLength());

load('R_trajectory_on_E1.mat', 'R_valid', 's_valid');
fprintf('3. Загружено %d точек R(z)\n', size(R_valid, 2));

%% 2. Построение траектории R(z)
[s_sorted, idx] = sort(s_valid);
R_sorted = R_valid(:, idx);
R_traj = InverseTask.Trajectory(R_sorted);
z_R_max = R_traj.totalLength();
fprintf('4. Траектория R(z): длина = %.3f мм\n', z_R_max);

%% 3. Начальные условия
fprintf('\n5. Установка начальных условий...\n');

zeta_start = 0;
zeta_end = z_R_max;
z_span = [zeta_start, zeta_end];
fprintf('   Диапазон интегрирования (по длине дуги R): zeta ? [%.3f, %.3f] мм\n', zeta_start, zeta_end);

R_start = R_traj.getPoint(zeta_start);
fprintf('   Начало R: точка = [%.3f, %.3f, %.3f]\n', R_start);

s_0 = 0;
u0_geo = lu_on_E3.getPoint(s_0);
u0_geo = u0_geo(:);
tau0_geo = lu_on_E3.getTangent(s_0);
tau0_geo = tau0_geo(:);
fprintf('   Начало ЛУ на E3: точка = [%.3f, %.3f, %.3f]\n', u0_geo);

z0 = u0_geo(3);
v0 = atan2(u0_geo(2), u0_geo(1));
s0 = E3.s_from_z(z0);
fprintf('   Параметры на E3: s0 = %.3f, v0 = %.3f, z_height = %.3f\n', s0, v0, z0);

[du0_s, dv0_s] = InverseTask.computeTangentCoeffs(E3, s0, v0, tau0_geo);
fprintf('   Начальные скорости: du/ds = %.6e, dv/ds = %.6e\n', du0_s, dv0_s);

DeltaZ = 100;
Percentage = 0;

%% 4. Вызов recoverLayer с обработкой ошибок
fprintf('\n6. Запуск recoverLayer...\n');
R_func = @(z) R_traj.getPoint(z);
Rprime_func = @(z) R_traj.getTangent(z);

try
    tic;
    [u_rec, v_rec, s_rec, z_rec] = InverseTask.recoverLayer(E3, R_func, Rprime_func, ...
        z_span, s0, v0, du0_s, dv0_s, DeltaZ, Percentage);
    toc;
    fprintf('   Восстановлено %d точек\n', length(u_rec));
catch ME
    fprintf('   ОШИБКА в recoverLayer: %s\n', ME.message);
    fprintf('   Попробуйте использовать DAE предиктор-корректор вместо RK4.\n');
    
    % Заглушка: используем простой метод Эйлера для демонстрации
    fprintf('\n   Используем заглушку (Euler метод)...\n');
    Nsteps = 1000;
    zeta_hist = linspace(zeta_start, zeta_end, Nsteps+1);
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
        r = E3.position(u, v); r = r(:);
        n = E3.normal(u, v); n = n(:);
        
        tau = (R - r) / norm(R - r);
        [du_s, dv_s] = InverseTask.computeTangentCoeffs(E3, u, v, tau);
        
        % Формула (9): ds/dz = <R', n> / (lambda * kappa_n)
        [L, M, N] = E3.second_fundamental_form(u, v);
        II = L*du_s^2 + 2*M*du_s*dv_s + N*dv_s^2;
        lambda = norm(R - r);
        
        if abs(II) < 1e-10 || lambda < 1e-10
            dsdz = 0;
            warning('Сингулярность на шаге %d: II=%.2e, lambda=%.2e', i, II, lambda);
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

%% 5. Построение восстановленной ЛУ
N_rec = length(u_rec);
rec_pts = zeros(3, N_rec);
for i = 1:N_rec
    rec_pts(:, i) = E3.position(u_rec(i), v_rec(i));
end
rec_traj = InverseTask.Trajectory(rec_pts);
fprintf('7. Восстановленная ЛУ: длина дуги = %.3f мм\n', rec_traj.totalLength());

%% 6. Сравнение с исходной ЛУ
fprintf('\n8. Сравнение с исходной ЛУ...\n');
s_orig = linspace(0, lu_on_E3.totalLength(), 3000);
pts_orig = zeros(3, length(s_orig));
for i = 1:length(s_orig)
    pts_orig(:, i) = lu_on_E3.getPoint(s_orig(i));
end
orig_traj = InverseTask.Trajectory(pts_orig);

orig_pts_at_rec = zeros(3, N_rec);
for i = 1:N_rec
    s_interp = min(s_rec(i), lu_on_E3.totalLength());
    s_interp = max(s_interp, 0);
    orig_pts_at_rec(:, i) = orig_traj.getPoint(s_interp);
end

err_xyz = sqrt(sum((rec_pts - orig_pts_at_rec).^2, 1));
fprintf('   Средняя ошибка: %.2e мм\n', mean(err_xyz));
fprintf('   Максимальная ошибка: %.2e мм\n', max(err_xyz));
fprintf('   Медианная ошибка: %.2e мм\n', median(err_xyz));

%% 7. Визуализация
fprintf('\n9. Построение визуализации...\n');

figure('Name', 'Обратная задача', 'Color', 'w', 'Position', [100, 100, 1200, 800]);
hold on; grid on; view(35, 25);

z_vis = linspace(E3.z_min, E3.z_max, 40);
for v0_vis = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:, ii) = E3.position_by_z(z_vis(ii), v0_vis);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end

s_plot = linspace(0, lu_on_E3.totalLength(), 500);
pts_orig_plot = zeros(3, length(s_plot));
for i = 1:length(s_plot)
    pts_orig_plot(:, i) = lu_on_E3.getPoint(s_plot(i));
end
plot3(pts_orig_plot(1,:), pts_orig_plot(2,:), pts_orig_plot(3,:), 'b-', 'LineWidth', 1, ...
    'DisplayName', 'Исходная ЛУ');

plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'r--', 'LineWidth', 3, ...
    'DisplayName', 'Восстановленная ЛУ');

R_plot = zeros(3, length(z_rec));
for i = 1:length(z_rec)
    R_plot(:, i) = R_func(z_rec(i));
end
plot3(R_plot(1,:), R_plot(2,:), R_plot(3,:), 'g-', 'LineWidth', 1.5, ...
    'DisplayName', 'R(z)');
% scatter3(R_plot(1,:), R_plot(2,:), R_plot(3,:), 15, 'g', 'filled', ...
%     'DisplayName', 'R(z)');

plot3(u0_geo(1), u0_geo(2), u0_geo(3), 'mo', 'MarkerSize', 12, 'MarkerFaceColor', 'm', ...
    'DisplayName', 'Начальная точка');

xlabel('X, мм', 'FontWeight', 'bold'); 
ylabel('Y, мм', 'FontWeight', 'bold'); 
zlabel('Z, мм', 'FontWeight', 'bold');
title('Обратная задача: восстановление линии укладки', 'FontWeight', 'bold');
legend('Location', 'bestoutside');

r_max_E3 = max(E3.radius(linspace(E3.z_min, E3.z_max, 100)));
xlim([-r_max_E3*1.3, r_max_E3*1.3]);
ylim([-r_max_E3*1.3, r_max_E3*1.3]);
zlim([E3.z_min-20, E3.z_max+20]);
daspect([1, 1, 1]);
hold off;

figure('Name', 'Ошибка восстановления', 'Color', 'w', 'Position', [150, 150, 900, 500]);
semilogy(s_rec, err_xyz, 'r-', 'LineWidth', 1.5);
xlabel('Длина дуги s, мм', 'FontWeight', 'bold'); 
ylabel('Евклидова ошибка, мм', 'FontWeight', 'bold');
title('Ошибка восстановления линии укладки', 'FontWeight', 'bold');
grid on;
yline(1e-6, 'g--', 'LineWidth', 1, 'Label', '1 мкм');
yline(1e-3, 'b--', 'LineWidth', 1, 'Label', '1 мм');

fprintf('\n? Готово.\n');
save('inverse_task_results.mat', 'rec_pts', 'err_xyz', 'rec_traj', 'u_rec', 'v_rec', 's_rec', 'z_rec');