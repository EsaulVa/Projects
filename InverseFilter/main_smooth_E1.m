% main_smooth_E1.m
% Сглаживание поверхности безопасности E1 (RevolutionSurface) по Лапласу–Бельтрами.
% Сохраняет сглаженную поверхность в E1_smoothed.mat для последующей трассировки.

clear; clc; close all;

%% 1. Параметры сглаживания (можно подобрать отдельно для E1)
FILTER_N       = 800;      % число точек дискретизации меридиана
FILTER_TAU     = 1.0;      % шаг диффузии (мм^2)
FILTER_NSTEPS  = 20;        % число шагов => T = tau * nsteps
PRESERVE_Z     = true;     % сохранять параметризацию по z

%% 2. Создание исходной поверхности безопасности E1 (как в main_shadow_trace.m)
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;
E1_raw = InverseTask.RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);
fprintf('Исходная E1: z ∈ [%.3f, %.3f] мм\n', E1_raw.u_min, E1_raw.u_max);

%% 3. Извлечение меридиана (z, r) из E1_raw
N_meridian = 1000;  % точек для извлечения
u_vals = linspace(E1_raw.u_min, E1_raw.u_max, N_meridian)';
z_raw = u_vals;
r_raw = zeros(size(z_raw));
for i = 1:N_meridian
    pt = E1_raw.position(u_vals(i), 0.0);
    r_raw(i) = sqrt(pt(1)^2 + pt(2)^2);
end

% Сохраним исходный меридиан в CSV для контроля
T_raw = table(z_raw, r_raw, 'VariableNames', {'z', 'r'});
writetable(T_raw, 'meridian_E1_raw_rz.csv');
fprintf('Исходный меридиан E1 сохранён в meridian_E1_raw_rz.csv\n');

%% 4. Создание дискретной поверхности вращения из меридиана
E1_disc = Filter.DiscreteRevolutionSurface(z_raw, r_raw);
fprintf('Дискретная поверхность E1 создана: u ∈ [%.3f, %.3f] мм\n', E1_disc.u_min, E1_disc.u_max);

%% 5. Диффузионное сглаживание -> E1_smooth
fprintf('\nЗапуск диффузии E1: tau=%.1f, nsteps=%d, N=%d, T=%.1f мм^2\n', ...
    FILTER_TAU, FILTER_NSTEPS, FILTER_N, FILTER_TAU*FILTER_NSTEPS);

E1_smooth = Filter.DiffusedRevolutionSurface(E1_disc, FILTER_N, FILTER_TAU, FILTER_NSTEPS, ...
    Filter.BoundaryCondition.dirichlet(0.0), ...   % левый конец фиксирован
    Filter.BoundaryCondition.dirichlet(0.0), ...   % правый конец фиксирован
    'PreserveZParameter', PRESERVE_Z, ...
    'SaveMeridianPath', 'meridian_E1_smooth_s.csv');

fprintf('Сглаженная E1: z ∈ [%.3f, %.3f] мм, s ∈ [%.3f, %.3f] мм\n', ...
    E1_smooth.z_min, E1_smooth.z_max, E1_smooth.u_min, E1_smooth.u_max);

%% 6. Сохранение сглаженной поверхности
save('E1_smoothed.mat', 'E1_smooth');
fprintf('Сглаженная поверхность E1 сохранена в E1_smoothed.mat\n');

%% 7. Визуализация профиля и кривизны (аналог client_visualize_diffusion_matlab)
N_PLOT = 2000;
z_plot = linspace(E1_raw.u_min, E1_raw.u_max, N_PLOT)';
r_plot_raw = zeros(size(z_plot));
for i = 1:N_PLOT
    pt = E1_raw.position(z_plot(i), 0.0);
    r_plot_raw(i) = sqrt(pt(1)^2 + pt(2)^2);
end

z_smooth_plot = linspace(E1_smooth.z_min, E1_smooth.z_max, N_PLOT)';
r_smooth_plot = arrayfun(@(z) E1_smooth.radius(z), z_smooth_plot);

% Кривизна исходной (численно)
dr_raw = gradient(r_plot_raw, z_plot);
d2r_raw = gradient(dr_raw, z_plot);
kappa_raw = abs(d2r_raw) ./ (1 + dr_raw.^2).^1.5;

% Кривизна сглаженной (через s)
s_curv = linspace(E1_smooth.u_min, E1_smooth.u_max, N_PLOT)';
pts_curv = arrayfun(@(s) E1_smooth.position(s, 0.0), s_curv, 'UniformOutput', false);
pts_curv = cell2mat(pts_curv);
r_s = sqrt(pts_curv(:,1).^2 + pts_curv(:,2).^2);
z_s = pts_curv(:,3);
dr_dz = gradient(r_s, z_s);
d2r_dz2 = gradient(dr_dz, z_s);
kappa_smooth = abs(d2r_dz2) ./ (1 + dr_dz.^2).^1.5;

figure('Name', 'Профиль E1', 'Position', [100 100 1200 500]);
subplot(1,2,1);
plot(z_plot, r_plot_raw, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Исходная E1');
hold on;
plot(z_smooth_plot, r_smooth_plot, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Сглаженная E1');
xlabel('Z, мм'); ylabel('R, мм'); title('Профиль меридиана E1');
legend; grid on; axis tight;

subplot(1,2,2);
delta_r = interp1(z_smooth_plot, r_smooth_plot, z_plot, 'linear') - r_plot_raw;
plot(z_plot, delta_r*1000, 'g-', 'LineWidth', 1.2);
xlabel('Z, мм'); ylabel('\Delta R, мкм'); title('Отклонение (сглаж. - исх.)');
grid on;
saveas(gcf, 'E1_diffusion_profile.png');

figure('Name', 'Кривизна E1', 'Position', [100 650 900 400]);
semilogy(z_plot, kappa_raw+1e-12, 'b-', 'LineWidth', 1.2, 'DisplayName', 'Исходная');
hold on;
semilogy(z_s, kappa_smooth+1e-12, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Сглаженная');
xlabel('Z, мм'); ylabel('|\kappa|, 1/мм'); title('Кривизна меридиана E1');
legend; grid on;
saveas(gcf, 'E1_diffusion_curvature.png');

fprintf('Графики сохранены: E1_diffusion_profile.png, E1_diffusion_curvature.png\n');
fprintf('Готово.\n');