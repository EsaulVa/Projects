function client_visualize_diffusion_matlab()
%CLIENT_VISUALIZE_DIFFUSION_MATLAB Аналог client_visualize_diffusion_1.py
%   Читает meridian_E2_raw_rz.csv, строит дискретную поверхность вращения,
%   пропускает через диффузионный фильтр с фиксацией концов (Дирихле),
%   визуализирует профиль и кривизну меридиана, сохраняет CSV и PNG.
%
%   Выходные файлы:
%     diffusion_meridian_profile.png  — профиль r(z) и отклонение
%     diffusion_curvature.png         — кривизна (логарифмическая шкала)
%     meridian_E2_smooth_rz.csv      — сглаженный меридиан в координатах (z, r)
%     diffusion_surface_3d.fig        — 3D-сцена (опционально)
%
%   Требования:
%     - DiscreteRevolutionSurface.m
%     - DiffusedRevolutionSurface.m
%     - BoundaryCondition.m
%     - meridian_E2_raw_rz.csv (колонки: z, r)

    %% 1. Параметры фильтра (подбирайте по графикам)
    FILTER_N       = 800;   % точек дискретизации меридиана
    FILTER_TAU     = 2.0;   % шаг диффузии, мм^2
    FILTER_NSTEPS  = 5;     % число шагов => полное время t = tau * n_steps
    N_PLOT         = 2000;  % точек для графиков
    N_3D           = 80;    % размер сетки для 3D surf

    fprintf('============================================================\n');
    fprintf('Клиент диффузионной фильтрации (MATLAB)\n');
    fprintf('============================================================\n\n');

    %% 2. Чтение CSV
    csv_name = 'meridian_E2_raw_rz.csv';
    if ~exist(csv_name, 'file')
        error('Файл %s не найден! Сначала создайте его (client_meridian_export.py).', csv_name);
    end
    data = readtable(csv_name);
    z_raw = data.z;
    r_raw = data.r;
    fprintf('Загружено %d точек меридиана, z ? [%.3f, %.3f] мм\n', ...
        length(z_raw), min(z_raw), max(z_raw));

    %% 3. Исходная дискретная поверхность вращения
    E2_raw = DiscreteRevolutionSurface(z_raw, r_raw);
    fprintf('DiscreteRevolutionSurface создана: u ? [%.3f, %.3f] мм\n', ...
        E2_raw.u_min, E2_raw.u_max);

    %% 4. Диффузия с фиксацией концов и сохранением z-параметризации
    fprintf('\nЗапуск диффузии: tau=%.1f, n_steps=%d, N=%d\n', ...
        FILTER_TAU, FILTER_NSTEPS, FILTER_N);
    fprintf('Полное время t = %.1f мм^2\n', FILTER_TAU * FILTER_NSTEPS);

    E2_smooth = DiffusedRevolutionSurface(E2_raw, FILTER_N, FILTER_TAU, FILTER_NSTEPS, ...
        BoundaryCondition.dirichlet(0.0), ...   % левый конец: фиксация
        BoundaryCondition.dirichlet(0.0), ...   % правый конец: фиксация
        'PreserveZParameter', true, ...
        'SaveMeridianPath', 'meridian_E2_diffused_s.csv');

    fprintf('DiffusedRevolutionSurface создана: s ? [%.3f, %.3f] мм\n', ...
        E2_smooth.u_min, E2_smooth.u_max);
    fprintf('z-параметризация: z ? [%.3f, %.3f] мм\n', ...
        E2_smooth.z_min, E2_smooth.z_max);

    %% 5. Демонстрация доступа по z
    fprintf('\n--- Демонстрация preserve_z_parameter ---\n');
    z_demo = linspace(E2_raw.u_min, E2_raw.u_max, 10)';
    for i = 1:length(z_demo)
        z_val = z_demo(i);
        r_val = E2_smooth.radius(z_val);
        s_val = E2_smooth.s_from_z(z_val);
        pt    = E2_smooth.position_by_z(z_val, 0.0);
        fprintf('  z = %7.2f мм  ->  r = %8.4f, s = %8.4f, pos = (%8.3f,%8.3f,%8.3f)\n', ...
            z_val, r_val, s_val, pt(1), pt(2), pt(3));
    end

    %% 6. Подготовка данных для графиков
    % --- Оригинал на равномерной сетке z ---
    z_plot = linspace(E2_raw.u_min, E2_raw.u_max, N_PLOT)';
    r_plot_raw = arrayfun(@(z) E2_raw.radius(z), z_plot);

    % --- Сглаженный на равномерной сетке z ---
    z_plot_smooth = linspace(E2_smooth.z_min, E2_smooth.z_max, N_PLOT)';
    r_plot_smooth = arrayfun(@(z) E2_smooth.radius(z), z_plot_smooth);

    % --- Кривизна оригинала (численная, может быть шумной) ---
    dr_raw  = gradient(r_plot_raw) ./ gradient(z_plot);
    d2r_raw = gradient(dr_raw) ./ gradient(z_plot);
    kappa_raw = abs(d2r_raw) ./ (1.0 + dr_raw.^2).^1.5;

    % --- Кривизна сглаженного (параметрическая через s, гладкая) ---
    s_curv = linspace(E2_smooth.u_min, E2_smooth.u_max, N_PLOT)';
    pts_curv = arrayfun(@(s) E2_smooth.position(s, 0.0), s_curv, 'UniformOutput', false);
    pts_curv = cell2mat(pts_curv);  % N x 3
    r_s = sqrt(pts_curv(:,1).^2 + pts_curv(:,2).^2);
    z_s = pts_curv(:,3);
    dr_dz = gradient(r_s) ./ gradient(z_s);
    d2r_dz2 = gradient(dr_dz) ./ gradient(z_s);
    kappa_smooth = abs(d2r_dz2) ./ (1.0 + dr_dz.^2).^1.5;

    % --- Разница радиусов (интерполяция на общую сетку) ---
    z_common = linspace(max(z_plot(1), z_plot_smooth(1)), ...
                        min(z_plot(end), z_plot_smooth(end)), N_PLOT)';
    r_raw_i = interp1(z_plot, r_plot_raw, z_common, 'linear', 'extrap');
    r_smooth_i = interp1(z_plot_smooth, r_plot_smooth, z_common, 'linear', 'extrap');
    delta_r = r_smooth_i - r_raw_i;

    %% 7. График 1: Профиль меридиана и отклонение
    figure('Name', 'Профиль меридиана', 'Position', [100 100 1200 500]);
    % Подграфик A: r(z)
    subplot(1, 2, 1);
    plot(z_plot, r_plot_raw, 'b-', 'LineWidth', 1.5, 'DisplayName', 'E2_{raw}  r(z)');
    hold on;
    plot(z_plot_smooth, r_plot_smooth, 'r-', 'LineWidth', 1.5, 'DisplayName', ...
        sprintf('E2_{smooth}  r(z)  (\\tau=%.1f, n=%d)', FILTER_TAU, FILTER_NSTEPS));
    hold off;
    xlabel('Z, мм'); ylabel('R, мм');
    title('Профиль меридиана (r vs z)');
    legend('Location', 'best'); grid on; axis tight;

    % Подграфик B: отклонение
    subplot(1, 2, 2);
    plot(z_common, delta_r * 1000, 'g-', 'LineWidth', 1.2);  % в микрометрах
    hold on;
    plot([z_common(1), z_common(end)], [0, 0], 'k--', 'LineWidth', 0.5);
    hold off;
    xlabel('Z, мм'); ylabel('\Delta R, мкм');
    title('Разница радиусов (сглаженный - оригинал)');
    grid on; axis tight;

    saveas(gcf, 'diffusion_meridian_profile.png');
    fprintf('\n[OK] График сохранён: diffusion_meridian_profile.png\n');

    %% 8. График 2: Кривизна (логарифмическая шкала)
    figure('Name', 'Кривизна меридиана', 'Position', [100 650 900 400]);
    semilogy(z_plot, kappa_raw + 1e-12, 'b-', 'LineWidth', 1.2, 'DisplayName', 'E2_{raw}  \kappa(z)');
    hold on;
    semilogy(z_s, kappa_smooth + 1e-12, 'r-', 'LineWidth', 1.2, 'DisplayName', 'E2_{smooth}  \kappa(z)');
    hold off;
    xlabel('Z, мм'); ylabel('|\kappa|, 1/мм');
    title('Кривизна профиля меридиана (параметрический расчёт)');
    legend('Location', 'best'); grid on; axis tight;

    saveas(gcf, 'diffusion_curvature.png');
    fprintf('[OK] График сохранён: diffusion_curvature.png\n');

    %% 9. Экспорт сглаженного меридиана r(z) в CSV
    df_out = table(z_plot_smooth, r_plot_smooth, ...
                   'VariableNames', {'z', 'r'});
    writetable(df_out, 'meridian_E2_smooth_rz.csv');
    fprintf('[OK] CSV сохранён: meridian_E2_smooth_rz.csv\n');

    %% 10. 3D-визуализация (surf) — опционально
    fprintf('\nПостроение 3D-сцены...\n');
    figure('Name', '3D: Диффузионная фильтрация', 'Position', [150 150 900 700]);

    % --- Сглаженная поверхность ---
    s_surf = linspace(E2_smooth.u_min, E2_smooth.u_max, N_3D);
    v_surf = linspace(0, 2*pi, N_3D);
    [SS, VV] = meshgrid(s_surf, v_surf);

    X_smooth = zeros(size(SS));
    Y_smooth = zeros(size(SS));
    Z_smooth = zeros(size(SS));
    for i = 1:size(SS, 1)
        for j = 1:size(SS, 2)
            pt = E2_smooth.position(SS(i,j), VV(i,j));
            X_smooth(i,j) = pt(1);
            Y_smooth(i,j) = pt(2);
            Z_smooth(i,j) = pt(3);
        end
    end

    surf(X_smooth, Y_smooth, Z_smooth, 'FaceColor', 'r', 'EdgeColor', 'none', ...
         'FaceAlpha', 0.6, 'DisplayName', 'Сглаженный');
    hold on;

    % --- Оригинальная поверхность (менее прозрачная) ---
    u_surf = linspace(E2_raw.u_min, E2_raw.u_max, N_3D);
    [UU, VV2] = meshgrid(u_surf, v_surf);
    X_orig = zeros(size(UU));
    Y_orig = zeros(size(UU));
    Z_orig = zeros(size(UU));
    for i = 1:size(UU, 1)
        for j = 1:size(UU, 2)
            pt = E2_raw.position(UU(i,j), VV2(i,j));
            X_orig(i,j) = pt(1);
            Y_orig(i,j) = pt(2);
            Z_orig(i,j) = pt(3);
        end
    end

    surf(X_orig, Y_orig, Z_orig, 'FaceColor', 'b', 'EdgeColor', 'none', ...
         'FaceAlpha', 0.3, 'DisplayName', 'Оригинал');

    % --- Меридианы (линии) ---
    u_mer = linspace(E2_raw.u_min, E2_raw.u_max, 200)';
    pts_mer_orig = zeros(200, 3);
    pts_mer_smooth = zeros(200, 3);
    for i = 1:200
        pts_mer_orig(i,:) = E2_raw.position(u_mer(i), 0.0);
        s_mer = E2_smooth.s_from_z(u_mer(i));  % примерно
        pts_mer_smooth(i,:) = E2_smooth.position(s_mer, 0.0);
    end
    plot3(pts_mer_orig(:,1), pts_mer_orig(:,2), pts_mer_orig(:,3), ...
          'b-', 'LineWidth', 3, 'DisplayName', 'Меридиан оригинал');
    plot3(pts_mer_smooth(:,1), pts_mer_smooth(:,2), pts_mer_smooth(:,3), ...
          'r-', 'LineWidth', 3, 'DisplayName', 'Меридиан сглаженный');

    hold off;
    axis equal; grid on;
    xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
    title(sprintf('Диффузионная фильтрация (\\tau=%.1f, n=%d)', FILTER_TAU, FILTER_NSTEPS));
    legend('Location', 'best');
    view(45, 25);

    saveas(gcf, 'diffusion_surface_3d.fig');
    fprintf('[OK] 3D-сцена сохранена: diffusion_surface_3d.fig\n');

    %% Завершение
    fprintf('\n============================================================\n');
    fprintf('ГОТОВО\n');
    fprintf('============================================================\n');
end

