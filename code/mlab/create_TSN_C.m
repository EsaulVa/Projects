function TSN = create_TSN_C(r, tau)
    %% 1. Подготовка данных
    x  = r(:,1);     y  = r(:,2);     z  = r(:,3);
    tx = tau(:,1);   ty = tau(:,2);   tz = tau(:,3);
    
    a_o = 0;    d_o = 768.54;
    a_p = 0;    d_p = 955.956;
    

    %% 2. Выравнивание геометрических центров
    fprintf('\n');
    z_offset = align_surface_centers(a_o, d_o, a_p, d_p);
    
    %% 3. Преобразование координат оправки в ГЛОБАЛЬНУЮ систему
    fprintf('\n--- Преобразование в глобальную систему ---\n');
    z_global = z + z_offset;
    fprintf('Z_локальный: [%.3f, %.3f]\n', min(z), max(z));
    fprintf('Z_глобальный: [%.3f, %.3f]\n', min(z_global), max(z_global));
    
    %% 4. Инициализация массивов результатов
    N = length(x);
    
    fprintf('Количество точек укладки: %d\n', N);
    X_tsn = zeros(N, 1);
    Y_tsn = zeros(N, 1);
    Z_tsn = zeros(N, 1);
    t_dist = zeros(N, 1);
    status = zeros(N, 1);
    
    %% 5. Параметры поиска корня
    % lyambda_min = 0;
    % lyambda_max = d_p;  % Увеличиваем максимум
    options = optimset('Display', 'off', 'TolX', 1e-10);
    
    %% 6. Основной цикл расчёта
    fprintf('\n--- Расчёт траектории ---\n');
    tic;
    
    for i = 1:N
        % Координаты точки в ГЛОБАЛЬНОЙ системе
        P0 = [x(i), y(i), z_global(i)];
        Tau = [tx(i), ty(i), tz(i)];
        
        % Поиск lyambda для i-й точки 
        t_start = z_offset * 2;  % = 187.4 мм
        fun = @(t) objective_func(t, P0, Tau);
        lyambda = fzero(fun, t_start, options);
        
        % Если нашли решение
        if ~isnan(lyambda)
            X_tsn(i) = P0(1) + lyambda * Tau(1);
            Y_tsn(i) = P0(2) + lyambda * Tau(2);
            Z_tsn(i) = P0(3) + lyambda * Tau(3);
            t_dist(i) = lyambda;
            status(i) = 1;
        else
            % Не удалось найти решение
            status(i) = 0;
            X_tsn(i) = NaN; Y_tsn(i) = NaN; Z_tsn(i) = NaN;
            t_dist(i) = NaN;
        end
        
        if mod(i, 100) == 0 || i == N
            fprintf('Обработано %d из %d точек (%.1f%%)\n', ...
                i, N, 100*i/N);
        end
    end
    
    elapsed_time = toc;
    fprintf('\nРасчёт завершён за %.2f сек.\n', elapsed_time);

    TSN = [X_tsn(:), Y_tsn(:), Z_tsn(:) - z_offset, tau(:,:)]; 
    
    %% 7. Статистика результатов
    valid_points = sum(status);
    fprintf('Успешно рассчитано: %d из %d точек (%.1f%%)\n', ...
        valid_points, N, 100*valid_points/N);
    
    if valid_points > 0
        fprintf('Средняя длина свободного участка: %.2f мм\n', ...
            mean(t_dist(status==1)));
        fprintf('Минимальная длина: %.2f мм\n', min(t_dist(status==1)));
        fprintf('Максимальная длина: %.2f мм\n', max(t_dist(status==1)));
        
        % Показываем номера неуспешных точек
        failed_points = find(status == 0);
        if ~isempty(failed_points)
            fprintf('Неуспешные точки: ');
            fprintf('%d ', failed_points(1:min(20, length(failed_points))));
            if length(failed_points) > 20
                fprintf('... и ещё %d', length(failed_points) - 20);
            end
            fprintf('\n');
        end
    end
    
    %% 8. Сохранение результатов
    save('winding_trajectory_result.mat', ...
        'X_tsn', 'Y_tsn', 'Z_tsn', 't_dist', 'status', 'z_offset');
    fprintf('\nРезультаты сохранены в winding_trajectory_result.mat\n');
    
    %% 9. Визуализация
    plot_results(x, y, z_global, X_tsn, Y_tsn, Z_tsn, status);
end

%% ------------------------------------------------------------------------
function f = objective_func(lyambda, T_lu, Tau)
    %% Шаг 1: Вычисляем координаты точки на луче:
    %         для точки T_lu (с линии укладки)
    %         на основании заданного lyambda
    %         считаем координаты полученного радиус-вектора
    %         (работа в ГЛОБАЛЬНОЙ системе координат)
    X = T_lu(1) + lyambda * Tau(1);
    Y = T_lu(2) + lyambda * Tau(2);
    Z = T_lu(3) + lyambda * Tau(3);

    %% Шаг 2: Вычисляем радиус луча от оси Z
    R_ray = sqrt(X^2 + Y^2);
    
    %% Шаг 3: Проверяем границы (защита от выхода за пределы)
    if Z < 0 || Z > 955.956
        f = 1e9;  % Очень большое значение (штраф)
        return;
    end
    
    %% Шаг 4: Ограничиваем Z для вызова surface_r_b
    Z_clamped = max(0, min(Z, 955.956));
    
    %% Шаг 5: Вычисляем радиус поверхности безопасности
    [r_vec, ~, ~, ~, ~, ~] = surface_r_b(Z_clamped, 0); 
    R_surf = sqrt(r_vec(1)^2 + r_vec(2)^2);
        
    %% Шаг 6: Проверяем на ошибки
    if isnan(R_surf) || isnan(R_ray) || R_surf == 0
        f = 1e9;   % При наличии ошибеи штрафуем
        return;
    end
    
    %% Шаг 7: Возвращаем невязку (разницу радиусов)
    f = R_ray - R_surf;
end

%% ------------------------------------------------------------------------
function plot_results(x, y, z, Xe, Ye, Ze, status)
    figure('Color', 'w', 'Position', [100, 100, 1200, 800]);
    hold on; grid on; axis equal;
    view(45, 20);
    
    % Линия укладки на оправке
    h_o = plot3(x, y, z, 'b-', 'LineWidth', 1);
    scatter3(x, y, z, 15, 'b', 'filled', 'MarkerFaceAlpha', 0.5);
    
    % Траектория глаза (только успешные точки) - точки ТСН
    valid_idx = (status == 1);
    h_p = plot3(Xe(valid_idx), Ye(valid_idx), Ze(valid_idx), 'r-');
    scatter3(Xe(valid_idx), Ye(valid_idx), Ze(valid_idx), ...
        30, 'r', 'filled', 'MarkerFaceAlpha', 0.7);
    
    % Линии схода
    for k = 1:1:length(x)
        if status(k) == 1 && ~isnan(Xe(k))
            plot3([x(k), Xe(k)], [y(k), Ye(k)], [z(k), Ze(k)],'g-');
        end
    end

    % Оси координат
    plot3([0, 0], [0, 0], [-100, 1100], 'k-');
    
    % Поверхности
    vivod_3D_ballon_PB(0, 768.54, @surface_r, 955.956, @surface_r_b, 500);
    
    xlabel('X, мм', 'FontSize', 12);
    ylabel('Y, мм', 'FontSize', 12);
    zlabel('Z, мм', 'FontSize', 12);
    title('Траектория схода нити', 'FontSize', 14);
    legend([h_o, h_p], {'ЛУ на оправке', 'ТСН на поверхности безопасности)'}, ...
           'Location', 'best', 'FontSize', 14);
    
    % Информация
    annotation('textbox', [0.15, 0.85, 0.2, 0.1], ...
        'String', sprintf('Точек: %d\nУспешно: %d\nПроцент: %.1f%%', ...
            length(x), sum(status), 100*sum(status)/length(x)), ...
        'FitBoxToText', 'on', 'FontSize', 14);
    
    hold off;
end

%% ------------------------------------------------------------------------
function z_offset = align_surface_centers(a_m, d_m, a_s, d_s)
 
    z_offset = (d_s - d_m) / 2;
    
    fprintf('=== ВЫРАВНИВАНИЕ ПОВЕРХНОСТЕЙ ===\n');
    fprintf('Оправка (локальная):     Z ∈ [%.3f, %.3f]\n', a_m, d_m);
    fprintf('Безопасность (глобальная): Z ∈ [%.3f, %.3f]\n', a_s, d_s);
    fprintf('z_offset = +%.3f мм\n', z_offset);
    fprintf('Оправка в глобальной системе: Z ∈ [%.3f, %.3f]\n', ...
        a_m + z_offset, d_m + z_offset);
end