function J_val = run_spline_optimization(center_coords, x_glob, y_glob, bc_spec, alpha)
%RUN_SPLINE_OPTIMIZATION Целевая функция для поиска оптимального центра.
% Вычисляет энергию изгиба сплайна при заданном положении центра (xc, yc).
%
% Вход:
%   center_coords - вектор [xc, yc] - координаты центра (оптимизируемые переменные).
%   x_glob, y_glob - исходные декартовы координаты точек.
%   bc_spec       - структура спецификаций ГУ (в декартовых или инвариантных терминах).
%   alpha         - параметр сглаживания.
%
% Выход:
%   J_val         - значение функционала (энергия + невязка).

    %% 1. Переход к локальным координатам
    xc = center_coords(1);
    yc = center_coords(2);
    
    x_loc = x_glob - xc;
    y_loc = y_glob - yc;
    
    % Проверка на валидность координат (защита от NaN/Inf)
    if any(isnan(x_loc)) || any(isnan(y_loc))
        J_val = 1e10;
        return;
    end

    %% 2. Динамическое перестроение Граничных Условий
    % Так как центр сместился, полярные ГУ нужно пересчитать, 
    % чтобы сохранить геометрический смысл (наклон, кривизна), 
    % заданный в bc_spec.
    
    % --- Обработка Start Node ---
    bc_start = struct();
    
    % Значения координат фиксируем в новой точке
    bc_start.r = struct('value', sqrt(x_loc(1)^2 + y_loc(1)^2));
    bc_start.phi = struct('value', atan2(y_loc(1), x_loc(1)));
    
    % Производные пересчитываем из спецификаций
    [bc_start.r.deriv1, bc_start.phi.deriv1, bc_start.r.deriv2, bc_start.phi.deriv2] = ...
        update_derivatives(bc_spec.start, x_loc(1), y_loc(1));

    % --- Обработка End Node ---
    bc_end = struct();
    
    bc_end.r = struct('value', sqrt(x_loc(end)^2 + y_loc(end)^2));
    bc_end.phi = struct('value', atan2(y_loc(end), x_loc(end)));
    
    [bc_end.r.deriv1, bc_end.phi.deriv1, bc_end.r.deriv2, bc_end.phi.deriv2] = ...
        update_derivatives(bc_spec.end, x_loc(end), y_loc(end));

    %% 3. Создание и оптимизация сплайна
    try
        % Используем ваш класс (убедитесь, что он в пути Matlab)
        spline = Splines.PolarSmoothingSplineCoupled(x_loc, y_loc);
        spline.setBC(bc_start, bc_end);
        
        % Запуск внутренней оптимизации
        spline.fit(alpha);
        
        % Извлечение значения функционала
        % В классе нужно убедиться, что метод objective доступен или результат сохранен
        % В данном случае пересчитываем через objective для надежности
        % (предполагаем, что spline.results содержит оптимальный вектор)
        if isfield(spline, 'results') && ~isempty(spline.results)
             x_opt = spline.results.x_opt; % Нужно реализовать сохранение x_opt в классе
             J_val = spline.objective(x_opt, alpha);
        else
             % Если оптимизация не сошлась, штрафуем
             J_val = 1e8; 
        end
        
    catch ME
        % Если произошла ошибка (например, сингулярность матрицы при плохом центре),
        % возвращаем большой штраф.
        warning('Optimization failed for center [%.2f, %.2f]: %s', xc, yc, ME.message);
        J_val = 1e10;
    end
end

%% Вспомогательная функция пересчета производных
function [d1_r, d1_phi, d2_r, d2_phi] = update_derivatives(spec, x, y)
    % spec - структура с полями 'dy_dx' и 'd2y_dx2'
    
    % По умолчанию свободные (NaN)
    d1_r = NaN; d1_phi = NaN; 
    d2_r = NaN; d2_phi = NaN;
    
    if ~isempty(spec)
        % 1. Первые производные (Связь через y')
        if isfield(spec, 'dy_dx') && ~isnan(spec.dy_dx)
            k = spec.dy_dx;
            link = compute_polar_slope_link(x, y, k);
            
            if strcmp(link.type, 'coupled')
                % Задаем связь: r' зависит от phi', phi' свободна
                d1_r = struct('coupling', link.C);
                d1_phi = NaN; 
            elseif strcmp(link.type, 'radial_tangent')
                % Особый случай: phi' фиксирована нулем, r' свободна
                d1_r = NaN; 
                d1_phi = 0;
            end
        end
        
        % 2. Вторые производные (Фиксация через y'')
        if isfield(spec, 'd2y_dx2') && ~isnan(spec.d2y_dx2)
            % Для корректного расчета нужен наклон k
            k = 0;
            if isfield(spec, 'dy_dx'), k = spec.dy_dx; end
            
            bc_calc = convert_cartesian_bc_to_polar(x, y, k, spec.d2y_dx2);
            
            d2_r = bc_calc.r.deriv2;
            d2_phi = bc_calc.phi.deriv2;
        end
    end
end

