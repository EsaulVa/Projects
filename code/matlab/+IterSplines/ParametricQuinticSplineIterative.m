classdef ParametricQuinticSplineIterative < handle
    %PARAMETRICQUINTICSPLINEITERATIVE Параметрический сплайн с итеративным уточнением весов
    % Реализует стабильный алгоритм минимизации энергии изгиба с использованием
    % эпсилон-стабилизации и демпфирования весов.
    
    properties
        points          % Исходные точки
        n               % Количество точек
        u               % Параметр (длина хорды)
        bc_start        % Граничные условия
        bc_end
        
        spline_x        % Объект SmoothingQuinticSplineNew для X
        spline_y        % Объект SmoothingQuinticSplineNew для Y
        
        % Кэши для быстрого доступа
        v_x, v_y
        m_x, m_y
        M_x, M_y
        coeffs_x, coeffs_y
        
        % --- Параметры стабильности ---
        epsilon = 1e-4;    % Параметр регуляризации знаменателя (из отчета)
        eta = 0.1;         % Коэффициент демпфирования весов (0 < eta <= 1)
                           % eta=1 - полное обновление (агрессивный метод)
                           % eta=0.3-0.7 - безопасное сглаживание
        
        % Вектор весов в узлах сетки (для демпфирования)
        W_values
    end
    
    methods
        function obj = ParametricQuinticSplineIterative(points, bc_start, bc_end)
            % Конструктор
            obj.points = points;
            obj.n = size(points, 1);
            obj.bc_start = bc_start;
            obj.bc_end = bc_end;
            
            % Параметризация (хордальная)
            obj.u = obj.calculateChordParameter(points);
            
            % Преобразование ГУ
            [bc_x, bc_y] = obj.convertBoundaryConditions();
            
            % Создание экземпляров SmoothingQuinticSplineNew
            obj.spline_x = IterSplines.SmoothingQuinticSplineNew(obj.u, points(:,1), bc_x{1}, bc_x{2});
            obj.spline_y = IterSplines.SmoothingQuinticSplineNew(obj.u, points(:,2), bc_y{1}, bc_y{2});
            
            % Если класс содержит свойство lambda_vel (от старой версии), обнуляем его
            if isprop(obj.spline_x, 'lambda_vel')
                obj.spline_x.lambda_vel = 0;
                obj.spline_y.lambda_vel = 0;
            end
            
            % Инициализация весов единицами
            obj.W_values = ones(size(obj.u));
        end
        
        function fit(obj, alpha, max_outer_iter)
            % Основной метод обучения (Стабильная версия)
            if nargin < 3, max_outer_iter = 10; end
            
            fprintf('=== Запуск итеративной линеаризации (Стабильная версия) ===\n');
            fprintf('Параметры: eps = %.1e, eta = %.2f\n', obj.epsilon, obj.eta);
            
            % 0. Инициализация (обычный сплайн, W=1)
            obj.spline_x.fit(alpha);
            obj.spline_y.fit(alpha);
            obj.updateCache();
            
            % Начальная энергия
            prev_J = obj.calculateTotalEnergy();
            
            for k = 1:max_outer_iter
                % --- Шаг А: Вычисление "идеальных" весов на текущей геометрии ---
                W_ideal = obj.calculateStableWeightsVector();
                
                % --- Шаг Б: Демпфирование весов (Рекомендация 3) ---
                % W_new = (1-eta)*W_old + eta*W_ideal
                obj.W_values = (1 - obj.eta) * obj.W_values + obj.eta * W_ideal;
                
                % --- Подготовка функции весов для передачи в сплайны ---
                % Используем интерполяцию, так как интегрирование идет по t
                W_func = @(u_val) interp1(obj.u, obj.W_values, u_val, 'linear', 'extrap');
                
                % --- Шаг В: Оптимизация X (фиксируем Y) ---
                y_der1 = @(u_val) obj.spline_y.predict(u_val, 1);
                y_der2 = @(u_val) obj.spline_y.predict(u_val, 2);
                
                obj.spline_x.setContext(W_func, y_der1, y_der2);
                x0_x = obj.packVariables(obj.spline_x);
                obj.spline_x.fit(alpha, x0_x); 
                obj.spline_x.clearContext();
                
                obj.updateCache(); % Обновляем кэш X
                
                % --- Шаг Г: Оптимизация Y (фиксируем НОВЫЙ X) ---
                x_der1 = @(u_val) obj.spline_x.predict(u_val, 1);
                x_der2 = @(u_val) obj.spline_x.predict(u_val, 2);
                
                obj.spline_y.setContext(W_func, x_der1, x_der2);
                x0_y = obj.packVariables(obj.spline_y);
                obj.spline_y.fit(alpha, x0_y);
                obj.spline_y.clearContext();
                
                obj.updateCache(); % Обновляем кэш Y
                
                % --- Проверка сходимости ---
                cur_J = obj.calculateTotalEnergy();
                diff_J = cur_J - prev_J;
                
                fprintf('Итерация %d: Энергия = %.6f, Изменение = %.2e', k, cur_J, diff_J);
                
                if diff_J > 1e-7
                    fprintf(' [Внимание: рост энергии]\n');
                    % Если энергия растет, можно уменьшить eta для следующего шага
                    obj.eta = max(0.1, obj.eta * 0.8); 
                else
                    fprintf('\n');
                end
                
                if abs(diff_J) < 1e-7
                    fprintf('Сходимость достигнута.\n');
                    break;
                end
                prev_J = cur_J;
            end
        end
        
        function updateCache(obj)
            obj.v_x = obj.spline_x.v; obj.v_y = obj.spline_y.v;
            obj.m_x = obj.spline_x.m; obj.m_y = obj.spline_y.m;
            obj.M_x = obj.spline_x.M; obj.M_y = obj.spline_y.M;
            obj.coeffs_x = obj.spline_x.segments_coeffs;
            obj.coeffs_y = obj.spline_y.segments_coeffs;
        end
        
        function vec = packVariables(~, spline_obj)
            n = spline_obj.n;
            vec = [spline_obj.v; spline_obj.m(2:end-1); spline_obj.M(2:end-1)];
        end
        
        function W_vec = calculateStableWeightsVector(obj)
            % Вычисляет веса во всех узлах u по формуле стабилизации
            % W = (x'^2 + y'^2 + eps^2)^(-2.5)
            
            xp = obj.spline_x.predict(obj.u, 1);
            yp = obj.spline_y.predict(obj.u, 1);
            
            v_sq = xp.^2 + yp.^2;
            
            % Стабилизированная формула (Рекомендация 1 отчета)
            W_vec = (v_sq + obj.epsilon^2).^-2.5;
        end
        
        function E = calculateTotalEnergy(obj)
            % Оценка энергии изгиба интегрированием
            u_check = linspace(obj.u(1), obj.u(end), 200)';
            
            % Кривизна
            k_vals = obj.curvature(u_check);
            
            % Элемент длины ds = |R'| du
            xp = obj.spline_x.predict(u_check, 1);
            yp = obj.spline_y.predict(u_check, 1);
            speed = sqrt(xp.^2 + yp.^2);
            
            % Интеграл (метод прямоугольников или трапеций)
            du_step = (obj.u(end) - obj.u(1)) / (length(u_check) - 1);
            E = sum(k_vals.^2 .* speed) * du_step;
        end
        
        % --- Стандартные методы predict и curvature ---
        
        function result = predict(obj, u_eval, der)
            if nargin < 3, der = 0; end
            result = [obj.spline_x.predict(u_eval, der), obj.spline_y.predict(u_eval, der)];
        end
        
        function curvature = curvature(obj, u_eval)
            xp = obj.spline_x.predict(u_eval, 1);
            yp = obj.spline_y.predict(u_eval, 1);
            xpp = obj.spline_x.predict(u_eval, 2);
            ypp = obj.spline_y.predict(u_eval, 2);
            
            numerator = abs(xp.*ypp - yp.*xpp);
            denominator = (xp.^2 + yp.^2).^1.5;
            denominator(denominator==0) = 1e-10;
            
            curvature = numerator ./ denominator;
        end
        
        function u = calculateChordParameter(obj, points)
            diffs = diff(points, 1);
            dists = sqrt(sum(diffs.^2, 2));
            u = zeros(size(points, 1), 1);
            u(2:end) = cumsum(dists);
        end
        
        function [bc_x, bc_y] = convertBoundaryConditions(obj)
            T_start = obj.bc_start.direction; kappa_start = obj.bc_start.curvature;
            T_norm_start = T_start / norm(T_start);
            N_start = [-T_norm_start(2); T_norm_start(1)];
            curvature_vec_start = kappa_start * N_start;
            
            T_end = obj.bc_end.direction; kappa_end = obj.bc_end.curvature;
            T_norm_end = T_end / norm(T_end);
            N_end = [-T_norm_end(2); T_norm_end(1)];
            curvature_vec_end = kappa_end * N_end;
            
            bc_x_start = struct('m', T_norm_start(1), 'M', curvature_vec_start(1));
            bc_y_start = struct('m', T_norm_start(2), 'M', curvature_vec_start(2));
            bc_x_end = struct('m', T_norm_end(1), 'M', curvature_vec_end(1));
            bc_y_end = struct('m', T_norm_end(2), 'M', curvature_vec_end(2));
            
            bc_x = {bc_x_start, bc_x_end}; bc_y = {bc_y_start, bc_y_end};
        end
    end
end