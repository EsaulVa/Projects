classdef ParametricQuinticSpline < handle
    %PARAMETRICQUINTICSPLINE Параметрический сглаживающий сплайн
    
    properties
        points          % Исходные точки (N?2)
        n               % Количество точек
        u               % Параметр (длина хорды)
        bc_start        % Граничные условия в начале
        bc_end          % Граничные условия в конце
        spline_x        % Сплайн для X координаты
        spline_y        % Сплайн для Y координаты
        v_x, v_y        % Оптимизированные значения в узлах
        m_x, m_y        % Первые производные
        M_x, M_y        % Вторые производные
        coeffs_x, coeffs_y % Коэффициенты сегментов
    end
    
    methods
        function obj = ParametricQuinticSpline(points, bc_start, bc_end)
            % Конструктор
            obj.points = points;
            obj.n = size(points, 1);
            obj.bc_start = bc_start;
            obj.bc_end = bc_end;
            
            % Хордальная параметризация
            obj.u = obj.calculateChordParameter(points);
            
            % Преобразование граничных условий
            [bc_x, bc_y] = obj.convertBoundaryConditions();
            
            % Создание сплайнов для X и Y
            obj.spline_x = SmoothingQuinticSpline(obj.u, points(:, 1), ...
                bc_x{1}, bc_x{2});
            obj.spline_y = SmoothingQuinticSpline(obj.u, points(:, 2), ...
                bc_y{1}, bc_y{2});
        end
        
        function u = calculateChordParameter(obj, points)
            % Вычисление кумулятивной хордальной параметризации
            diffs = diff(points, 1);
            dists = sqrt(sum(diffs.^2, 2));
            u = zeros(size(points, 1), 1);
            u(2:end) = cumsum(dists);
        end
        
        function [bc_x, bc_y] = convertBoundaryConditions(obj)
            % Преобразование геометрических граничных условий в производные
            % Начальные условия
            T_start = obj.bc_start.direction;
            kappa_start = obj.bc_start.curvature;
            
            T_norm_start = T_start / norm(T_start);
            N_start = [-T_norm_start(2); T_norm_start(1)];
            speed_start = 1.0;
            curvature_vec_start = kappa_start * speed_start^2 * N_start;
            
            % Конечные условия
            T_end = obj.bc_end.direction;
            kappa_end = obj.bc_end.curvature;
            
            T_norm_end = T_end / norm(T_end);
            N_end = [-T_norm_end(2); T_norm_end(1)];
            speed_end = 1.0;
            curvature_vec_end = kappa_end * speed_end^2 * N_end;
            
            % Граничные условия для одномерных сплайнов
            bc_x_start = struct('m', T_norm_start(1)*speed_start, ...
                'M', curvature_vec_start(1));
            bc_y_start = struct('m', T_norm_start(2)*speed_start, ...
                'M', curvature_vec_start(2));
            
            bc_x_end = struct('m', T_norm_end(1)*speed_end, ...
                'M', curvature_vec_end(1));
            bc_y_end = struct('m', T_norm_end(2)*speed_end, ...
                'M', curvature_vec_end(2));
            
            bc_x = {bc_x_start, bc_x_end};
            bc_y = {bc_y_start, bc_y_end};
        end
        
        function fit(obj, alpha, alpha_x, alpha_y)
            % Оптимизация сплайна
            if nargin < 3
                alpha_x = alpha;
            end
            if nargin < 4
                alpha_y = alpha;
            end
            
            fprintf('Обучение параметрического сплайна...\n');
            fprintf('alpha_x=%.3f, alpha_y=%.3f\n', alpha_x, alpha_y);
            
            % Обучение сплайнов для X и Y
            obj.spline_x.fit(alpha_x);
            obj.spline_y.fit(alpha_y);
            
            % Сохраняем результаты
            obj.v_x = obj.spline_x.v;
            obj.v_y = obj.spline_y.v;
            obj.m_x = obj.spline_x.m;
            obj.m_y = obj.spline_y.m;
            obj.M_x = obj.spline_x.M;
            obj.M_y = obj.spline_y.M;
            
            obj.coeffs_x = obj.spline_x.segments_coeffs;
            obj.coeffs_y = obj.spline_y.segments_coeffs;
        end
        
        function result = predict(obj, u_eval, der)
            % Вычисление кривой или производных
            if nargin < 3
                der = 0;
            end
            
            u_eval = u_eval(:);
            
            if der == 3
                % Кривизна
                result = obj.curvature(u_eval);
                return;
            end
            
            result = zeros(length(u_eval), 2);
            
            for k = 1:length(u_eval)
                u = u_eval(k);
                
                if u < obj.u(1) || u > obj.u(end)
                    result(k, :) = [NaN, NaN];
                    continue;
                end
                
                % Находим интервал
                idx = find(obj.u <= u, 1, 'last');
                if idx < 1
                    idx = 1;
                elseif idx > obj.n-1
                    idx = obj.n-1;
                end
                
                t = u - obj.u(idx);
                
                % X компонента
                coeffs_x = obj.coeffs_x{idx};
                a3_x = coeffs_x(1); a4_x = coeffs_x(2); a5_x = coeffs_x(3);
                
                switch der
                    case 0
                        x_val = obj.v_x(idx) + obj.m_x(idx)*t + ...
                            0.5*obj.M_x(idx)*t^2 + a3_x*t^3 + a4_x*t^4 + a5_x*t^5;
                    case 1
                        x_val = obj.m_x(idx) + obj.M_x(idx)*t + ...
                            3*a3_x*t^2 + 4*a4_x*t^3 + 5*a5_x*t^4;
                    case 2
                        x_val = obj.M_x(idx) + 6*a3_x*t + 12*a4_x*t^2 + 20*a5_x*t^3;
                end
                
                % Y компонента
                coeffs_y = obj.coeffs_y{idx};
                a3_y = coeffs_y(1); a4_y = coeffs_y(2); a5_y = coeffs_y(3);
                
                switch der
                    case 0
                        y_val = obj.v_y(idx) + obj.m_y(idx)*t + ...
                            0.5*obj.M_y(idx)*t^2 + a3_y*t^3 + a4_y*t^4 + a5_y*t^5;
                    case 1
                        y_val = obj.m_y(idx) + obj.M_y(idx)*t + ...
                            3*a3_y*t^2 + 4*a4_y*t^3 + 5*a5_y*t^4;
                    case 2
                        y_val = obj.M_y(idx) + 6*a3_y*t + 12*a4_y*t^2 + 20*a5_y*t^3;
                end
                
                result(k, :) = [x_val, y_val];
            end
            
            % Нормализация касательных векторов
            if der == 1
                norms = sqrt(sum(result.^2, 2));
                norms(norms == 0) = 1;
                result = result ./ norms;
            end
        end
        
        function curvature = curvature(obj, u_eval)
            % Вычисление кривизны
            u_eval = u_eval(:);
            curvature = zeros(size(u_eval));
            
            for k = 1:length(u_eval)
                u = u_eval(k);
                
                if u < obj.u(1) || u > obj.u(end)
                    curvature(k) = NaN;
                    continue;
                end
                
                % Находим интервал
                idx = find(obj.u <= u, 1, 'last');
                if idx < 1
                    idx = 1;
                elseif idx > obj.n-1
                    idx = obj.n-1;
                end
                
                t = u - obj.u(idx);
                
                % Первые производные
                coeffs_x = obj.coeffs_x{idx};
                a3_x = coeffs_x(1); a4_x = coeffs_x(2); a5_x = coeffs_x(3);
                xp = obj.m_x(idx) + obj.M_x(idx)*t + ...
                    3*a3_x*t^2 + 4*a4_x*t^3 + 5*a5_x*t^4;
                
                coeffs_y = obj.coeffs_y{idx};
                a3_y = coeffs_y(1); a4_y = coeffs_y(2); a5_y = coeffs_y(3);
                yp = obj.m_y(idx) + obj.M_y(idx)*t + ...
                    3*a3_y*t^2 + 4*a4_y*t^3 + 5*a5_y*t^4;
                
                % Вторые производные
                xpp = obj.M_x(idx) + 6*a3_x*t + 12*a4_x*t^2 + 20*a5_x*t^3;
                ypp = obj.M_y(idx) + 6*a3_y*t + 12*a4_y*t^2 + 20*a5_y*t^3;
                
                % Кривизна
                numerator = xp*ypp - yp*xpp;
                denominator = (xp^2 + yp^2)^1.5;
                
                if denominator ~= 0
                    curvature(k) = numerator / denominator;
                end
            end
        end
        
        function L = length(obj, n_samples)
            % Оценка длины кривой
            if nargin < 2
                n_samples = 1000;
            end
            
            u_fine = linspace(obj.u(1), obj.u(end), n_samples)';
            points = obj.predict(u_fine, 0);
            diffs = diff(points, 1);
            L = sum(sqrt(sum(diffs.^2, 2)));
        end
    end
end

