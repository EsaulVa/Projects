classdef SmoothingQuinticSpline < QuinticSpline
    %SMOOTHINGQUINTICSPLINE Сглаживающий квинтический сплайн
    
    properties
        y_data      % Исходные данные
        alpha       % Параметр сглаживания
        v           % Оптимизированные значения в узлах
    end
    
    methods
        function obj = SmoothingQuinticSpline(x, y_data, bc_left, bc_right)
            % Конструктор
            obj = obj@QuinticSpline(x, y_data, bc_left, bc_right);
            obj.y_data = y_data(:);
        end
        
        function J = objective(obj, vars_opt, alpha)
            % Целевая функция для оптимизации
            n = obj.n;
            
            % Распаковка переменных
            v = vars_opt(1:n);
            m_internal = vars_opt(n+1:n+(n-2));
            M_internal = vars_opt(n+(n-2)+1:end);
            
            % Сборка полных массивов
            m = zeros(n, 1);
            M = zeros(n, 1);
            
            % Граничные условия
            m(1) = obj.bc_left.m;
            M(1) = obj.bc_left.M;
            m(end) = obj.bc_right.m;
            M(end) = obj.bc_right.M;
            
            % Внутренние узлы
            m(2:end-1) = m_internal;
            M(2:end-1) = M_internal;
            
            % Штраф за отклонение от данных
            fidelity_term = sum((v - obj.y_data).^2);
            
            % Энергия изгиба
            smoothness_term = 0;
            
            for i = 1:n-1
                coeffs = obj.getSegmentCoeffs(i, v, m, M);
                a3 = coeffs(1); a4 = coeffs(2); a5 = coeffs(3);
                h = obj.x(i+1) - obj.x(i);
                
                % Интеграл квадрата второй производной
                fun = @(t) (M(i) + 6*a3*t + 12*a4*t.^2 + 20*a5*t.^3).^2;
                integral_val = integral(fun, 0, h);
                smoothness_term = smoothness_term + integral_val;
            end
            
            J = alpha * fidelity_term + (1 - alpha) * smoothness_term;
        end
        
        function fit(obj, alpha)
            % Оптимизация сплайна
            obj.alpha = alpha;
            n = obj.n;
            
            % Начальное приближение
            v_init = obj.y_data;
            
            % Используем кубический сплайн для инициализации производных
            pp = spline(obj.x, [obj.bc_left.m; obj.y_data; obj.bc_right.m]);
            pp_der1 = fnder(pp, 1);
            pp_der2 = fnder(pp, 2);
            
            m_init = ppval(pp_der1, obj.x);
            M_init = ppval(pp_der2, obj.x);
            
            % Формируем вектор оптимизации
            m_opt_vars = m_init(2:end-1);
            M_opt_vars = M_init(2:end-1);
            
            x0 = [v_init; m_opt_vars; M_opt_vars];
            
            fprintf('Оптимизация сглаживающего сплайна (alpha=%.3f)...\n', alpha);
            fprintf('Размер вектора переменных: %d\n', length(x0));
            
            % Оптимизация с помощью fminunc
            options = optimoptions('fminunc', ...
                'Display', 'iter', ...
                'Algorithm', 'quasi-newton', ...
                'MaxIterations', 1000, ...
                'OptimalityTolerance', 1e-8);
            
            objFun = @(x) obj.objective(x, alpha);
            [x_opt, ~, exitflag] = fminunc(objFun, x0, options);
            
            fprintf('Оптимизация завершена. Статус: %d\n', exitflag);
            
            % Сохраняем результаты
            obj.v = x_opt(1:n);
            
            obj.m(1) = obj.bc_left.m; obj.m(end) = obj.bc_right.m;
            obj.M(1) = obj.bc_left.M; obj.M(end) = obj.bc_right.M;
            
            obj.m(2:end-1) = x_opt(n+1:n+(n-2));
            obj.M(2:end-1) = x_opt(n+(n-2)+1:end);
            
            % Вычисляем коэффициенты сегментов
            obj.segments_coeffs = cell(n-1, 1);
            for i = 1:n-1
                obj.segments_coeffs{i} = obj.getSegmentCoeffs(i, obj.v, obj.m, obj.M);
            end
        end
        
        function result = predict(obj, x_eval, der)
            % Вычисление сплайна или его производных
            if nargin < 3
                der = 0;
            end
            
            x_eval = x_eval(:);
            result = zeros(size(x_eval));
            
            for k = 1:length(x_eval)
                xv = x_eval(k);
                
                if xv < obj.x(1) || xv > obj.x(end)
                    result(k) = NaN;
                    continue;
                end
                
                % Находим интервал
                idx = find(obj.x <= xv, 1, 'last');
                if isempty(idx) || idx < 1
                    idx = 1;
                elseif idx > obj.n-1
                    idx = obj.n-1;
                end
                
                t = xv - obj.x(idx);
                vi = obj.v(idx);
                mi = obj.m(idx);
                Mi = obj.M(idx);
                coeffs = obj.segments_coeffs{idx};
                a3 = coeffs(1); a4 = coeffs(2); a5 = coeffs(3);
                
                switch der
                    case 0
                        val = vi + mi*t + (Mi/2)*t^2 + a3*t^3 + a4*t^4 + a5*t^5;
                    case 1
                        val = mi + Mi*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4;
                    case 2
                        val = Mi + 6*a3*t + 12*a4*t^2 + 20*a5*t^3;
                    otherwise
                        error('Неподдерживаемый порядок производной');
                end
                
                result(k) = val;
            end
        end
    end
end

