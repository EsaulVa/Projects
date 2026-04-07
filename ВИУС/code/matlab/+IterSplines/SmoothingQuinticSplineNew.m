classdef SmoothingQuinticSplineNew < Splines.SmoothingQuinticSpline
    %SMOOTHINGQUINTICSPLINENEW Сплайн с учетом весов, кросс-термов и штрафа за скорость
    
    properties
        weights_func      % Function handle: @(u) W(u)
        partner_der1_func % Function handle: @(u) y'(u) или x'(u)
        partner_der2_func % Function handle: @(u) y''(u) или x''(u)
        useWeightedMode = false; % Флаг режима
        
        % Коэффициент штрафа за отклонение скорости от 1
        lambda_vel = 1e-3; 
    end
    
    methods
        function obj = SmoothingQuinticSplineNew(x, y_data, bc_left, bc_right)
            % Конструктор вызывает родительский, указывая полный путь к пакету
            obj = obj@Splines.SmoothingQuinticSpline(x, y_data, bc_left, bc_right);
        end
        
        function setContext(obj, W_func, p_der1, p_der2)
            % Установка внешнего контекста (веса и данные партнера)
            obj.weights_func = W_func;
            obj.partner_der1_func = p_der1;
            obj.partner_der2_func = p_der2;
            obj.useWeightedMode = true;
        end
        
        function clearContext(obj)
            % Сброс контекста (возврат к простому режиму)
            obj.useWeightedMode = false;
        end
        
        function J = objective(obj, vars_opt, alpha)
            % Переопределенный расчет функционала
            
            % 1. Распаковка переменных (как в родителе)
            n = obj.n;
            v = vars_opt(1:n);
            m_internal = vars_opt(n+1:n+(n-2));
            M_internal = vars_opt(n+(n-2)+1:end);
            
            m = zeros(n, 1); M = zeros(n, 1);
            m(1) = obj.bc_left.m; M(1) = obj.bc_left.M;
            m(end) = obj.bc_right.m; M(end) = obj.bc_right.M;
            m(2:end-1) = m_internal;
            M(2:end-1) = M_internal;
            
            fidelity_term = sum((v - obj.y_data).^2);
            smoothness_term = 0;
            velocity_penalty_term = 0;
            
            for i = 1:n-1
                % Получаем коэффициенты сегмента
                coeffs = obj.getSegmentCoeffs(i, v, m, M);
                a3 = coeffs(1); a4 = coeffs(2); a5 = coeffs(3);
                h = obj.x(i+1) - obj.x(i);
                
                % Инициализируем производные текущего сегмента
                mi = m(i);
                Mi = M(i);
                
                if obj.useWeightedMode
                    % === НОВЫЙ ФУНКЦИОНАЛ (Взвешенная кривизна) ===
                    u_start = obj.x(i);
                    
                    % 1. Энергия кривизны
                    fun_curv = @(t) obj.computeWeightedIntegrand(t, u_start, mi, Mi, a3, a4, a5);
                    integral_curv = integral(fun_curv, 0, h, 'AbsTol', 1e-10);
                    
                    % 2. Штраф за скорость
                    fun_vel = @(t) obj.computeVelocityPenalty(t, u_start, mi, Mi, a3, a4, a5);
                    integral_vel = integral(fun_vel, 0, h, 'AbsTol', 1e-10);
                    
                    smoothness_term = smoothness_term + integral_curv;
                    velocity_penalty_term = velocity_penalty_term + integral_vel;
                else
                    % === СТАРЫЙ ФУНКЦИОНАЛ ===
                    fun = @(t) (Mi + 6*a3*t + 12*a4*t.^2 + 20*a5*t.^3).^2;
                    integral_val = integral(fun, 0, h, 'AbsTol', 1e-10);
                    smoothness_term = smoothness_term + integral_val;
                end
            end
            
            % Итоговый функционал
            J = alpha * fidelity_term + (1 - alpha) * smoothness_term + velocity_penalty_term;
        end
        
        function val = computeWeightedIntegrand(obj, t, u_start, mi, Mi, a3, a4, a5)
            % Вспомогательная функция для кривизны
            sz = size(t);
            
            % 1. Производные оптимизируемого сплайна (переменные)
            der1_opt = mi + Mi*t + 3*a3*t.^2 + 4*a4*t.^3 + 5*a5*t.^4;
            der2_opt = Mi + 6*a3*t + 12*a4*t.^2 + 20*a5*t.^3;
            
            % 2. Глобальный параметр u
            u_val = u_start + t;
            
            % 3. Данные партнера (с коррекцией формы)
            der1_part = obj.partner_der1_func(u_val);
            der1_part = reshape(der1_part, sz);
            
            der2_part = obj.partner_der2_func(u_val);
            der2_part = reshape(der2_part, sz);
            
            % 4. Веса
            W = obj.weights_func(u_val);
            W = reshape(W, sz);
            
            % 5. Формула кривизны: числитель (x'y'' - x''y')
            numerator = der1_opt .* der2_part - der2_opt .* der1_part;
            
            val = W .* (numerator.^2);
        end
        
        function val = computeVelocityPenalty(obj, t, u_start, mi, Mi, a3, a4, a5)
            % Вспомогательная функция для штрафа за скорость
            sz = size(t);
            
            % 1. Производная оптимизируемого сплайна
            der1_opt = mi + Mi*t + 3*a3*t.^2 + 4*a4*t.^3 + 5*a5*t.^4;
            
            % 2. Производная партнера (с коррекцией формы)
            u_val = u_start + t;
            der1_part = obj.partner_der1_func(u_val);
            der1_part = reshape(der1_part, sz);
            
            % 3. Скорость |R'|^2
            v_sq = der1_opt.^2 + der1_part.^2;
            
            % 4. Штраф: lambda * (v^2 - 1)^2
            % lambda_vel берем из свойств объекта
            val = obj.lambda_vel * (v_sq -1).^2;
        end
        
        function fit(obj, alpha, init_guess)
            if nargin < 3
                fit@Splines.SmoothingQuinticSpline(obj, alpha);
            else
                obj.alpha = alpha;
                x0 = init_guess;
                
                options = optimoptions('fminunc', ...
                    'Display', 'off', ...
                    'Algorithm', 'quasi-newton', ...
                    'MaxIterations', 500, ...
                    'OptimalityTolerance', 1e-8);
                
                objFun = @(x) obj.objective(x, alpha);
                [x_opt, ~, ~] = fminunc(objFun, x0, options);
                
                % Распаковка результатов
                n = obj.n;
                obj.v = x_opt(1:n);
                
                obj.m = zeros(n,1); obj.M = zeros(n,1);
                obj.m(1) = obj.bc_left.m; obj.m(end) = obj.bc_right.m;
                obj.M(1) = obj.bc_left.M; obj.M(end) = obj.bc_right.M;
                
                obj.m(2:end-1) = x_opt(n+1:n+(n-2));
                obj.M(2:end-1) = x_opt(n+(n-2)+1:end);
                
                obj.segments_coeffs = cell(n-1, 1);
                for i = 1:n-1
                    obj.segments_coeffs{i} = obj.getSegmentCoeffs(i, obj.v, obj.m, obj.M);
                end
            end
        end
    end
end