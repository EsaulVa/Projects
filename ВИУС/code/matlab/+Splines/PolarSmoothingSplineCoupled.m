classdef PolarSmoothingSplineCoupled < handle
    %POLARSMOOTHINGSPLINE Оптимизатор сплайна с поддержкой связанных ГУ.
    
    properties
        u           % Параметр
        r_data      % Данные
        phi_data
        n           % Кол-во точек
        
        geom_r      % Движок геометрии
        geom_phi
        
        % Результаты
        results     
        
        % Карта оптимизации
        map         % .indices, .fixed_vals, .links
    end
    
    methods
        function obj = PolarSmoothingSplineCoupled(x, y)
            obj.n = length(x);
            obj.geom_r = Splines.QuinticSplineGeometry();
            obj.geom_phi = Splines.QuinticSplineGeometry();
            
            dists = sqrt(diff(x).^2 + diff(y).^2);
            obj.u = [0; cumsum(dists)];
            
            obj.r_data = sqrt(x.^2 + y.^2);
            obj.phi_data = unwrap(atan2(y, x));
        end
        function setBC(obj, bc_start, bc_end)
    % Формирует карту переменных: свободные, фиксированные, зависимые.
    % Структура вектора состояния P (6*n):
    % [r_v(1..n), r_m(1..n), r_M(1..n), phi_v(1..n), phi_m(1..n), phi_M(1..n)]
    
    n = obj.n;
    map_size = 6 * n;
    
    % Статус: 0=Free (свободная), 1=Fixed (фиксированная), 2=Dependent (зависимая)
    status = zeros(map_size, 1);
    fixed_vals = zeros(map_size, 1);
    
    % Массив связей: dep_idx зависит от prim_idx с коэффициентом C
    links = struct('dep_idx', {}, 'prim_idx', {}, 'C', {});
    
    % --- Вспомогательная функция обработки одного узла ---
    function process_node(idx, bc, prefix, shift)
        % Определяем базовые индексы в полном векторе
        idx_v = shift + idx;
        idx_m = shift + n + idx;
        idx_M = shift + 2*n + idx;
        
        % 1. Обработка значения (Value)
        if isfield(bc, prefix) && isfield(bc.(prefix), 'value')
            val = bc.(prefix).value;
            if isnumeric(val) && ~isnan(val)
                status(idx_v) = 1;      % Fixed
                fixed_vals(idx_v) = val;
            end
        end
        
        % 2. Обработка первой производной (Deriv1)
        if isfield(bc, prefix) && isfield(bc.(prefix), 'deriv1')
            deriv_spec = bc.(prefix).deriv1;
            
            % === ИСПРАВЛЕНИЕ ПОРЯДКА ПРОВЕРОК ===
            
            % Сначала проверяем, это структура? (Случай Coupling)
            if isstruct(deriv_spec) && isfield(deriv_spec, 'coupling')
                
                C_val = deriv_spec.coupling;
                
                if strcmp(prefix, 'r')
                    dep_idx = idx_m;          % r' (зависимая)
                    prim_idx = idx_m + 3*n;   % phi' (первичная)
                else
                    dep_idx = idx_m;          % phi' (зависимая)
                    prim_idx = idx_m - 3*n;   % r' (первичная)
                end
                
                status(dep_idx) = 2;      % Ставим статус "Зависимая"
                
                % Статус prim_idx НЕ ТРОГАЕМ (как задал пользователь, так и осталось)
                
                % Добавляем связь
                links(end+1) = struct('dep_idx', dep_idx, 'prim_idx', prim_idx, 'C', C_val);
            
            % Потом проверяем, это число и NaN? (Случай Free)
            elseif isnumeric(deriv_spec) && isnan(deriv_spec)
                status(idx_m) = 0; 
            
            % Иначе считаем, что это число (Случай Fixed)
            else
                status(idx_m) = 1;        % Fixed
                fixed_vals(idx_m) = deriv_spec;
            end
        end
        
        % 3. Обработка второй производной (Deriv2)
        if isfield(bc, prefix) && isfield(bc.(prefix), 'deriv2')
            val = bc.(prefix).deriv2;
            if isnumeric(val) && isnan(val)
                status(idx_M) = 0;       % Free
            elseif isnumeric(val)
                status(idx_M) = 1;       % Fixed
                fixed_vals(idx_M) = val;
            end
        end
    end
    
    % Сдвиги блоков в векторе состояния
    shift_r = 0;
    shift_phi = 3 * n;
    
    % Обработка границ
    process_node(1, bc_start, 'r', shift_r);
    process_node(1, bc_start, 'phi', shift_phi);
    process_node(n, bc_end, 'r', shift_r);
    process_node(n, bc_end, 'phi', shift_phi);
    
    % Сохраняем результаты в свойства объекта
    obj.map.status = status;
    obj.map.fixed_vals = fixed_vals;
    obj.map.links = links;
    
    % Индексы для оптимизатора (статус 0 = Free)
    obj.map.free_idx = find(status == 0);
end
        

        
        function val = compute_velocity_error(obj, t, h, vr, mr, Mr, cr, vp, mp, Mp, cp)
            % Вычисляем r'(t) и phi'(t)
            r   = obj.geom_r.evalValue(t, vr, mr, Mr, cr);
            r1  = obj.geom_r.evalDeriv1(t, mr, Mr, cr);

            phi1 = obj.geom_phi.evalDeriv1(t, mp, Mp, cp);

            % Квадрат модуля скорости: v^2 = r'^2 + (r*phi')^2
            r_phi1=(r.*phi1);
            v_sq = r1.^2 + r_phi1.^2;
            v = sqrt(v_sq);

            % Штрафуем за отклонение от 1
            val = (v - 1).^2;
        end
        function J = objective(obj, vars_opt, alpha,beta)
            % 1. Восстановление полного вектора состояния P
            P_full = obj.map.fixed_vals; % Начинаем с констант
            
            % Вставляем свободные переменные
            P_full(obj.map.free_idx) = vars_opt;
            
            % === ВЫЧИСЛЕНИЕ ЗАВИСИМЫХ ПЕРЕМЕННЫХ ===
            % Применяем связи: m_r = C * m_phi
            for k = 1:length(obj.map.links)
                link = obj.map.links(k);
                % Берем значение первичной переменной из P_full
                prim_val = P_full(link.prim_idx);
                % Вычисляем зависимую
                P_full(link.dep_idx) = link.C * prim_val;
            end
            
            % 2. Распаковка в массивы
            n = obj.n;
            v_r   = P_full(1:n);
            m_r   = P_full(n+1:2*n);
            M_r   = P_full(2*n+1:3*n);
            
            v_phi = P_full(3*n+1:4*n);
            m_phi = P_full(4*n+1:5*n);
            M_phi = P_full(5*n+1:6*n);
            
            % 3. Функционал энергии
            J_fid = sum((v_r - obj.r_data).^2 + v_r.^2 .* (v_phi - obj.phi_data).^2);
            J_smooth = 0;
            
            for i = 1:n-1
                h = obj.u(i+1) - obj.u(i);
                
                coeffs_r = obj.geom_r.getSegmentCoeffs(v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
                coeffs_phi = obj.geom_phi.getSegmentCoeffs(v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);
                
                fun = @(t) obj.compute_segment_energy(t, h, ...
                    v_r(i), m_r(i), M_r(i), coeffs_r, ...
                    v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
                
                J_smooth = J_smooth + integral(fun, 0, h);
            end
            
            J_param = 0;    
            % Интегрируем квадрат ошибки скорости по всем сегментам
            for i = 1:n-1
                h = obj.u(i+1) - obj.u(i);

                % Функция ошибки скорости: (|v| - 1)^2
                % |v|^2 = (r')^2 + (r*phi')^2

                % Берем производные для сегмента i
                coeffs_r = obj.geom_r.getSegmentCoeffs(v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
                coeffs_phi = obj.geom_phi.getSegmentCoeffs(v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);

                fun_param = @(t) obj.compute_velocity_error(t, h, ...
                    v_r(i), m_r(i), M_r(i), coeffs_r, ...
                    v_phi(i), m_phi(i), M_phi(i), coeffs_phi);

                J_param = J_param + integral(fun_param, 0, h);
            end
%             beta=1e-2;
            J = alpha * J_fid + (1 - alpha) * J_smooth+beta*J_param;
        end
        
        function val = compute_segment_energy(obj, t, h, vr, mr, Mr, cr, vp, mp, Mp, cp)
            r   = obj.geom_r.evalValue(t, vr, mr, Mr, cr);
            r1  = obj.geom_r.evalDeriv1(t, mr, Mr, cr);
            r2  = obj.geom_r.evalDeriv2(t, Mr, cr);
            
            phi1 = obj.geom_phi.evalDeriv1(t, mp, Mp, cp);
            phi2 = obj.geom_phi.evalDeriv2(t, Mp, cp);
            
            acc_r = r2 - r .* (phi1.^2);
            acc_phi = r .* phi2 + 2 .* r1 .* phi1;
            val = acc_r.^2 + acc_phi.^2;
        end
        
        function unpack_results(obj, x_opt)
            n = obj.n;

            % 1. Восстанавливаем полный вектор P_full
            P_full = obj.map.fixed_vals; % Начинаем с фиксированных значений
            P_full(obj.map.free_idx) = x_opt; % Подставляем найденные свободные переменные

            % 2. Вычисляем зависимые переменные (Связи)
            % Если было условие r' = C * phi', то r' нужно вычислить здесь
            for k = 1:length(obj.map.links)
                link = obj.map.links(k);
                prim_val = P_full(link.prim_idx); % Значение первичной переменной (найденное оптимизатором)
                P_full(link.dep_idx) = link.C * prim_val; % Вычисляем зависимую
            end

            % 3. Распаковываем в свойства класса
            % R
            obj.results.r.v = P_full(1:n);
            obj.results.r.m = P_full(n+1:2*n);
            obj.results.r.M = P_full(2*n+1:3*n);

            % Phi
            obj.results.phi.v = P_full(3*n+1:4*n);
            obj.results.phi.m = P_full(4*n+1:5*n);
            obj.results.phi.M = P_full(5*n+1:6*n);
        end
%                 function J = objective(obj, vars_opt, alpha, beta, gamma)
%             % 1. Восстановление полного вектора состояния P
%             P_full = obj.map.fixed_vals; 
%             P_full(obj.map.free_idx) = vars_opt;
%             
%             for k = 1:length(obj.map.links)
%                 link = obj.map.links(k);
%                 prim_val = P_full(link.prim_idx);
%                 P_full(link.dep_idx) = link.C * prim_val;
%             end
%             
%             % 2. Распаковка
%             n = obj.n;
%             v_r   = P_full(1:n);
%             m_r   = P_full(n+1:2*n);
%             M_r   = P_full(2*n+1:3*n);
%             v_phi = P_full(3*n+1:4*n);
%             m_phi = P_full(4*n+1:5*n);
%             M_phi = P_full(5*n+1:6*n);
%             
%             % 3. Функционал энергии
%             J_fid = sum((v_r - obj.r_data).^2 + v_r.^2 .* (v_phi - obj.phi_data).^2);
%             J_smooth = 0;
%             J_param = 0;
%             J_jerk = 0; % Инициализация
%             
%             % Переменные для хранения коэффициентов предыдущего сегмента
%             coeffs_r_prev = [];
%             coeffs_phi_prev = [];
%             h_prev = 0;
%             
%             for i = 1:n-1
%                 h = obj.u(i+1) - obj.u(i);
%                 
%                 coeffs_r = obj.geom_r.getSegmentCoeffs(v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
%                 coeffs_phi = obj.geom_phi.getSegmentCoeffs(v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);
%                 
%                 % --- Smoothness Term ---
%                 fun = @(t) obj.compute_segment_energy(t, h, ...
%                     v_r(i), m_r(i), M_r(i), coeffs_r, ...
%                     v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
%                 J_smooth = J_smooth + integral(fun, 0, h);
%                 
%                 % --- Parametrization Term ---
%                 fun_param = @(t) obj.compute_velocity_error(t, h, ...
%                     v_r(i), m_r(i), M_r(i), coeffs_r, ...
%                     v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
%                 J_param = J_param + integral(fun_param, 0, h);
%                 
%                 % --- Jerk Term (Связность C3) ---
%                 % Вычисляем только если есть предыдущий сегмент (узлы i=2..n-1)
%                 if i > 1
%                     % Ошибка в узле i (стык сегмента i-1 и i)
%                     err_r = obj.compute_jerk_node_error(coeffs_r_prev, coeffs_r, h_prev);
%                     err_phi = obj.compute_jerk_node_error(coeffs_phi_prev, coeffs_phi, h_prev);
%                     J_jerk = J_jerk + err_r + err_phi;
%                 end
%                 
%                 % Запоминаем текущие коэффициенты для следующей итерации
%                 coeffs_r_prev = coeffs_r;
%                 coeffs_phi_prev = coeffs_phi;
%                 h_prev = h;
%             end
%             
%             J = alpha * J_fid + (1 - alpha) * J_smooth + beta * J_param + gamma * J_jerk;
%         end
        
         % === НОВЫЙ МЕТОД ===
        function val = compute_jerk_node_error(obj, coeffs_left, coeffs_right, h_left)
            % Вычисляет квадрат разности третьих производных на стыке сегментов.
            % S'''(t) = 6*a3 + 24*a4*t + 60*a5*t^2
            
            % 1. Левый сегмент (конец, t = h_left)
            d3_left = 6*coeffs_left(1) + 24*coeffs_left(2)*h_left + 60*coeffs_left(3)*h_left^2;
            
            % 2. Правый сегмент (начало, t = 0)
            d3_right = 6*coeffs_right(1);
            
            % Возвращаем квадрат невязки
            val = (d3_left - d3_right)^2;
        end
        
        function fit(obj, alpha,beta,gamma)
            % Инициализация
            n = obj.n;
            P_init = zeros(6*n, 1);
            
            % Заполняем данными
            P_init(1:n) = obj.r_data;
            P_init(3*n+1:4*n) = obj.phi_data;
            
            % Вставим в карту инициализации
            obj.map.fixed_vals = P_init; 
            
            % Восстанавливаем фиксированные (чтобы не затереть их данными)
            % и вычисляем зависимые для начальной точки
            % (хотя для x0 нужны только свободные индексы)
            
            x0 = P_init(obj.map.free_idx);
            
            fprintf('Запуск оптимизации. Свободных переменных: %d\n', length(x0));
            options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter','MaxIterations', 3000);
            objFun = @(x) obj.objective(x, alpha,beta);
            
            [x_opt, f_val, flag] = fminunc(objFun, x0, options);
            fprintf('Оптимизация завершена. Код: %d\n', flag);
            
            % Сохранение результатов (аналогично логике в objective)
            P_final = obj.map.fixed_vals;
            P_final(obj.map.free_idx) = x_opt;
            
            % Вычисляем зависимые переменные в финале
            for k = 1:length(obj.map.links)
                link = obj.map.links(k);
                P_final(link.dep_idx) = link.C * P_final(link.prim_idx);
            end
            
            obj.results.r.v = P_final(1:n);
            obj.results.r.m = P_final(n+1:2*n);
            obj.results.r.M = P_final(2*n+1:3*n);
            
            obj.results.phi.v = P_final(3*n+1:4*n);
            obj.results.phi.m = P_final(4*n+1:5*n);
            obj.results.phi.M = P_final(5*n+1:6*n);
            % Сохраняем результат
            obj.results.x_opt = x_opt;      % <--- Нужно добавить это
            obj.results.f_val = f_val;
            obj.unpack_results(x_opt); % Ваш метод распаковки
        end
    end
end

