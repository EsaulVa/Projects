classdef PolarSmoothingSplineCoupledNew < handle
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
        function obj = PolarSmoothingSplineCoupledNew(x, y)
            obj.n = length(x);
            obj.geom_r = Splines.QuinticSplineGeometry();
            obj.geom_phi = Splines.QuinticSplineGeometry();
            
            dists = sqrt(diff(x).^2 + diff(y).^2);
            obj.u = [0; cumsum(dists)];
            
            obj.r_data = sqrt(x.^2 + y.^2);
            obj.phi_data = unwrap(atan2(y, x));
        end
        function setBC(obj, bc_start, bc_end)
            %SETBC Формирует карту переменных с поддержкой Soft Constraints.
            %
            % Структура вектора состояния P (6*n):
            % [r_v(1..n), r_m(1..n), r_M(1..n), phi_v(1..n), phi_m(1..n), phi_M(1..n)]

            n = obj.n;
            map_size = 6 * n;

            % Статус: 0=Free, 1=Fixed, 2=Dependent
            status = zeros(map_size, 1);
            fixed_vals = zeros(map_size, 1);

            % Массивы для связей и штрафов
            links = struct('dep_idx', {}, 'prim_idx', {}, 'C', {});
            soft_constraints = struct('var_idx', {}, 'target', {}, 'weight', {}, 'type', {}, 'node_idx', {});

            % --- Вспомогательная функция обработки одного узла ---
            function process_node(idx, bc, prefix, shift)
                % Определяем базовые индексы в полном векторе
                idx_v = shift + idx;           % Value
                idx_m = shift + n + idx;       % Deriv1
                idx_M = shift + 2*n + idx;     % Deriv2

                % === 1. Обработка значения (Value) ===
                if isfield(bc, prefix) && isfield(bc.(prefix), 'value')
                    val = bc.(prefix).value;

                    % Проверка на Soft Constraint
                    if isstruct(val) && isfield(val, 'target')
                        status(idx_v) = 0; % Free
                        soft_constraints(end+1) = struct('var_idx', idx_v, 'target', val.target, 'weight', val.weight, 'type', 'simple', 'node_idx', idx);

                    % Обычное Fixed значение
                    elseif isnumeric(val) && ~isnan(val)
                        status(idx_v) = 1;      % Fixed
                        fixed_vals(idx_v) = val;
                    end
                end

                % === 2. Обработка первой производной (Deriv1) ===
                if isfield(bc, prefix) && isfield(bc.(prefix), 'deriv1')
                    spec = bc.(prefix).deriv1;

                    % Случай A: Линейная связь (Coupling) - r' = C * phi'
                    if isstruct(spec) && isfield(spec, 'coupling')
                        C_val = spec.coupling;

                        if strcmp(prefix, 'r')
                            dep_idx = idx_m;          % r' (зависимая)
                            prim_idx = idx_m + 3*n;   % phi' (первичная)
                        else
                            dep_idx = idx_m;          % phi' (зависимая)
                            prim_idx = idx_m - 3*n;   % r' (первичная)
                        end

                        status(dep_idx) = 2;      % Dependent
                        links(end+1) = struct('dep_idx', dep_idx, 'prim_idx', prim_idx, 'C', C_val);

                    % Случай B: Soft Constraint (Целевое значение с весом)
                    elseif isstruct(spec) && isfield(spec, 'target')
                        status(idx_m) = 0; % Free
                        soft_constraints(end+1) = struct('var_idx', idx_m, 'target', spec.target, 'weight', spec.weight, 'type', 'simple', 'node_idx', idx);

                    % Случай C: Free (NaN)
                    elseif isnumeric(spec) && isnan(spec)
                        status(idx_m) = 0;

                    % Случай D: Fixed (Число)
                    elseif isnumeric(spec)
                        status(idx_m) = 1;        % Fixed
                        fixed_vals(idx_m) = spec;
                    end
                end

                % === 3. Обработка второй производной (Deriv2) ===
                if isfield(bc, prefix) && isfield(bc.(prefix), 'deriv2')
                    spec = bc.(prefix).deriv2;

                    % Случай A: Soft Constraint
                    if isstruct(spec) && isfield(spec, 'target')
                        status(idx_M) = 0; % Free
                        soft_constraints(end+1) = struct('var_idx', idx_M, 'target', spec.target, 'weight', spec.weight, 'type', 'simple', 'node_idx', idx);

                    % Случай B: Free
                    elseif isnumeric(spec) && isnan(spec)
                        status(idx_M) = 0;

                    % Случай C: Fixed
                    elseif isnumeric(spec)
                        status(idx_M) = 1;
                        fixed_vals(idx_M) = spec;
                    end
                end
            end

            % === Обработка комплексных условий (y_deriv2) ===
            % Эти условия затрагивают несколько переменных одновременно (r, phi и их производные)
            function process_complex_bc(bc, node_idx)
                if isfield(bc, 'y_deriv2')
                    spec = bc.y_deriv2;
                    if isstruct(spec) && isfield(spec, 'target')
                        % Для этого условия переменные должны быть свободны (status=0).
                        % Мы не проверяем это здесь явно, но предполагаем, что 
                        % в bc для переменных (deriv2) передано NaN или Soft.

                        % Добавляем запись о штрафе
                        soft_constraints(end+1) = struct(...
                            'var_idx', NaN, ... % Нет одной переменной
                            'target', spec.target, ...
                            'weight', spec.weight, ...
                            'type', 'y_ddx', ... % Специальный тип
                            'node_idx', node_idx);
                    end
                end
            end

            % Сдвиги блоков в векторе состояния
            shift_r = 0;
            shift_phi = 3 * n;

            % Обработка границ
            process_node(1, bc_start, 'r', shift_r);
            process_node(1, bc_start, 'phi', shift_phi);
            process_complex_bc(bc_start, 1); % Если нужно условие на кривизну в начале

            process_node(n, bc_end, 'r', shift_r);
            process_node(n, bc_end, 'phi', shift_phi);
            process_complex_bc(bc_end, n);   % Условие на кривизну в конце

            % Сохраняем результаты в свойства объекта
            obj.map.status = status;
            obj.map.fixed_vals = fixed_vals;
            obj.map.links = links;
            obj.map.soft_constraints = soft_constraints; % Новое поле

            % Индексы для оптимизатора (статус 0 = Free)
            obj.map.free_idx = find(status == 0);
        end
        function val = compute_velocity_error(obj, t, h, vr, mr, Mr, cr, vp, mp, Mp, cp)
            r   = obj.geom_r.evalValue(t, vr, mr, Mr, cr);
            r1  = obj.geom_r.evalDeriv1(t, mr, Mr, cr);
            phi1 = obj.geom_phi.evalDeriv1(t, mp, Mp, cp);
            r_phi1=(r.*phi1);
            v_sq = r1.^2 + r_phi1.^2;
            v = sqrt(v_sq);
            val = (v - 1).^2;
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
        
%         function J = objective(obj, vars_opt, alpha, beta, gamma)
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
%             
%              % === Обработка мягких ограничений ===
%             if isfield(obj.map, 'soft_constraints')
%                 constraints = obj.map.soft_constraints;
%                 for k = 1:length(constraints)
%                     con = constraints(k);
% 
%                     % Тип 1: Простое соответствие переменной значению
%                     if strcmp(con.type, 'simple')
%                         val = P_full(con.var_idx);
%                         penalty = (val - con.target)^2;
%                         J = J + con.weight * penalty;
% 
%                     % Тип 2: Геометрическое условие y''(x)
%                     elseif strcmp(con.type, 'y_ddx')
%                         % Берем данные с КОНЦА сплайна (узел n)
%                         n = obj.n;
%                         r     = P_full(1:n);      % Массив r
%                         phi   = P_full(3*n+1:4*n); % Массив phi
%                         r1    = P_full(n+1:2*n);
%                         phi1  = P_full(4*n+1:5*n);
%                         r2    = P_full(2*n+1:3*n);
%                         phi2  = P_full(5*n+1:6*n);
% 
%                         % Значения в последнем узле
%                         rn = r(n); phin = phi(n);
%                         r1n = r1(n); phi1n = phi1(n);
%                         r2n = r2(n); phi2n = phi2(n);
% 
%                         % Формула для y''(x) при y'(x)=0 (горизонтальная касательная)
%                         % y'' = a_r * sin(phi) + a_phi * cos(phi)
%                         ar = r2n - rn * (phi1n^2);
%                         aphi = rn * phi2n + 2 * r1n * phi1n;
% 
%                         y2_val = ar * sin(phin) + aphi * cos(phin);
% 
%                         penalty = (y2_val - con.target)^2;
%                         J = J + con.weight * penalty;
%                     end
%                 end
%             end
%         end
        
%         function J = objective(obj, vars_opt, alpha, beta, gamma)
%     %OBJECTIVE Целевая функция оптимизации.
%     % Включает: Fidelity, Smoothness, Parametrization, Jerk, Soft Constraints.
%     
%     % --- 1. Восстановление полного вектора состояния P ---
%     P_full = obj.map.fixed_vals; 
%     P_full(obj.map.free_idx) = vars_opt;
%     
%     % --- 2. Вычисление зависимых переменных (Links) ---
%     % Применяем связи: m_r = C * m_phi и т.д.
%     for k = 1:length(obj.map.links)
%         link = obj.map.links(k);
%         prim_val = P_full(link.prim_idx);
%         P_full(link.dep_idx) = link.C * prim_val;
%     end
%     
%     % --- 3. Распаковка в массивы ---
%     n = obj.n;
%     v_r   = P_full(1:n);
%     m_r   = P_full(n+1:2*n);
%     M_r   = P_full(2*n+1:3*n);
%     
%     v_phi = P_full(3*n+1:4*n);
%     m_phi = P_full(4*n+1:5*n);
%     M_phi = P_full(5*n+1:6*n);
%     
%     % --- 4. Вычисление базовых функционалов ---
%     
%     % Fidelity (Штраф за невязку данных)
%     J_fid = sum((v_r - obj.r_data).^2 + v_r.^2 .* (v_phi - obj.phi_data).^2);
%     
%     J_smooth = 0;
%     J_param = 0;
%     J_jerk = 0;
%     
%     coeffs_r_prev = [];
%     coeffs_phi_prev = [];
%     h_prev = 0;
%     
%     % Цикл по сегментам
%     for i = 1:n-1
%         h = obj.u(i+1) - obj.u(i);
%         
%         % Коэффициенты сегмента
%         coeffs_r = obj.geom_r.getSegmentCoeffs(v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
%         coeffs_phi = obj.geom_phi.getSegmentCoeffs(v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);
%         
%         % -- Smoothness (Энергия изгиба) --
%         fun = @(t) obj.compute_segment_energy(t, h, ...
%             v_r(i), m_r(i), M_r(i), coeffs_r, ...
%             v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
%         J_smooth = J_smooth + integral(fun, 0, h);
%         
%         % -- Parametrization (Штраф |v|-1) --
%         fun_param = @(t) obj.compute_velocity_error(t, h, ...
%             v_r(i), m_r(i), M_r(i), coeffs_r, ...
%             v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
%         J_param = J_param + integral(fun_param, 0, h);
%         
%         % -- Jerk (Непрерывность 3-й производной) --
%         if i > 1
%             err_r = obj.compute_jerk_node_error(coeffs_r_prev, coeffs_r, h_prev);
%             err_phi = obj.compute_jerk_node_error(coeffs_phi_prev, coeffs_phi, h_prev);
%             J_jerk = J_jerk + err_r + err_phi;
%         end
%         
%         % Запоминаем текущие коэффициенты для следующего шага
%         coeffs_r_prev = coeffs_r;
%         coeffs_phi_prev = coeffs_phi;
%         h_prev = h;
%     end
%     
%     % --- 5. Обработка Soft Constraints (Мягкие ограничения) ---
%     J_bc = 0;
%     
%     if isfield(obj.map, 'soft_constraints')
%         constraints = obj.map.soft_constraints;
%         
%         for k = 1:length(constraints)
%             con = constraints(k);
%             
%             % -- Тип 1: Простое соответствие переменной значению --
%             if strcmp(con.type, 'simple')
%                 val = P_full(con.var_idx);
%                 penalty = (val - con.target)^2;
%                 J_bc = J_bc + con.weight * penalty;
%             
%             % -- Тип 2: Геометрическая кривизна y''(x) --
%             elseif strcmp(con.type, 'y_ddx')
%                 idx = con.node_idx;
%                 
%                 % Извлекаем данные для узла idx
%                 r_val   = P_full(idx);
%                 phi_val = P_full(3*n + idx);
%                 r1_val  = P_full(n + idx);
%                 phi1_val= P_full(4*n + idx);
%                 r2_val  = P_full(2*n + idx);
%                 phi2_val= P_full(5*n + idx);
%                 
%                 % Компоненты скорости (v_r, v_phi)
%                 vr = r1_val;
%                 vp = r_val * phi1_val;
%                 
%                 % Компоненты ускорения (a_r, a_phi)
%                 ar = r2_val - r_val * (phi1_val^2);
%                 ap = r_val * phi2_val + 2 * r1_val * phi1_val;
%                 
%                 % Декартовы компоненты скорости и ускорения
%                 cos_p = cos(phi_val);
%                 sin_p = sin(phi_val);
%                 
%                 x_u = vr * cos_p - vp * sin_p; % x'
%                 y_u = vr * sin_p + vp * cos_p; % y'
%                 
%                 x_uu = ar * cos_p - ap * sin_p; % x''
%                 y_uu = ar * sin_p + ap * cos_p; % y''
%                 
%                 % Формула y''(x) = (y'' * x' - y' * x'') / (x')^3
%                 % Защита от деления на ноль (вертикальная касательная)
%                 if abs(x_u) > 1e-6
%                     y_ddx = (y_uu * x_u - y_u * x_uu) / (x_u^3);
%                 else
%                     % Если касательная вертикальна, условие y''(x) некорректно,
%                     % можно штрафовать бесконечность или просто пропустить
%                     y_ddx = 0; 
%                 end
%                 
%                 penalty = (y_ddx - con.target)^2;
%                 J_bc = J_bc + con.weight * penalty;
%             end
%         end
%     end
%     
%     % --- 6. Итоговая сумма ---
%     J = alpha * J_fid + (1 - alpha) * J_smooth + beta * J_param + gamma * J_jerk + J_bc;
% end
        function J = objective(obj, vars_opt, alpha, beta, gamma)
    % ... (начало метода: восстановление P_full, распаковка, вычисление J_fid, J_smooth, J_param, J_jerk - БЕЗ ИЗМЕНЕНИЙ) ...
    % Вставь сюда код из предыдущего ответа до блока Soft Constraints.
    
    % === Сохраняем старый код для целостности, но с улучшениями ===
    
    % --- 1. Восстановление P_full ---
    P_full = obj.map.fixed_vals; 
    P_full(obj.map.free_idx) = vars_opt;
    for k = 1:length(obj.map.links)
        link = obj.map.links(k);
        P_full(link.dep_idx) = link.C * P_full(link.prim_idx);
    end
    
    % --- 2. Распаковка ---
    n = obj.n;
    v_r   = P_full(1:n); m_r = P_full(n+1:2*n); M_r = P_full(2*n+1:3*n);
    v_phi = P_full(3*n+1:4*n); m_phi = P_full(4*n+1:5*n); M_phi = P_full(5*n+1:6*n);
    
    % --- 3. Базовые функционалы ---
    J_fid = sum((v_r - obj.r_data).^2 + v_r.^2 .* (v_phi - obj.phi_data).^2);
    J_smooth = 0; J_param = 0; J_jerk = 0;
    
    coeffs_r_prev = []; coeffs_phi_prev = []; h_prev = 0;
    
    for i = 1:n-1
        h = obj.u(i+1) - obj.u(i);
        coeffs_r = obj.geom_r.getSegmentCoeffs(v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
        coeffs_phi = obj.geom_phi.getSegmentCoeffs(v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);
        
        % Smoothness
        fun = @(t) obj.compute_segment_energy(t, h, v_r(i), m_r(i), M_r(i), coeffs_r, v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
        J_smooth = J_smooth + integral(fun, 0, h);
        
        % Parametrization
        fun_param = @(t) obj.compute_velocity_error(t, h, v_r(i), m_r(i), M_r(i), coeffs_r, v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
        J_param = J_param + integral(fun_param, 0, h);
        
        % Jerk
        if i > 1
            err_r = obj.compute_jerk_node_error(coeffs_r_prev, coeffs_r, h_prev);
            err_phi = obj.compute_jerk_node_error(coeffs_phi_prev, coeffs_phi, h_prev);
            J_jerk = J_jerk + err_r + err_phi;
        end
        coeffs_r_prev = coeffs_r; coeffs_phi_prev = coeffs_phi; h_prev = h;
    end
    
    % --- 4. Soft Constraints ---
    J_bc = 0;
    
    if isfield(obj.map, 'soft_constraints')
        constraints = obj.map.soft_constraints;
        
        for k = 1:length(constraints)
            con = constraints(k);
            
            if strcmp(con.type, 'simple')
                val = P_full(con.var_idx);
                penalty = (val - con.target)^2;
                J_bc = J_bc + con.weight * penalty;
                
            elseif strcmp(con.type, 'y_ddx')
                idx = con.node_idx;
                
                % Извлекаем данные для узла idx
                r   = P_full(idx);       phi = P_full(3*n + idx);
                r1  = P_full(n + idx);    phi1 = P_full(4*n + idx);
                r2  = P_full(2*n + idx);  phi2 = P_full(5*n + idx);
                
                % Компоненты скорости
                vr = r1;
                vp = r * phi1;
                
                % Компоненты ускорения
                ar = r2 - r * (phi1^2);
                ap = r * phi2 + 2 * r1 * phi1;
                
                % Декартовы компоненты
                cos_p = cos(phi); sin_p = sin(phi);
                
                x_u = vr * cos_p - vp * sin_p; % x'
                y_u = vr * sin_p + vp * cos_p; % y'
                
                x_uu = ar * cos_p - ap * sin_p; % x''
                y_uu = ar * sin_p + ap * cos_p; % y''
                
                % 1. Вычисляем числитель (N)
                numerator = y_uu * x_u - y_u * x_uu;
                
                % 2. Вычисляем знаменатель/масштаб (D)
                % Используем полную скорость, а не только x'
                speed_sq = x_u^2 + y_u^2;
                D = speed_sq^1.5; % (x'^2+y'^2)^1.5 = |v|^3
                
                % 3. Штраф за кривизну (Shape)
                % (N - D * target)^2
                val_shape = numerator - D * con.target;
                
                % 4. Штраф за скорость (Scale) - ТВОЕ ПРЕДЛОЖЕНИЕ
                % (D - 1)^2 заставляет |v| быть равным 1
                val_scale = D^2 - 1; 
                
                % Итоговый штраф
                % weight_shape - за геометрию (прямая линия)
                % weight_scale - за натуральную параметризацию (скорость)
                % Можно хранить в структуре, но для простоты зададим здесь:
                w_shape = con.weight;
                w_scale = con.weight; % Можно вынести в настройки или умножить на beta
                
                penalty = w_shape * val_shape^2 + w_scale * val_scale^2;
                
                J_bc = J_bc + penalty;
                
            end
        end
    end
    
    J = alpha * J_fid + (1 - alpha) * J_smooth + beta * J_param + gamma * J_jerk + J_bc;
end
        function unpack_results(obj, x_opt)
            % ... (код без изменений) ...
            n = obj.n;
            P_full = obj.map.fixed_vals; 
            P_full(obj.map.free_idx) = x_opt; 
            for k = 1:length(obj.map.links)
                link = obj.map.links(k);
                prim_val = P_full(link.prim_idx);
                P_full(link.dep_idx) = link.C * prim_val;
            end
            obj.results.r.v = P_full(1:n);
            obj.results.r.m = P_full(n+1:2*n);
            obj.results.r.M = P_full(2*n+1:3*n);
            obj.results.phi.v = P_full(3*n+1:4*n);
            obj.results.phi.m = P_full(4*n+1:5*n);
            obj.results.phi.M = P_full(5*n+1:6*n);
        end
        
        function fit(obj, alpha, beta, gamma)
            % Инициализация
            n = obj.n;
            P_init = zeros(6*n, 1);
            P_init(1:n) = obj.r_data;
            P_init(3*n+1:4*n) = obj.phi_data;
            obj.map.fixed_vals = P_init; 
            
            x0 = P_init(obj.map.free_idx);
            
            fprintf('Запуск оптимизации. Свободных переменных: %d\n', length(x0));
            options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'MaxIterations', 10000);
            
            % Передаем gamma в целевую функцию
            objFun = @(x) obj.objective(x, alpha, beta, gamma);
            
            [x_opt, f_val, flag] = fminunc(objFun, x0, options);
            fprintf('Оптимизация завершена. Код: %d\n', flag);
            
            % Сохранение результатов
            P_final = obj.map.fixed_vals;
            P_final(obj.map.free_idx) = x_opt;
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
            obj.results.x_opt = x_opt;
            obj.results.f_val = f_val;
            obj.unpack_results(x_opt);
        end
    end
end

