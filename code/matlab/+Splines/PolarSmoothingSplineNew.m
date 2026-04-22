classdef PolarSmoothingSplineNew < handle
    %POLARSMOOTHINGSPLINE Сглаживающий сплайн в полярных координатах.
    % Использует QuinticSplineGeometry для вычислений.
    
    properties
        u           % Параметр (длина хорды)
        r_data      % Исходные данные (радиус)
        phi_data    % Исходные данные (угол)
        n           % Кол-во точек
        
        % Геометрические движки
        geom_r
        geom_phi
        
        % Результаты оптимизации
        results     % Struct: .r (.v, .m, .M), .phi (.v, .m, .M)
        
        % Карта оптимизации
        map         % Struct: .free_idx, .fixed_idx, .fixed_vals
    end
    
    methods
        function obj = PolarSmoothingSplineNew(x, y)
            % Конструктор: преобразует декартовы координаты в полярные
            obj.n = length(x);
            obj.geom_r = Splines.QuinticSplineGeometry();
            obj.geom_phi = Splines.QuinticSplineGeometry();
            
            % Параметризация (хордальная длина)
            dists = sqrt(diff(x).^2 + diff(y).^2);
            obj.u = [0; cumsum(dists)];
            
            % Данные
            obj.r_data = sqrt(x.^2 + y.^2);
            obj.phi_data = unwrap(atan2(y, x)); % Разрываем фазу
        end
        
        function setBC(obj, bc_start, bc_end)
            % Формирование карты переменных (Fixed/Free)
            % Структура полного вектора состояния P (размер 6*n):
            % [r_vals(n), r_m(n), r_M(n), phi_vals(n), phi_m(n), phi_M(n)]
            
            n = obj.n;
            total_size = 6 * n;
            is_fixed = false(total_size, 1);
            fixed_vals = zeros(total_size, 1);
            
            % Вспомогательная функция для установки ГУ
            function set_boundary(node_idx, bc, prefix, shift)
                % Индексы в полном векторе
                idx_v = shift + node_idx;
                idx_m = shift + n + node_idx;
                idx_M = shift + 2*n + node_idx;
                
                if isfield(bc, prefix)
                    spec = bc.(prefix);
                    if isfield(spec, 'value') && ~isnan(spec.value)
                        is_fixed(idx_v) = true; fixed_vals(idx_v) = spec.value;
                    end
                    if isfield(spec, 'deriv1') && ~isnan(spec.deriv1)
                        is_fixed(idx_m) = true; fixed_vals(idx_m) = spec.deriv1;
                    end
                    if isfield(spec, 'deriv2') && ~isnan(spec.deriv2)
                        is_fixed(idx_M) = true; fixed_vals(idx_M) = spec.deriv2;
                    end
                end
            end
            
            % Сдвиги для r (0) и phi (3n)
            shift_r = 0;
            shift_phi = 3 * n;
            
            % Установка границ
            set_boundary(1, bc_start, 'r', shift_r);
            set_boundary(1, bc_start, 'phi', shift_phi);
            set_boundary(n, bc_end, 'r', shift_r);
            set_boundary(n, bc_end, 'phi', shift_phi);
            
            % Сохраняем карту
            obj.map.free_idx = find(~is_fixed);
            obj.map.fixed_idx = find(is_fixed);
            obj.map.fixed_vals = fixed_vals(is_fixed);
            obj.map.full_init = fixed_vals; % Для инициализации
        end
        
        function J = objective(obj, vars_opt, alpha)
            % 1. Сборка полного вектора состояния
            P_full = obj.map.full_init;
            P_full(obj.map.free_idx) = vars_opt;
            
            n = obj.n;
            % Распаковка R
            v_r = P_full(1:n);
            m_r = P_full(n+1:2*n);
            M_r = P_full(2*n+1:3*n);
            % Распаковка Phi
            v_phi = P_full(3*n+1:4*n);
            m_phi = P_full(4*n+1:5*n);
            M_phi = P_full(5*n+1:6*n);
            
            % 2. Fidelity Term (Метрический штраф)
            J_fid = sum((v_r - obj.r_data).^2 + v_r.^2 .* (v_phi - obj.phi_data).^2);
            
            % 3. Smoothness Term (Полярная энергия)
            J_smooth = 0;
            
            for i = 1:n-1
                h = obj.u(i+1) - obj.u(i);
                
                % Получаем коэффициенты через геометрический движок
                coeffs_r = obj.geom_r.getSegmentCoeffs(...
                    v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
                    
                coeffs_phi = obj.geom_phi.getSegmentCoeffs(...
                    v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);
                
                % Подынтегральная функция
                fun = @(t) obj.compute_segment_energy(t, h, ...
                    v_r(i), m_r(i), M_r(i), coeffs_r, ...
                    v_phi(i), m_phi(i), M_phi(i), coeffs_phi);
                
                J_smooth = J_smooth + integral(fun, 0, h);
            end
            
            J = alpha * J_fid + (1 - alpha) * J_smooth;
        end
        
        function val = compute_segment_energy(obj, t, h, vr, mr, Mr, cr, vp, mp, Mp, cp)
            % Вычисление энергии в точке t через движок
            r   = obj.geom_r.evalValue(t, vr, mr, Mr, cr);
            r1  = obj.geom_r.evalDeriv1(t, mr, Mr, cr);
            r2  = obj.geom_r.evalDeriv2(t, Mr, cr);
            
            phi1 = obj.geom_phi.evalDeriv1(t, mp, Mp, cp);
            phi2 = obj.geom_phi.evalDeriv2(t, Mp, cp);
            
            % Кинематика полярных координат
            acc_r = r2 - r .* (phi1.^2);
            acc_phi = r .* phi2 + 2 .* r1 .* phi1;
            
            val = acc_r.^2 + acc_phi.^2;
        end
        
        function fit(obj, alpha)
            % Инициализация начального приближения (просто данные)
            n = obj.n;
            P_init = zeros(6*n, 1);
            
            % Заполняем данными
            P_init(1:n) = obj.r_data;
            P_init(3*n+1:4*n) = obj.phi_data;
            
            % Производные можно оценить грубо или занулить
            % Здесь используем нули для надежности (если BC свободны)
            
            % Обновляем карту инициализации данными
            obj.map.full_init = P_init;
            % Восстанавливаем фиксированные значения (переписываем данные тем, что задал юзер)
            obj.map.full_init(obj.map.fixed_idx) = obj.map.fixed_vals;
            
            % Формируем x0
            x0 = obj.map.full_init(obj.map.free_idx);
            
            fprintf('Запуск оптимизации. Переменных: %d\n', length(x0));
            options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter');
            objFun = @(x) obj.objective(x, alpha);
            
            [x_opt, ~, flag] = fminunc(objFun, x0, options);
            fprintf('Оптимизация завершена. Код: %d\n', flag);
            
            % Сохранение результатов
            P_final = obj.map.full_init;
            P_final(obj.map.free_idx) = x_opt;
            
            obj.results.r.v = P_final(1:n);
            obj.results.r.m = P_final(n+1:2*n);
            obj.results.r.M = P_final(2*n+1:3*n);
            
            obj.results.phi.v = P_final(3*n+1:4*n);
            obj.results.phi.m = P_final(4*n+1:5*n);
            obj.results.phi.M = P_final(5*n+1:6*n);
        end
    end
end

