classdef PolarSmoothingSpline < handle
    %POLARSMOOTHINGSPLINE Сглаживающий сплайн в полярных координатах
    %   Минимизирует функционал энергии изгиба, учитывающий связь r и phi
    
    properties
        u           % Параметр (длина хорды)
        r_data      % Исходные радиусы
        phi_data    % Исходные углы
        n           % Кол-во точек
        bc_start    % Граничные условия {r, dr, ddr, phi, dphi, ddphi}
        bc_end
        
        % Результаты
        r_opt       % Оптимизированные значения r в узлах
        phi_opt     % Оптимизированные значения phi в узлах
        coeffs_r    % Коэффициенты полиномов для r
        coeffs_phi  % Коэффициенты полиномов для phi
        m_r_opt   % Первые производные r (сохраненные)
        M_r_opt   % Вторые производные r
        m_phi_opt % Первые производные phi
        M_phi_opt % Вторые производные phi
    end
    
    methods
        function obj = PolarSmoothingSpline(x_data, y_data, bc_start, bc_end)
            % Конструктор. Принимает декартовы координаты.
            obj.n = length(x_data);
            
            % Перевод в полярные координаты (центр 0,0)
            obj.r_data = sqrt(x_data.^2 + y_data.^2);
            obj.phi_data = atan2(y_data, x_data);
            
            % Убираем разрывы по углу (делаем монотонным)
            obj.phi_data = unwrap(obj.phi_data);
            
            obj.bc_start = bc_start;
            obj.bc_end = bc_end;
            
            % Параметризация по длине хорды (в декартовом смысле)
            dists = sqrt(diff(x_data).^2 + diff(y_data).^2);
            obj.u = [0; cumsum(dists)];
        end
        
        function J = objective(obj, vars_opt, alpha)
            % Целевая функция (Связанная оптимизация)
            n = obj.n;
            
            % --- 1. Распаковка переменных ---
            % Структура вектора: [r_vals; r_m; r_M; phi_vals; phi_m; phi_M]
            idx = 1;
            
            v_r = vars_opt(idx:n); idx = idx + n;
            
            m_r = zeros(n, 1); 
            m_r(2:end-1) = vars_opt(idx:idx+n-3); idx = idx + n-2;
            m_r(1) = obj.bc_start.dr; m_r(end) = obj.bc_end.dr;
            
            M_r = zeros(n, 1);
            M_r(2:end-1) = vars_opt(idx:idx+n-3); idx = idx + n-2;
            M_r(1) = obj.bc_start.ddr; M_r(end) = obj.bc_end.ddr;
            
            v_phi = vars_opt(idx:idx+n-1); idx = idx + n;
            
            m_phi = zeros(n, 1);
            m_phi(2:end-1) = vars_opt(idx:idx+n-3); idx = idx + n-2;
            m_phi(1) = obj.bc_start.dphi; m_phi(end) = obj.bc_end.dphi;
            
            M_phi = zeros(n, 1);
            M_phi(2:end-1) = vars_opt(idx:idx+n-3); 
            M_phi(1) = obj.bc_start.ddphi; M_phi(end) = obj.bc_end.ddphi;
            
            % --- 2. Штраф за невязку (Fidelity) ---
            % Метрический штраф: ds^2 = dr^2 + r^2 dphi^2
            err_r = v_r - obj.r_data;
            err_phi = v_phi - obj.phi_data;
            J_fid = sum(err_r.^2 + (v_r.^2) .* (err_phi.^2));
            
            % --- 3. Энергия изгиба (Smoothness) ---
            % Integral [ (r'' - r*phi'^2)^2 + (r*phi'' + 2*r'*phi')^2 ] du
                       % --- 3. Энергия изгиба (Smoothness) ---
            % J = Integral [ (r'' - r*phi'^2)^2 + (r*phi'' + 2*r'*phi')^2 ] du
            
            J_smooth = 0;
            
            for i = 1:n-1
                h = obj.u(i+1) - obj.u(i);
                
                % Восстанавливаем коэффициенты полинома для сегмента i
                % ИСПРАВЛЕНО: принимаем один вектор вместо трех переменных
                coeffs_r_seg = obj.get_coeffs(v_r(i), m_r(i), M_r(i), v_r(i+1), m_r(i+1), M_r(i+1), h);
                a3r = coeffs_r_seg(1); a4r = coeffs_r_seg(2); a5r = coeffs_r_seg(3);
                
                coeffs_phi_seg = obj.get_coeffs(v_phi(i), m_phi(i), M_phi(i), v_phi(i+1), m_phi(i+1), M_phi(i+1), h);
                a3p = coeffs_phi_seg(1); a4p = coeffs_phi_seg(2); a5p = coeffs_phi_seg(3);
                
                % Подынтегральная функция
                fun = @(t) obj.segment_energy(t, h, ...
                    v_r(i), m_r(i), M_r(i), a3r, a4r, a5r, ...
                    v_phi(i), m_phi(i), M_phi(i), a3p, a4p, a5p);
                
                J_smooth = J_smooth + integral(fun, 0, h);
            end
            
            J = alpha * J_fid + (1 - alpha) * J_smooth;
        end
        
        function val = segment_energy(obj, t, h, vr, mr, Mr, a3r, a4r, a5r, vp, mp, Mp, a3p, a4p, a5p)
            % Вычисление квадрата ускорения в точке t
            
            % R и производные
            rt  = vr + mr*t + 0.5*Mr*t.^2 + a3r*t.^3 + a4r*t.^4 + a5r*t.^5;
            r1  = mr + Mr*t + 3*a3r*t.^2 + 4*a4r*t.^3 + 5*a5r*t.^4;
            r2  = Mr + 6*a3r*t + 12*a4r*t.^2 + 20*a5r*t.^3;
            
            % Phi и производные
            p1  = mp + Mp*t + 3*a3p*t.^2 + 4*a4p*t.^3 + 5*a5p*t.^4;
            p2  = Mp + 6*a3p*t + 12*a4p*t.^2 + 20*a5p*t.^3;
            
            % Полярные компоненты ускорения
            acc_r = r2 - rt .* (p1.^2);
            acc_phi = rt .* p2 + 2 .* r1 .* p1;
            
            val = acc_r.^2 + acc_phi.^2;
        end
        
%         function [a3, a4, a5] = get_coeffs(obj, yi, mi, Mi, yip1, mip1, Mip1, h)
%             % Решение СЛАУ для коэффициентов (из статьи)
%             B = [yip1 - (yi + mi*h + 0.5*Mi*h^2);
%                  mip1 - (mi + Mi*h);
%                  Mip1 - Mi];
%             A = [h^3, h^4, h^5;
%                  3*h^2, 4*h^3, 5*h^4;
%                  6*h, 12*h^2, 20*h^3];
%             coeffs = A \ B;
%             a3 = coeffs(1); a4 = coeffs(2); a5 = coeffs(3);
%         end
        
        function coeffs = get_coeffs(obj, yi, mi, Mi, yip1, mip1, Mip1, h)
            % Решение СЛАУ для коэффициентов
            % Возвращает вектор [a3; a4; a5]

            B = [yip1 - (yi + mi*h + 0.5*Mi*h^2);
                 mip1 - (mi + Mi*h);
                 Mip1 - Mi];

            A = [h^3, h^4, h^5;
                 3*h^2, 4*h^3, 5*h^4;
                 6*h, 12*h^2, 20*h^3];

            coeffs = A \ B; % Вернет вектор 3x1
        end
        function fit(obj, alpha)
            % Запуск оптимизации
            n = obj.n;
            
            % Начальное приближение (просто данные)
            v_r_init = obj.r_data;
            v_phi_init = obj.phi_data;
            
            % Инициализация производных (конечные разности)
            dr_init = [0; diff(obj.r_data)./diff(obj.u); 0];
            ddr_init = [0; diff(dr_init); 0];
            dphi_init = [0; diff(obj.phi_data)./diff(obj.u); 0];
            ddphi_init = [0; diff(dphi_init); 0];
            
            % Сборка вектора x0
            % [r_vals(n), r_m(n-2), r_M(n-2), phi_vals(n), phi_m(n-2), phi_M(n-2)]
            x0 = [v_r_init; dr_init(2:end-1); ddr_init(2:end-1); ...
                  v_phi_init; dphi_init(2:end-1); ddphi_init(2:end-1)];
            
            fprintf('Запуск оптимизации (размерность %d)...\n', length(x0));
            options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
            
            objFun = @(x) obj.objective(x, alpha);
            [x_opt, ~, exitflag] = fminunc(objFun, x0, options);
            fprintf('Оптимизация завершена. Код выхода: %d\n', exitflag);
            
            % Распаковка результатов и сохранение
            obj.unpack_results(x_opt);
        end
       function unpack_results(obj, x_opt)
            n = obj.n;
            idx = 1;

            % --- R ---
            obj.r_opt = x_opt(idx:n); idx = idx + n;

            obj.m_r_opt = zeros(n, 1); 
            obj.M_r_opt = zeros(n, 1);

            obj.m_r_opt(2:end-1) = x_opt(idx:idx+n-3); idx = idx + n-2;
            obj.M_r_opt(2:end-1) = x_opt(idx:idx+n-3); idx = idx + n-2;

            % Граничные условия (берем из исходных данных)
            obj.m_r_opt([1, end]) = [obj.bc_start.dr, obj.bc_end.dr];
            obj.M_r_opt([1, end]) = [obj.bc_start.ddr, obj.bc_end.ddr];

            % --- Phi ---
            obj.phi_opt = x_opt(idx:idx+n-1); idx = idx + n;

            obj.m_phi_opt = zeros(n, 1); 
            obj.M_phi_opt = zeros(n, 1);

            obj.m_phi_opt(2:end-1) = x_opt(idx:idx+n-3); idx = idx + n-2;
            obj.M_phi_opt(2:end-1) = x_opt(idx:idx+n-3);

            obj.m_phi_opt([1, end]) = [obj.bc_start.dphi, obj.bc_end.dphi];
            obj.M_phi_opt([1, end]) = [obj.bc_start.ddphi, obj.bc_end.ddphi];

            % Расчет коэффициентов сегментов (как было)
            obj.coeffs_r = cell(n-1, 1);
            obj.coeffs_phi = cell(n-1, 1);
            for i = 1:n-1
                h = obj.u(i+1) - obj.u(i);
                obj.coeffs_r{i} = obj.get_coeffs(obj.r_opt(i), obj.m_r_opt(i), obj.M_r_opt(i), ...
                    obj.r_opt(i+1), obj.m_r_opt(i+1), obj.M_r_opt(i+1), h);
                obj.coeffs_phi{i} = obj.get_coeffs(obj.phi_opt(i), obj.m_phi_opt(i), obj.M_phi_opt(i), ...
                    obj.phi_opt(i+1), obj.m_phi_opt(i+1), obj.M_phi_opt(i+1), h);
            end
       end
       
       function [val_r, val_phi] = predict(obj, u_eval)
            u_eval = u_eval(:);
            val_r = zeros(size(u_eval));
            val_phi = zeros(size(u_eval));

            for k = 1:length(u_eval)
                u = u_eval(k);
                % Поиск сегмента
                idx = find(obj.u <= u, 1, 'last');
                if isempty(idx), idx = 1; end
                if idx > obj.n-1, idx = obj.n-1; end

                t = u - obj.u(idx);

                % --- Расчет R ---
                % Берем данные из сохраненных свойств
                yi_r = obj.r_opt(idx);
                mi_r = obj.m_r_opt(idx);
                Mi_r = obj.M_r_opt(idx);
                coeffs = obj.coeffs_r{idx};
                a3 = coeffs(1); a4 = coeffs(2); a5 = coeffs(3);

                % Формула (3) из статьи
                val_r(k) = yi_r + mi_r*t + 0.5*Mi_r*t^2 + a3*t^3 + a4*t^4 + a5*t^5;

                % --- Расчет Phi ---
                yi_p = obj.phi_opt(idx);
                mi_p = obj.m_phi_opt(idx);
                Mi_p = obj.M_phi_opt(idx);
                coeffs = obj.coeffs_phi{idx};
                a3 = coeffs(1); a4 = coeffs(2); a5 = coeffs(3);

                val_phi(k) = yi_p + mi_p*t + 0.5*Mi_p*t^2 + a3*t^3 + a4*t^4 + a5*t^5;
            end
        end
        
        
%         function unpack_results(obj, x_opt)
%             % Восстановление полных массивов из вектора оптимизации
%             n = obj.n;
%             idx = 1;
%             
%             obj.r_opt = x_opt(idx:n); idx = idx + n;
%             
%             m_r = zeros(n, 1); M_r = zeros(n, 1);
%             m_r(2:end-1) = x_opt(idx:idx+n-3); idx = idx + n-2;
%             M_r(2:end-1) = x_opt(idx:idx+n-3); idx = idx + n-2;
%             m_r([1, end]) = [obj.bc_start.dr, obj.bc_end.dr];
%             M_r([1, end]) = [obj.bc_start.ddr, obj.bc_end.ddr];
%             
%             obj.phi_opt = x_opt(idx:idx+n-1); idx = idx + n;
%             
%             m_phi = zeros(n, 1); M_phi = zeros(n, 1);
%             m_phi(2:end-1) = x_opt(idx:idx+n-3); idx = idx + n-2;
%             M_phi(2:end-1) = x_opt(idx:idx+n-3);
%             m_phi([1, end]) = [obj.bc_start.dphi, obj.bc_end.dphi];
%             M_phi([1, end]) = [obj.bc_start.ddphi, obj.bc_end.ddphi];
%             
%             % Вычисляем коэффициенты для всех сегментов
%             obj.coeffs_r = cell(n-1, 1);
%             obj.coeffs_phi = cell(n-1, 1);
%             for i = 1:n-1
%                 h = obj.u(i+1) - obj.u(i);
%                 obj.coeffs_r{i} = obj.get_coeffs(obj.r_opt(i), m_r(i), M_r(i), obj.r_opt(i+1), m_r(i+1), M_r(i+1), h);
%                 obj.coeffs_phi{i} = obj.get_coeffs(obj.phi_opt(i), m_phi(i), M_phi(i), obj.phi_opt(i+1), m_phi(i+1), M_phi(i+1), h);
%             end
%         end
        
        
%         function [val_r, val_phi, der] = predict(obj, u_eval, der_order)
%             % Предсказание значений и производных по u
%             if nargin < 3, der_order = 0; end
%             
%             u_eval = u_eval(:);
%             val_r = zeros(size(u_eval));
%             val_phi = zeros(size(u_eval));
%             der = struct('r1', zeros(size(u_eval)), 'r2', zeros(size(u_eval)), ...
%                          'phi1', zeros(size(u_eval)), 'phi2', zeros(size(u_eval)));
%             
%             for k = 1:length(u_eval)
%                 u = u_eval(k);
%                 % Поиск сегмента
%                 idx = find(obj.u <= u, 1, 'last');
%                 if isempty(idx), idx = 1; end
%                 if idx > obj.n-1, idx = obj.n-1; end
%                 
%                 t = u - obj.u(idx);
%                 h = obj.u(idx+1) - obj.u(idx);
%                 
%                 % R
%                 cr = obj.coeffs_r{idx};
%                 val_r(k) = obj.eval_poly(t, obj.r_opt(idx), obj.r_opt(idx+1), cr(1), cr(2), cr(3), h, der_order, 'r', der, k);
%                 
%                 % Phi
%                 cp = obj.coeffs_phi{idx};
%                 val_phi(k) = obj.eval_poly(t, obj.phi_opt(idx), obj.phi_opt(idx+1), cp(1), cp(2), cp(3), h, der_order, 'phi', der, k);
%             end
%         end
        
        function val = eval_poly(~, t, yi, a3, a4, a5, h, der_order, type, der, k)
            % Вспомогательная функция для вычисления полинома и производных
            % (упрощенная запись без хранения mi, Mi для компактности используем пересчет)
            % Здесь мы используем форму (3) из статьи, но для простоты берем сохраненные коэфф.
            
            % Для корректного вычисления производных в точке t, нам нужны mi, Mi.
            % Но в predict они не сохранены. 
            % Проще вычислить значения напрямую через полином:
            % S(t) = yi + mi*t + Mi/2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            % Производные:
            % S'  = mi + Mi*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
            % S'' = Mi + 6*a3*t + 12*a4*t^2 + 20*a5*t^3
            % mi и Mi мы потеряли при передаче... 
            % ИСПРАВЛЕНИЕ: метод predict должен быть полным. 
            % Но чтобы не раздувать код, сделаем простой расчет значений (der=0).
            % Если нужны производные, ниже используется измененная логика.
            val = 0; % placeholder
        end
    end
end

