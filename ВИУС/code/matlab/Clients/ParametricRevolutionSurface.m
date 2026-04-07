claclassdef ParametricRevolutionSurface < RevolutionSurface
    % PARAMETRICREVOLUTIONSURFACE - поверхность вращения 
    % с ПАРАМЕТРИЧЕСКИ заданной образующей через квинтический сплайн
    % Образующая: ?(?) = (?(?), z(?)), где ? — хордовый параметр
    
    %=== СВОЙСТВА (ТОЛЬКО НОВЫЕ — унаследованные НЕ объявляются повторно!) ===
    properties (Access = protected)
        spline_profile                % ParametricQuinticSpline — образующая ?(?) = (?(?), z(?))
        chordal_parameter_range       % [?_min, ?_max] — диапазон хордового параметра
    end
    
    %=== КОНСТРУКТОР ===
    methods
        function obj = ParametricRevolutionSurface(points, bc_start, bc_end, axis_dir)
            % points: [N?2] — точки образующей (?, z)
            % bc_start, bc_end — граничные условия для сплайна
            % axis_dir (опционально): направление оси вращения
            
            % 1. СОЗДАЁМ ВРЕМЕННУЮ ПОВЕРХНОСТЬ ДЛЯ ИНИЦИАЛИЗАЦИИ СУПЕРКЛАССА
            % Используем заглушку: цилиндр радиуса 1 на отрезке [0, 1]
            dummy_profile = @(z) 1.0;  % постоянный радиус = 1
            z_range_temp = [0, 1];
            
            if nargin < 4 || isempty(axis_dir)
                axis_dir = [0; 0; 1];
            end
            
            % ВАЖНО: сначала вызываем конструктор суперкласса
            obj@RevolutionSurface(dummy_profile, z_range_temp, axis_dir);
            
            % 2. ПЕРЕОПРЕДЕЛЯЕМ АБСТРАКТНЫЕ СВОЙСТВА ЧЕРЕЗ ПРИСВАИВАНИЕ (НЕ объявление!)
            obj.type = 'parametric_revolution';  % ? правильно: присваивание, а не объявление
            
            % 3. СОЗДАЁМ ПАРАМЕТРИЧЕСКИЙ СПЛАЙН ДЛЯ ОБРАЗУЮЩЕЙ
            obj.spline_profile = ParametricQuinticSpline(points, bc_start, bc_end);
            obj.chordal_parameter_range = [obj.spline_profile.u(1), obj.spline_profile.u(end)];
            
            % 4. ОБНОВЛЯЕМ ДОМЕН ПОВЕРХНОСТИ НА ОСНОВЕ СПЛАЙНА
            obj.domain = [obj.chordal_parameter_range(1), obj.chordal_parameter_range(2); ...
                          0, 2*pi];
            
            % 5. ПЕРЕОПРЕДЕЛЯЕМ ФУНКЦИИ ОБРАЗУЮЩЕЙ ЧЕРЕЗ ОБЁРТКИ НАД СПЛАЙНОМ
            % Теперь все геометрические методы будут использовать сплайн вместо заглушки
        end
    end
    
    %=== ПЕРЕОПРЕДЕЛЁННЫЕ ГЕОМЕТРИЧЕСКИЕ МЕТОДЫ (учитывают параметрическую природу ?) ===
    methods
        function r = position(obj, tau, v)
            % Радиус-вектор точки на параметрической поверхности вращения
            % ? — хордовый параметр образующей (НЕ координата по оси!)
            profile_point = obj.spline_profile.predict(tau, 0);
            rho = profile_point(1);  % радиальная координата ?(?)
            z = profile_point(2);    % координата по оси z(?)
            r = [rho * cos(v); rho * sin(v); z];
        end
        
        function ru = partial_u(obj, tau, v)
            % ?r/?? — производная по хордовому параметру образующей
            % ВАЖНО: это НЕ ?r/?z! ? и z связаны через сплайн: z = z(?)
            profile_deriv = obj.spline_profile.predict(tau, 1); % [d?/d?, dz/d?]
            rho_p = profile_deriv(1);
            z_p = profile_deriv(2);
            ru = [rho_p * cos(v); rho_p * sin(v); z_p];
        end
        
        function rv = partial_v(obj, tau, v)
            % ?r/?v — производная по азимутальному углу (не меняется)
            profile_point = obj.spline_profile.predict(tau, 0);
            rho = profile_point(1);
            rv = [-rho * sin(v); rho * cos(v); 0];
        end
        
        function ruu = partial_uu(obj, tau, v)
            % ??r/???
            profile_deriv2 = obj.spline_profile.predict(tau, 2); % [d??/d??, d?z/d??]
            rho_pp = profile_deriv2(1);
            z_pp = profile_deriv2(2);
            ruu = [rho_pp * cos(v); rho_pp * sin(v); z_pp];
        end
        
        function ruv = partial_uv(obj, tau, v)
            % ??r/???v
            profile_deriv = obj.spline_profile.predict(tau, 1);
            rho_p = profile_deriv(1);
            ruv = [-rho_p * sin(v); rho_p * cos(v); 0];
        end
        
        function rvv = partial_vv(obj, tau, v)
            % ??r/?v? (не меняется)
            profile_point = obj.spline_profile.predict(tau, 0);
            rho = profile_point(1);
            rvv = [-rho * cos(v); -rho * sin(v); 0];
        end
        
        function [E, F, G] = first_form(obj, tau, v)
            % Ключевая особенность: коэффициент E ? 1 в общем случае!
            % Для хордовой параметризации: ||?'(?)|| ? 1, но не точно
            profile_deriv = obj.spline_profile.predict(tau, 1);
            rho_p = profile_deriv(1);
            z_p = profile_deriv(2);
            rho = obj.spline_profile.predict(tau, 0)(1);
            
            E = rho_p^2 + z_p^2;  % ? 1 — хордовая параметризация НЕ натуральная!
            F = 0;                % ортогональность сохраняется (сетка (?, v) ортогональна)
            G = rho^2;
        end
        
        function [z_out, v_out] = cartesian_to_surface(obj, x, y, z_in)
            % Преобразование декартовых ? криволинейных координат
            % 1. Находим азимутальный угол
            v_out = atan2(y, x);
            v_out = mod(v_out, 2*pi);
            
            % 2. Находим радиус в сечении
            rho_section = sqrt(x^2 + y^2);
            
            % 3. Находим параметр ? методом ближайшей точки на сплайне
            % (простая реализация — можно улучшить через оптимизацию)
            u_samples = linspace(obj.chordal_parameter_range(1), ...
                                obj.chordal_parameter_range(2), 1000);
            profile_points = obj.spline_profile.predict(u_samples, 0);
            distances = sqrt((profile_points(:,1) - rho_section).^2 + ...
                           (profile_points(:,2) - z_in).^2);
            [~, idx] = min(distances);
            z_out = u_samples(idx);  % ? как "координата" вдоль образующей
        end
        
        function [x, y, z_out] = surface_to_cartesian(obj, tau, v)
            % Преобразование криволинейных ? декартовых координат
            profile_point = obj.spline_profile.predict(tau, 0);
            rho = profile_point(1);
            z_out = profile_point(2);
            x = rho * cos(v);
            y = rho * sin(v);
        end
    end
    
    %=== СПЕЦИАЛИЗИРОВАННЫЕ МЕТОДЫ ДЛЯ ПАРАМЕТРИЧЕСКОЙ ПОВЕРХНОСТИ ===
    methods
        function fit(obj, alpha)
            % Оптимизация сплайна образующей
            % Вызывает метод fit() из ParametricQuinticSpline
            obj.spline_profile.fit(alpha);
        end
        
        function is_natural = is_natural_parameterization(obj, tolerance)
            % Проверка: близка ли хордовая параметризация к натуральной?
            % Условие: ||?'(?)|| ? 1 для всех ?
            if nargin < 2
                tolerance = 0.05;  % 5% допуск
            end
            
            tau_samples = linspace(obj.chordal_parameter_range(1), ...
                                  obj.chordal_parameter_range(2), 100);
            speeds = zeros(size(tau_samples));
            
            for i = 1:length(tau_samples)
                deriv = obj.spline_profile.predict(tau_samples(i), 1);
                speeds(i) = norm(deriv);  % ||?'(?)||
            end
            
            max_deviation = max(abs(speeds - 1));
            is_natural = (max_deviation < tolerance);
            
            if nargout == 0
                fprintf('Параметризация %sнатуральной (макс. отклонение = %.3f%%)\n', ...
                    ternary(is_natural, '', 'НЕ '), max_deviation*100);
            end
        end
        
        function L = profile_length(obj, n_samples)
            % Длина образующей (дуга сплайна ?(?))
            if nargin < 2
                n_samples = 1000;
            end
            
            tau_fine = linspace(obj.chordal_parameter_range(1), ...
                               obj.chordal_parameter_range(2), n_samples);
            points = obj.spline_profile.predict(tau_fine, 0);
            diffs = diff(points, 1);
            L = sum(sqrt(sum(diffs.^2, 2)));
        end
        
        function kappa_max = max_normal_curvature(obj, n_samples)
            % Максимальная кривизна нормального сечения по образующей
            if nargin < 2
                n_samples = 100;
            end
            
            tau_samples = linspace(obj.chordal_parameter_range(1), ...
                                  obj.chordal_parameter_range(2), n_samples);
            v_samples = linspace(0, 2*pi, 10);
            
            kappa_vals = [];
            for i = 1:length(tau_samples)
                for j = 1:length(v_samples)
                    % Направление вдоль образующей: (du, dv) = (1, 0)
                    kappa = obj.normal_curvature(tau_samples(i), v_samples(j), 1, 0);
                    kappa_vals = [kappa_vals; abs(kappa)];
                end
            end
            
            kappa_max = max(kappa_vals);
        end
    end
end

% Вспомогательная функция для тернарного оператора
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end