classdef (Abstract) Surface < handle
    % SURFACE - абстрактная поверхность в 3D пространстве
    % Единый интерфейс для всех типов поверхностей (вращения, параметрических и др.)
    
    %=== АБСТРАКТНЫЕ СВОЙСТВА (должны быть инициализированы в подклассах) ===
    properties (Abstract)
        type          % строка: 'revolution', 'parametric_revolution', 'cylinder' и т.д.
        domain        % [2?2] матрица: [u_min, u_max; v_min, v_max] — область определения
    end
    
    %=== АБСТРАКТНЫЕ МЕТОДЫ ===
    methods (Abstract)
        % Основные геометрические операции
        r = position(obj, u, v)          % радиус-вектор точки на поверхности
        ru = partial_u(obj, u, v)        % ?r/?u — частная производная по первому параметру
        rv = partial_v(obj, u, v)        % ?r/?v — частная производная по второму параметру
        ruu = partial_uu(obj, u, v)      % ??r/?u?
        ruv = partial_uv(obj, u, v)      % ??r/?u?v
        rvv = partial_vv(obj, u, v)      % ??r/?v?
        normal = normal_vector(obj, u, v) % единичная нормаль к поверхности
        
        % Метрические характеристики
        [E, F, G] = first_form(obj, u, v)    % коэффициенты первой квадратичной формы
        [L, M, N] = second_form(obj, u, v)   % коэффициенты второй квадратичной формы
        
        % Преобразования координат
        [u, v] = cartesian_to_surface(obj, x, y, z) % декартовы ? криволинейные
        [x, y, z] = surface_to_cartesian(obj, u, v) % криволинейные ? декартовы
    end
    
    %=== КОНКРЕТНЫЕ МЕТОДЫ (общие для всех поверхностей) ===
    methods
        function kappa = normal_curvature(obj, u, v, du, dv)
            % Кривизна нормального сечения в направлении (du, dv)
            % Формула: k_n = (L du? + 2M du dv + N dv?) / (E du? + 2F du dv + G dv?)
            [E, F, G] = obj.first_form(u, v);
            [L, M, N] = obj.second_form(u, v);
            
            numerator = L*du^2 + 2*M*du*dv + N*dv^2;
            denominator = E*du^2 + 2*F*du*dv + G*dv^2;
            
            if abs(denominator) < eps
                kappa = NaN;
            else
                kappa = numerator / denominator;
            end
        end
        
        function ds = arc_length_element(obj, u, v, du, dv)
            % Элемент длины дуги: ds = sqrt(E du? + 2F du dv + G dv?)
            [E, F, G] = obj.first_form(u, v);
            ds = sqrt(E*du^2 + 2*F*du*dv + G*dv^2);
        end
    end
end