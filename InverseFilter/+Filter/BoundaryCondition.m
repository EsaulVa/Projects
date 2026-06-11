classdef BoundaryCondition < handle
    %BOUNDARYCONDITION Граничное условие для диффузии на конце меридиана.
    %   Дирихле  : h = value
    %   Неймана  : dh/ds = flux
    %   Робин    : a*h + b*dh/dn = value
    %
    %   Пример:
    %      bc = BoundaryCondition.dirichlet(50.0);
    %      bc = BoundaryCondition.neumann(0.0);
    %      bc = BoundaryCondition.robin(1.0, 0.5, 50.0);

    properties
        kind    % 'dirichlet', 'neumann', 'robin'
        value   % числовое значение или правая часть
        coeff_a % коэффициент при h (для Robin)
        coeff_b % коэффициент при dh/dn (для Robin)
    end

    methods (Static)
        function bc = dirichlet(val)
            %DIRICHLET h = val
            bc = Filter.BoundaryCondition('dirichlet', val, 1.0, 0.0);
        end

        function bc = neumann(flux)
            %NEUMANN dh/ds = flux
            bc = Filter.BoundaryCondition('neumann', flux, 1.0, 0.0);
        end

        function bc = robin(a, b, val)
            %ROBIN a*h + b*dh/dn = val
            bc = Filter.BoundaryCondition('robin', val, a, b);
        end
    end

    methods
        function obj = BoundaryCondition(kind, value, coeff_a, coeff_b)
            obj.kind = kind;
            obj.value = value;
            obj.coeff_a = coeff_a;
            obj.coeff_b = coeff_b;
        end

        function bc = with_value(obj, new_val)
            %WITH_VALUE Копия с другим значением (для автоподстановки Дирихле).
            bc = Filter.BoundaryCondition(obj.kind, new_val, obj.coeff_a, obj.coeff_b);
        end
    end
end
