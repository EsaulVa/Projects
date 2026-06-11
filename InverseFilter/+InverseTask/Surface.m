classdef Surface < handle
    % Абстрактный класс поверхности
    methods (Abstract)
        % Параметрическое отображение: r = [x;y;z]
        r = getPoint(obj, u, v)
        % Первые производные: ru, rv
        [ru, rv] = getFirstDerivatives(obj, u, v)
        % Вторые производные: ruu, ruv, rvv
        [ruu, ruv, rvv] = getSecondDerivatives(obj, u, v)
    end
    
    methods
        % Нормаль к поверхности (единичная)
        function n = getNormal(obj, u, v)
            [ru, rv] = obj.getFirstDerivatives(u, v);
            n = cross(ru, rv);
            n = n / norm(n);
        end
        
        % Коэффициенты первой фундаментальной формы: E, F, G
        function [E, F, G] = getFirstFundamental(obj, u, v)
            [ru, rv] = obj.getFirstDerivatives(u, v);
            E = dot(ru, ru);
            F = dot(ru, rv);
            G = dot(rv, rv);
        end
        
        % Коэффициенты второй фундаментальной формы: L, M, N
        function [L, M, N] = getSecondFundamental(obj, u, v)
            [ru, rv] = obj.getFirstDerivatives(u, v);
            [ruu, ruv, rvv] = obj.getSecondDerivatives(u, v);
            n = obj.getNormal(u, v);
            L = dot(ruu, n);
            M = dot(ruv, n);
            N = dot(rvv, n);
        end
    end
end

