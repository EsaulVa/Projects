classdef Trajectory < Geometry.BaseTrajectory
    properties (Access = private)
        s_breaks   % узлы длины дуги (1 x N)
        spline_x   % сплайн x(s)
        spline_y
        spline_z
        dspline_x  % производные сплайнов (для касательной)
        dspline_y
        dspline_z
        L          % полная длина
    end
    
    methods
        function obj = Trajectory(points)
            % points - матрица 3 x N, столбцы - точки
            if size(points,1) ~= 3
                error('points must be 3xN');
            end
            N = size(points,2);
            if N < 2
                error('At least 2 points required');
            end
            
            % 1. Вычисляем длины хорд
            dist = sqrt(sum(diff(points,1,2).^2, 1));  % вектор 1 x (N-1)
            obj.s_breaks = [0, cumsum(dist)];         % длина дуги в узлах
            
            % 2. Строим сплайны x(s), y(s), z(s)
            % Используем pchip для монотонности и отсутствия осцилляций
            obj.spline_x = pchip(obj.s_breaks, points(1,:));
            obj.spline_y = pchip(obj.s_breaks, points(2,:));
            obj.spline_z = pchip(obj.s_breaks, points(3,:));
            
            % 3. Производные сплайнов для касательной
            obj.dspline_x = fnder(obj.spline_x);
            obj.dspline_y = fnder(obj.spline_y);
            obj.dspline_z = fnder(obj.spline_z);
            
            obj.L = obj.s_breaks(end);
        end
        
        function r = getPoint(obj, s)
            % Интерполяция точки по натуральному параметру s
            % s должно быть в пределах [0, L]
            if s < 0 || s > obj.L
                warning('s out of range [0, %f], clamping', obj.L);
                s = max(0, min(s, obj.L));
            end
            r = [ppval(obj.spline_x, s);
                 ppval(obj.spline_y, s);
                 ppval(obj.spline_z, s)];
        end
        
        function tau = getTangent(obj, s)
            % Касательный вектор dr/ds (нормированный)
            if s < 0 || s > obj.L
                s = max(0, min(s, obj.L));
            end
            dx = ppval(obj.dspline_x, s);
            dy = ppval(obj.dspline_y, s);
            dz = ppval(obj.dspline_z, s);
            tau = [dx; dy; dz];
            nrm = norm(tau);
            if nrm > 1e-12
                tau = tau / nrm;
            end
        end
        
        function L = totalLength(obj)
            L = obj.L;
        end
        
        function s_vals = getSValues(obj)
            % Возвращает узловые значения s (для отладки)
            s_vals = obj.s_breaks;
        end
    end
end

