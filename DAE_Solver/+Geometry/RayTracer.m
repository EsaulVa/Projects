classdef RayTracer < handle
    % Трассировщик лучей от линии укладки до внешней поверхности
    properties (Access = private)
        mandrel      % Surface (внутренняя оправка, E2)
        safetySurf   % Surface (внешняя поверхность, E1)
        layerTraj    % Trajectory (линия укладки на mandrel, параметризованная s)
    end
    
    methods
        function obj = RayTracer(mandrel, safetySurf, layerTraj)
            % mandrel, safetySurf - объекты классов, наследующих Surface
            % layerTraj - объект Trajectory (линия укладки на mandrel)
            obj.mandrel = mandrel;
            obj.safetySurf = safetySurf;
            obj.layerTraj = layerTraj;
        end
        
        function [R_traj, R_points, z_values] = trace(obj, step_s)
            % step_s - шаг по натуральному параметру s линии укладки
            % Возвращает:
            %   R_traj   - объект ChordalTrajectory для R(z)
            %   R_points - массив точек 3xN (декартовы)
            %   z_values - массив длин дуги z (натуральный параметр R)
            
            L_total = obj.layerTraj.totalLength();
            s_vals = 0:step_s:L_total;
            N = length(s_vals);
            R_points = zeros(3, N);
            
            for i = 1:N
                s = s_vals(i);
                P = obj.layerTraj.getPoint(s);      % точка на линии укладки
                tau = obj.layerTraj.getTangent(s);  % единичный касательный
                
                % Ищем пересечение луча P + t*tau с safetySurf
                t = obj.intersectEllipsoid(P, tau);
                if isempty(t) || isnan(t)
                    warning('No intersection at s = %.3f', s);
                    R_points(:,i) = NaN;
                else
                    R_points(:,i) = P + t * tau;
                end
            end
            
            % Удаляем точки, где пересечения не найдены
            valid = ~isnan(R_points(1,:));
            R_points = R_points(:, valid);
            
            % Строим траекторию R(z) как ChordalTrajectory
            if size(R_points,2) < 2
                error('Not enough valid intersection points');
            end
            R_traj = Trajectory(R_points);
            % Получаем натуральный параметр z (длины дуги) от R_traj
            z_values = linspace(0, R_traj.totalLength(), size(R_points,2));
        end
    end
    
    methods (Access = private)
        function t = intersectEllipsoid(obj, P, tau)
            % Находит наименьшее положительное t пересечения луча P + t*tau
            % с поверхностью safetySurf (которая является эллипсоидом).
            % Уравнение эллипсоида: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1.
            % Для обобщения на любую поверхность потребовался бы решатель,
            % но здесь safetySurf - Ellipsoid.
            
            % Получаем параметры эллипсоида (предполагаем, что safetySurf - Ellipsoid)
            if ~isa(obj.safetySurf, 'Ellipsoid')
                error('safetySurf должен быть Ellipsoid для данного метода');
            end
            a = obj.safetySurf.a;
            b = obj.safetySurf.b;
            c = obj.safetySurf.c;
            
            % Коэффициенты квадратного уравнения At^2 + Bt + C = 0
            A = (tau(1)^2)/a^2 + (tau(2)^2)/b^2 + (tau(3)^2)/c^2;
            B = 2*(P(1)*tau(1)/a^2 + P(2)*tau(2)/b^2 + P(3)*tau(3)/c^2);
            C = (P(1)^2)/a^2 + (P(2)^2)/b^2 + (P(3)^2)/c^2 - 1;
            
            D = B^2 - 4*A*C;
            if D < 0
                t = [];
                return;
            end
            sqrtD = sqrt(D);
            t1 = (-B - sqrtD) / (2*A);
            t2 = (-B + sqrtD) / (2*A);
            % Выбираем наименьшее положительное
            t = [];
            if t1 > 1e-8
                t = t1;
            end
            if t2 > 1e-8 && (isempty(t) || t2 < t)
                t = t2;
            end
        end
    end
end
