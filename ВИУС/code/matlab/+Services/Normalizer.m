classdef Normalizer
    properties
        ScaleFactor = 1  % Значение по умолчанию
        Center = [0, 0]  % Значение по умолчанию
    end
    methods
        function obj = Normalizer(x, y)
            % NORMALIZER Конструктор.
            % Если x, y заданы - вычисляет центр и масштаб автоматически.
            % Если вызван без аргументов - создается объект с масштабом 1 и центром (0,0),
            % который можно задать вручную позже.
            
            if nargin == 2
                % Автоматический расчет (как было)
                obj.Center = [mean(x), mean(y)];
                r_vals = sqrt((x - obj.Center(1)).^2 + (y - obj.Center(2)).^2);
                obj.ScaleFactor = mean(r_vals);
                
                if obj.ScaleFactor < 1e-6
                    obj.ScaleFactor = 1;
                end
            end
            % Если nargin == 0, просто возвращаем объект с дефолтными значениями
        end
        
        function [xn, yn] = normalize(obj, x, y)
            xn = (x - obj.Center(1)) / obj.ScaleFactor;
            yn = (y - obj.Center(2)) / obj.ScaleFactor;
        end
        
        function [x, y] = denormalize(obj, xn, yn)
            x = xn * obj.ScaleFactor + obj.Center(1);
            y = yn * obj.ScaleFactor + obj.Center(2);
        end
        
        function r_phys = denormRadius(obj, r_norm)
            r_phys = r_norm * obj.ScaleFactor;
        end
        
        function dr_phys = denormDeriv2(obj, dr_norm)
            dr_phys = dr_norm / obj.ScaleFactor;
        end
    end
end

