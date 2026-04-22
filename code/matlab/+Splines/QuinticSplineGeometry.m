classdef QuinticSplineGeometry < handle
    %QUINTICSPLINEGEOMETRY Математический движок для квинтических сплайнов.
    % Не хранит состояние. Предназначен для вычисления коэффициентов и значений.
    
    methods
        function coeffs = getSegmentCoeffs(obj, yi, mi, Mi, yip1, mip1, Mip1, h)
            % Вычисление коэффициентов a3, a4, a5 для сегмента
            B = [yip1 - (yi + mi*h + 0.5*Mi*h^2);
                 mip1 - (mi + Mi*h);
                 Mip1 - Mi];
                 
            A = [h^3, h^4, h^5;
                 3*h^2, 4*h^3, 5*h^4;
                 6*h, 12*h^2, 20*h^3];
            
            coeffs = A \ B;
        end
        
        function val = evalValue(obj, t, yi, mi, Mi, coeffs)
            val = yi + mi*t + 0.5*Mi*t.^2 + coeffs(1)*t.^3 + coeffs(2)*t.^4 + coeffs(3)*t.^5;
        end
        
        function val = evalDeriv1(obj, t, mi, Mi, coeffs)
            val = mi + Mi*t + 3*coeffs(1)*t.^2 + 4*coeffs(2)*t.^3 + 5*coeffs(3)*t.^4;
        end
        
        function val = evalDeriv2(obj, t, Mi, coeffs)
            val = Mi + 6*coeffs(1)*t + 12*coeffs(2)*t.^2 + 20*coeffs(3)*t.^3;
        end
    end
end

