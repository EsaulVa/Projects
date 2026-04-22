classdef QuinticSpline < handle
    %QUINTICSPLINE Базовый класс для квинтических сплайнов
    
    properties
        x           % Узлы сетки
        y           % Значения в узлах
        n           % Количество узлов
        bc_left     % Граничные условия слева
        bc_right    % Граничные условия справа
        m           % Первые производные в узлах
        M           % Вторые производные в узлах
        segments_coeffs % Коэффициенты сегментов
    end
    
    methods
        function obj = QuinticSpline(x, y, bc_left, bc_right)
            % Конструктор класса
            obj.x = x(:);
            obj.y = y(:);
            obj.n = length(x);
            obj.bc_left = bc_left;
            obj.bc_right = bc_right;
            obj.m = zeros(obj.n, 1);
            obj.M = zeros(obj.n, 1);
            obj.segments_coeffs = cell(obj.n-1, 1);
        end
        
        function coeffs = getSegmentCoeffs(obj, i, y_all, m_all, M_all)
            % Вычисляет коэффициенты a3, a4, a5 для отрезка i
            h = obj.x(i+1) - obj.x(i);
            yi = y_all(i);
            yip1 = y_all(i+1);
            mi = m_all(i);
            mip1 = m_all(i+1);
            Mi = M_all(i);
            Mip1 = M_all(i+1);
            
            B1 = yip1 - (yi + mi*h + 0.5*Mi*h^2);
            B2 = mip1 - (mi + Mi*h);
            B3 = Mip1 - Mi;
            
            A = [h^3, h^4, h^5;
                 3*h^2, 4*h^3, 5*h^4;
                 6*h, 12*h^2, 20*h^3];
            
            B = [B1; B2; B3];
            
            coeffs = A\B;
        end
    end
end

