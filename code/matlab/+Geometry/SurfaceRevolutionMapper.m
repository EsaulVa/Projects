classdef SurfaceRevolutionMapper < handle
    %SURFACEREVOLUTIONMAPPER Модуль конвертации параметрической образующей в поверхность вращения
    %   Работает с объектом ParametricQuinticSpline, который задает профиль в плоскости ZR.
    %   Предполагается, что x-координата кривой -> Z (ось изделия), y-координата -> R (радиус).
    
    properties (Access = private)
        generatrix % Объект класса ParametricQuinticSpline (образующая в плоскости)
    end
    
    methods
        function obj = SurfaceRevolutionMapper(generatrix)
            %SURFACEREVOLUTIONMAPPER Конструктор
            %   generatrix: Объект ParametricQuinticSpline, описывающий профиль (Z vs R)
            
            if nargin < 1
                error('Необходимо передать объект ParametricQuinticSpline.');
            end
            
            obj.generatrix = generatrix;
        end
        
        %% --- Основная геометрия ---
        
        function [X, Y, Z] = getPosition(obj, u, v)
            %GETPOSITION Вычисляет декартовы координаты точки на поверхности

            % Сохраняем исходный размер сетки (например, 60x50)
            originalSize = size(u);

            % Получаем данные из сплайна (predict вернет вектор-столбец)
            curveData = obj.generatrix.predict(u, 0);

            % Извлекаем координаты (пока это векторы)
            Z_vec = curveData(:, 1);
            R_vec = curveData(:, 2);

            % !!! Ключевой момент: преобразуем векторы обратно в матрицы сетки !!!
            Z_vals = reshape(Z_vec, originalSize);
            R_vals = reshape(R_vec, originalSize);

            % Вычисляем X и Y (теперь это матричные операции, которые работают корректно)
            X = R_vals .* cos(v);
            Y = R_vals .* sin(v);

            % Z уже матрица
            Z = Z_vals;
        end
        function splineObj = getSpline(obj)
            %GETSPLINE Возвращает объект параметрического сплайна образующей
            splineObj = obj.generatrix;
        end
        function [ru, rv] = getFirstDerivatives(obj, u, v)
            %GETFIRSTDERIVATIVES Вычисляет касательные вектора r_u и r_v
            %   Возвращает векторы размера 3xN (где N - число точек)
            
            % Первые производные кривой (dZ/du, dR/du)
            curveDer1 = obj.generatrix.predict(u, 1);
            Z1 = curveDer1(:, 1);
            R1 = curveDer1(:, 2);
            
            % Значения радиуса (нужны для r_v)
            curveData = obj.generatrix.predict(u, 0);
            R0 = curveData(:, 2);
            
            cos_v = cos(v);
            sin_v = sin(v);
            
            % Вектор касательный к образующей (r_u)
            % r_u = [ R' * cos(v); R' * sin(v); Z' ]
            ru = [R1 .* cos_v; R1 .* sin_v; Z1];
            
            % Вектор касательный к параллели (r_v)
            % r_v = [ -R * sin(v); R * cos(v); 0 ]
            rv = [-R0 .* sin_v; R0 .* cos_v; zeros(size(u))];
        end
        
        function [E, F, G] = getMetricCoeffs(obj, u)
            %GETMETRICCOEFFS Вычисляет коэффициенты первой квадратичной формы
            
            % Получаем 1-е производные
            curveDer1 = obj.generatrix.predict(u, 1);
            Z1 = curveDer1(:, 1);
            R1 = curveDer1(:, 2);
            
            % Получаем значения радиуса
            curveData = obj.generatrix.predict(u, 0);
            R0 = curveData(:, 2);
            
            % E = (Z')^2 + (R')^2
            E = Z1.^2 + R1.^2;
            
            % G = R^2
            G = R0.^2;
            
            % F = 0
            F = zeros(size(u));
        end
        
        %% --- Данные для уравнений геодезических ---
        
        function Gamma = getChristoffelSymbols(obj, u)
            %GETCHRISTOFFELSYMBOLS Вычисляет символы Кристоффеля 2-го рода
            %   Использует упрощенные формулы для поверхности вращения.
            
            % Получаем 1-е и 2-е производные
            curveDer1 = obj.generatrix.predict(u, 1);
            Z1 = curveDer1(:, 1);
            R1 = curveDer1(:, 2);
            
            curveDer2 = obj.generatrix.predict(u, 2);
            Z2 = curveDer2(:, 1);
            R2 = curveDer2(:, 2);
            
            % Получаем радиус
            curveData = obj.generatrix.predict(u, 0);
            R0 = curveData(:, 2);
            
            % Коэффициенты метрики
            E = Z1.^2 + R1.^2;
            G = R0.^2;
            
            % Производные коэффициентов метрики по u
            Et = 2 .* (Z1 .* Z2 + R1 .* R2);
            Gt = 2 .* R0 .* R1;
            
            % Вычисляем символы
            eps_val = 1e-12; 
            E_safe = max(E, eps_val);
            G_safe = max(G, eps_val);
            
            Gamma = struct();
            
            Gamma.uuu = Et ./ (2 .* E_safe); % Гамма^1_11 (u corresponds to t)
            Gamma.uvv = -Gt ./ (2 .* E_safe); % Гамма^1_22
            
            Gamma.vuv = Gt ./ (2 .* G_safe);  % Гамма^2_12 (v corresponds to angle)
            Gamma.vvu = Gt ./ (2 .* G_safe);  % Гамма^2_21
        end
    end
end
