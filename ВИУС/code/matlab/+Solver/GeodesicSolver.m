classdef GeodesicSolver < handle
    %GEODESICSOLVER Планировщик траектории (Решатель геодезических)
    %   Вычисляет путь на поверхности вращения от начальной точки до целевой,
    %   используя дифференциальные уравнения геодезических линий.
    
    properties (Access = private)
        mapper          % Объект SurfaceRevolutionMapper
        u_target        % Целевой параметр образующей
        v_target        % Целевой угловой параметр
        stopTolerance   % Допуск остановки (радиус "мишени")
    end
    
    methods
        function obj = GeodesicSolver(mapper)
            %GEODESICSOLVER Конструктор
            %   mapper: Объект класса Splines.SurfaceRevolutionMapper
            
            if nargin < 1
                error('Необходимо передать SurfaceMapper');
            end
            obj.mapper = mapper;
        end
        
        function pathStruct = solve(obj, u0, v0, alpha0, u_target, v_target, tolerance)
            %SOLVE Расчет траектории
            %   u0, v0: Начальные параметры
            %   alpha0: Начальный угол намотки (радианы)
            %   u_target, v_target: Целевые параметры
            %   tolerance: Радиус зоны остановки
            
            if nargin < 7
                tolerance = 1e-3; % Дефолтный допуск
            end
            
            % Сохраняем цель для функции событий
            obj.u_target = u_target;
            obj.v_target = v_target;
            obj.stopTolerance = tolerance;
            
            % 1. Вычисляем начальные скорости (u', v') из угла alpha0
            % Получаем метрику в начальной точке
            [E, ~, G] = obj.mapper.getMetricCoeffs(u0);
            
            % Нормировка скорости (ds - длина дуги, поэтому u'^2*E + v'^2*G = 1)
            % u' = cos(alpha) / sqrt(E)
            % v' = sin(alpha) / sqrt(G)
            
            % Защита от деления на ноль
            E = max(E, 1e-12);
            G = max(G, 1e-12);
            
            u_prime0 = cos(alpha0) / sqrt(E);
            v_prime0 = sin(alpha0) / sqrt(G);
            
            % Вектор начального состояния Y = [u; v; u'; v']
            Y0 = [u0; v0; u_prime0; v_prime0];
            
            % 2. Настройка решателя (включая функцию остановки)
            options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9, ...
                             'Events', @obj.eventsFunction);
            
            % 3. Интегрирование
            % Интегрируем до бесконечности (по факту до срабатывания Events)
            fprintf('Запуск расчета траектории...\n');
            [T, Y] = ode15s(@obj.systemEquations, [0 Inf], Y0, options);
            fprintf('Расчет завершен. Точек траектории: %d\n', length(T));
            
            % 4. Преобразование параметров в декартовы координаты
            u_sol = Y(:, 1);
            v_sol = Y(:, 2);
            
            [X, Y_cart, Z] = obj.mapper.getPosition(u_sol, v_sol);
            
            % 5. Формирование результата
            pathStruct.u = u_sol;
            pathStruct.v = v_sol;
            pathStruct.s = T;       % Длина дуги (параметр интегрирования)
            pathStruct.X = X;
            pathStruct.Y = Y_cart;  % Переименовываем, чтобы не конфликтовать с вектором состояния Y
            pathStruct.Z = Z;
        end
    end
    
    methods (Access = private)
%         %% Система дифференциальных уравнений
%         function dY = systemEquations(obj, ~, Y)
%     %SYSTEMEQUATIONS Уравнения геодезических с защитой от NaN
%     
%             % Распаковка
%             u = Y(1);
%             v = Y(2);
%             u_prime = Y(3);
%             v_prime = Y(4);
% 
%             % Получаем ссылку на сплайн
% %             gen = obj.mapper.generatrix;
%            gen = obj.mapper.getSpline();
% 
%             % --- ЗАЩИТА 1: Клиппинг параметра u ---
%             % Не даем уйти за границы определения сплайна (даже при пробных шагах решателя)
%             u_min = gen.u(1);
%             u_max = gen.u(end);
%             u = max(min(u, u_max), u_min); 
% 
%             % Получаем данные из сплайнов
%             % Первые производные
%             der1 = gen.predict(u, 1); 
%             Z1 = der1(1);
%             R1 = der1(2);
% 
%             % Вторые производные
%             der2 = gen.predict(u, 2);
%             Z2 = der2(1);
%             R2 = der2(2);
% 
%             % Радиус
%             data = gen.predict(u, 0);
%             R0 = data(2);
% 
%             % --- ЗАЩИТА 2: Защита радиуса ---
%             % Если сплайн "уходит" в отрицательный радиус из-за осцилляций, 
% %             // считаем, что радиус минимально положительный (например, 1 мм)
%             R0 = max(R0, 1e-3); 
%             R1 = max(R1, 1e-3); % Защита производной тоже не помешает
% 
%             % Вычисляем метрику
%             E = Z1^2 + R1^2;
%             G = R0^2;
% 
%             % Производные метрики
%             E_prime = 2 * (Z1 * Z2 + R1 * R2);
%             G_prime = 2 * R0 * R1;
% 
%             % --- ЗАЩИТА 3: Избегаем деления на ноль ---
%             % Добавляем машинный эпсилон к знаменателям
%             eps_val = 1e-12;
%             E_safe = max(E, eps_val);
%             G_safe = max(G, eps_val);
% 
%             % Ускорения
%             du_prime = - (E_prime / (2 * E_safe)) * u_prime^2 + (G_prime / (2 * E_safe)) * v_prime^2;
%             dv_prime = - (G_prime / G_safe) * u_prime * v_prime;
% 
%             % Возвращаем производные
%             dY = [u_prime; v_prime; du_prime; dv_prime];
%        end
        
        function dY = systemEquations(obj, ~, Y)
            %SYSTEMEQUATIONS Уравнения геодезических на поверхности вращения
            
            % Распаковка вектора состояния
            u = Y(1);
            v = Y(2); % хотя v не используется явно в ускорениях, он есть в состоянии
            u_prime = Y(3);
            v_prime = Y(4);
            
            % Получаем данные из сплайнов образующей
            % Обращаемся к generatrix (ParametricQuinticSpline) внутри маппера
%             gen = obj.mapper.generatrix;
            gen = obj.mapper.getSpline();
            
            % Первые производные (d/du)
            der1 = gen.predict(u, 1); 
            Z1 = der1(1);
            R1 = der1(2);
            
            % Вторые производные (d^2/du^2)
            der2 = gen.predict(u, 2);
            Z2 = der2(1);
            R2 = der2(2);
            
            % Значение радиуса
            data = gen.predict(u, 0);
            R0 = data(2);
            
            % Вычисляем коэффициенты метрики и их производные
            % E = (Z')^2 + (R')^2
            E = Z1^2 + R1^2;
            
            % G = R^2
            G = R0^2;
            
            % Производные метрики
            % E' = 2 * (Z'*Z'' + R'*R'')
            E_prime = 2 * (Z1 * Z2 + R1 * R2);
            
            % G' = 2 * R * R'
            G_prime = 2 * R0 * R1;
            
            % Ускорения (вторые производные по s)
            % u'' = - (E' / 2E) * (u')^2 + (G' / 2E) * (v')^2
            du_prime = - (E_prime / (2 * E)) * u_prime^2 + (G_prime / (2 * E)) * v_prime^2;
            
            % v'' = - (G' / G) * u' * v'
            dv_prime = - (G_prime / G) * u_prime * v_prime;
            
            % Возвращаем производные вектора состояния [u'; v'; u''; v'']
            dY = [u_prime; v_prime; du_prime; dv_prime];
        end
        
        %% Функция событий (Остановка у цели)
        
%         function [value, isterminal, direction] = eventsFunction(obj, ~, Y)
%             %EVENTSFUNCTION Определяет, когда остановить интеграцию
%             
%             u_curr = Y(1);
%             v_curr = Y(2);
%             
%             % Расстояние в квадрате до целевой точки в пространстве параметров (u, v)
%             dist_sq = (u_curr - obj.u_target)^2 + (v_curr - obj.v_target)^2;
%             
%             % Событие: когда расстояние становится меньше допуска (то есть dist^2 - tol^2 < 0)
%             value = dist_sq - (obj.stopTolerance)^2;
%             
%             isterminal = 1; % Остановить интеграцию при пересечении нуля
%             direction = -1; % Остановить только при движении к уменьшению расстояния
%         end
        function [value, isterminal, direction] = eventsFunction(obj, ~, Y)
            u_curr = Y(1);
            v_curr = Y(2);

            % 1. Проверка выхода за пределы оправки (Safety stop)
            if u_curr >= obj.u_target || u_curr <= obj.u_target 
               % (для обратного хода, если нужно)
               value = 1; 
               isterminal = 1;
               direction = -1;
               return;
            end

            % 2. Проверка достижения целевого угла (Your actual goal)
            % Останавливаемся, когда v_curr превысил v_target
            value = v_curr - obj.v_target; 

            isterminal = 1; % Остановить интеграцию
            direction = 1;   % При пересечении снизу вверх
        end
    end
end
