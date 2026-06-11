function [u_hist, v_hist, s_hist] = geodesicOnEllipsoid(surf, u0, v0, azimuth0, s_end, method, steps)
% Геодезическая на поверхности (обобщённая версия)
% Вход:
%   surf      - объект класса Surface (методы getFirstFundamental, getFirstDerivatives)
%   u0, v0    - начальные криволинейные координаты
%   azimuth0  - начальный азимут (угол между касательным вектором и ru) в радианах
%   s_end     - полная длина геодезической
%   method    - 'ode45' (по умолчанию) или 'rk4'
%   steps     - количество шагов для RK4 (по умолчанию 1000)
% Выход:
%   u_hist, v_hist, s_hist - массивы значений вдоль длины s

    if nargin < 6
        method = 'ode45';
    end
    if nargin < 7
        steps = 1000;
    end

    % 1. Вычисляем начальные p0, q0 из азимута
    [ru0, rv0] = surf.getFirstDerivatives(u0, v0);
    [E0, F0, G0] = surf.getFirstFundamental(u0, v0);
    
    % Ортогонализация базиса для задания направления
    e1 = ru0 / sqrt(E0);
    e2 = (rv0 - (F0/sqrt(E0)) * e1) / sqrt(G0 - F0^2/E0);
    t = cos(azimuth0) * e1 + sin(azimuth0) * e2;   % единичный касательный вектор в 3D
    
    % Разложение t по ru, rv: t = p0 * ru0 + q0 * rv0
    M = [E0, F0; F0, G0];
    rhs = [dot(t, ru0); dot(t, rv0)];
    pq0 = M \ rhs;
    p0 = pq0(1);
    q0 = pq0(2);
    
    % 2. Интегрирование системы
    % dy/ds = f(s, y), где y = [u; v; p; q]
    function dyds = geodesic_rhs(s, y)
        u = y(1); v = y(2); p = y(3); q = y(4);
        
        % Получаем метрику и её производные (численно)
        [E, F, G] = surf.getFirstFundamental(u, v);
        
        % Частные производные метрики по u и v (центральные разности)
        delta = 1e-6;
        [E_u, F_u, G_u] = surf.getFirstFundamental(u+delta, v);
        [E_v, F_v, G_v] = surf.getFirstFundamental(u, v+delta);
        E_u = (E_u - E) / delta;
        F_u = (F_u - F) / delta;
        G_u = (G_u - G) / delta;
        E_v = (E_v - E) / delta;
        F_v = (F_v - F) / delta;
        G_v = (G_v - G) / delta;
        
        % Символы Кристоффеля второго рода
        det = E*G - F^2;
        Gammau_uu = ( G*E_u - 2*F*F_u + F*E_v ) / (2*det);
        Gammau_uv = ( G*E_v - F*G_u ) / (2*det);
        Gammau_vv = ( 2*G*F_v - G*G_u - F*G_v ) / (2*det);
        Gammav_uu = ( 2*E*F_u - E*E_v - F*E_u ) / (2*det);
        Gammav_uv = ( E*G_u - F*E_v ) / (2*det);
        Gammav_vv = ( E*G_v - 2*F*F_v + F*G_u ) / (2*det);
        
        dudt = p;
        dvdt = q;
        dpdt = -Gammau_uu * p^2 - 2*Gammau_uv * p*q - Gammau_vv * q^2;
        dqdt = -Gammav_uu * p^2 - 2*Gammav_uv * p*q - Gammav_vv * q^2;
        
        dyds = [dudt; dvdt; dpdt; dqdt];
    end

    % Начальные условия
    y0 = [u0; v0; p0; q0];
    
    switch lower(method)
        case 'ode45'
            opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
            [s_hist, y_hist] = ode45(@geodesic_rhs, [0, s_end], y0, opts);
            u_hist = y_hist(:,1)';
            v_hist = y_hist(:,2)';
            s_hist = s_hist';
            
        case 'rk4'
            s_hist = linspace(0, s_end, steps+1);
            h = s_hist(2) - s_hist(1);
            y = y0;
            u_hist = zeros(1, steps+1);
            v_hist = zeros(1, steps+1);
            u_hist(1) = y(1);
            v_hist(1) = y(2);
            
            for i = 1:steps
                s = s_hist(i);
                % k1
                k1 = geodesic_rhs(s, y);
                % k2
                k2 = geodesic_rhs(s + h/2, y + h/2*k1);
                % k3
                k3 = geodesic_rhs(s + h/2, y + h/2*k2);
                % k4
                k4 = geodesic_rhs(s + h, y + h*k3);
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4);
                u_hist(i+1) = y(1);
                v_hist(i+1) = y(2);
            end
            s_hist = s_hist;
            
        otherwise
            error('Метод должен быть ''ode45'' или ''rk4''');
    end
end