function [dy_dx, d2y_dx2] = get_cartesian_derivatives(r, phi, r1, phi1, r2, phi2)
    % 1. Вычисляем компоненты скорости (v_r, v_phi)
    vr = r1;
    vp = r .* phi1;
    
    % 2. Вычисляем компоненты ускорения (a_r, a_phi)
    % ar = r'' - r*(phi')^2
    ar = r2 - r .* (phi1.^2);
    % ap = r*phi'' + 2*r'*phi'
    ap = r .* phi2 + 2 .* r1 .* phi1;
    
    % 3. Декартовы производные по u (x', y', x'', y'')
    cos_phi = cos(phi);
    sin_phi = sin(phi);
    
    x1 = vr .* cos_phi - vp .* sin_phi;
    y1 = vr .* sin_phi + vp .* cos_phi;
    
    x2 = ar .* cos_phi - ap .* sin_phi;
    y2 = ar .* sin_phi + ap .* cos_phi;
    
    % 4. Производные по x
    % y'(x) = y' / x'
    % Защита от деления на 0 (вертикальная касательная)
    dy_dx = y1 ./ x1; 
    
    % y''(x) = (y''x' - y'x'') / (x')^3
    d2y_dx2 = (y2 .* x1 - y1 .* x2) ./ (x1.^3);
end

