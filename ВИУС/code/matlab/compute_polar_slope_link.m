function bc_link = compute_polar_slope_link(x, y, dy_dx)
    % COMPUTE_POLAR_SLOPE_LINK Вычисляет коэффициент связи для ГУ r' = C * phi'
    %
    % Вход:
    %   x, y    - координаты точки (локальные, относительно центра)
    %   dy_dx   - декартова производная y'(x)
    %
    % Выход:
    %   bc_link - структура с полями:
    %       .type     - 'coupled' или 'radial'
    %       .C        - коэффициент связи (r' = C * phi')
    %       .norm_r   - оценка r'(u) при условии |v|=1 (для справки)
    %       .norm_phi - оценка phi'(u) при условии |v|=1 (для справки)

    r = sqrt(x^2 + y^2);
    
    if isnan(dy_dx)
        bc_link.type = 'free';
        bc_link.C = NaN;
        return;
    end
    
    % 1. Формируем единичный касательный вектор T из y'(x)
    denom = sqrt(1 + dy_dx^2);
    tx = 1 / denom;
    ty = dy_dx / denom;
    
    % 2. Вычисляем производные по u (в предположении u=s, просто для получения пропорции)
    % r'(u) = (x*tx + y*ty)/r
    % phi'(u) = (x*ty - y*tx)/r^2
    
    % Защита от r=0
    if r < 1e-9
        bc_link.type = 'singular_center';
        bc_link.C = NaN;
        return;
    end
    
    phi_prime_u = (x*ty - y*tx) / r^2;
    
    % 3. Проверка на радиальный случай (касательная проходит через центр)
    % В этом случае phi'(u) = 0, значит знаменатель обнуляется.
    if abs(phi_prime_u) < 1e-9
        bc_link.type = 'radial_tangent';
        % В этом случае условие связи r' = C*phi' невозможно (C = inf).
        % Вместо этого мы должны задать жесткое условие: phi'(u) = 0.
        bc_link.C = Inf; 
        bc_link.fixed_phi_prime = 0;
        
        % r'(u) в этом случае просто проекция T на радиус
        bc_link.norm_r = tx*(x/r) + ty*(y/r); % = dot(T, e_r)
    else
        bc_link.type = 'coupled';
        % Вычисляем r'(u) для числителя
        r_prime_u = (x*tx + y*ty) / r;
        
        % Вычисляем коэффициент C = r'(phi)
        bc_link.C = r_prime_u / phi_prime_u;
        
        % Сохраняем нормированные значения на всякий случай
        bc_link.norm_r = r_prime_u;
        bc_link.norm_phi = phi_prime_u;
    end
end