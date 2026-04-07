function bc_polar = convert_cartesian_bc_to_polar(x, y, dy_dx, d2y_dx2)
    % Конвертация ГУ с учетом второй производной (кривизны).
    % Если d2y_dx2 задано (не NaN), вычисляем фиксированные r'' и phi''.
    
    r_val = sqrt(x^2 + y^2);
    bc_polar.r = struct('value', r_val);
    bc_polar.phi = struct('value', atan2(y, x));
    
    % --- Первые производные ---
    if isnan(dy_dx)
        bc_polar.r.deriv1 = NaN;
        bc_polar.phi.deriv1 = NaN;
        % Если нет первой производной, вторую считать нельзя
        bc_polar.r.deriv2 = NaN;
        bc_polar.phi.deriv2 = NaN;
        return;
    end
    
    % 1. Касательный вектор и скорость
    denom = sqrt(1 + dy_dx^2);
    tx = 1 / denom;
    ty = dy_dx / denom;
    
    x_dot = tx;
    y_dot = ty;
    
    dr_du = (x * x_dot + y * y_dot) / r_val;
    dphi_du = (x * y_dot - y * x_dot) / r_val^2;
    
    bc_polar.r.deriv1 = dr_du;
    bc_polar.phi.deriv1 = dphi_du;
    
    % --- Вторые производные ---
    if isnan(d2y_dx2)
        bc_polar.r.deriv2 = NaN;
        bc_polar.phi.deriv2 = NaN;
    else
        % 2. Кривизна
        kappa = d2y_dx2 / (denom^3);
        
        % 3. Вектор ускорения (a = kappa * n)
        % Нормаль n = (-ty, tx)
        x_ddot = -kappa * ty;
        y_ddot =  kappa * tx;
        
        % 4. Переход к полярным ускорениям
        % r'' = (x*x'' + y*y'' + x'^2 + y'^2 - r'^2) / r
        % x'^2 + y'^2 = 1 (при u=s)
        d2r_du2 = (x * x_ddot + y * y_ddot + 1 - dr_du^2) / r_val;
        
        % phi'' = (x*y'' - y*x'' - 2*r*r'*phi') / r^2
        d2phi_du2 = (x * y_ddot - y * x_ddot - 2 * r_val * dr_du * dphi_du) / r_val^2;
        
        bc_polar.r.deriv2 = d2r_du2;
        bc_polar.phi.deriv2 = d2phi_du2;
    end
end

