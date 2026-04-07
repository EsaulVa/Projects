%% Вспомогательная функция для получения "сырых" первых производных
function [xp, yp] = getRawDerivatives(spline_obj, u_eval)
    % Обход нормализации в методе predict(der=1)
    % Вычисляем вручную на основе коэффициентов
    
    u_eval = u_eval(:);
    n = length(u_eval);
    xp = zeros(n, 1);
    yp = zeros(n, 1);
    
    % Структуры данных сплайна
    u_knots = spline_obj.u;
    
    for k = 1:n
        u_val = u_eval(k);
        
        % Поиск сегмента
        idx = find(u_knots <= u_val, 1, 'last');
        if isempty(idx), idx = 1; end
        if idx > length(spline_obj.m_x)-1, idx = length(spline_obj.m_x)-1; end
        
        t = u_val - u_knots(idx);
        
        % Коэффициенты X
        a3 = spline_obj.coeffs_x{idx}(1);
        a4 = spline_obj.coeffs_x{idx}(2);
        a5 = spline_obj.coeffs_x{idx}(3);
        
        % Коэффициенты Y
        b3 = spline_obj.coeffs_y{idx}(1);
        b4 = spline_obj.coeffs_y{idx}(2);
        b5 = spline_obj.coeffs_y{idx}(3);
        
        % Производные полинома 5-й степени:
        % S(t) = v + m*t + M/2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        % S'(t) = m + M*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
        
        xp(k) = spline_obj.m_x(idx) + spline_obj.M_x(idx)*t + ...
                3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4;
                
        yp(k) = spline_obj.m_y(idx) + spline_obj.M_y(idx)*t + ...
                3*b3*t^2 + 4*b4*t^3 + 5*b5*t^4;
    end
end

