function [t, point] = trace_ray(surf, origin, direction, t_min, t_max)
% Численное пересечение луча с поверхностью вращения.
% Луч: P(t) = origin + t * direction, t >= t_min.
% Уравнение: sqrt(x(t)^2+y(t)^2) - R_surf(z(t)) = 0.
    ro = origin(:)';
    rd = direction(:)';
    
    function f = objective(t)
        pt = ro + t * rd;
        x = pt(1); y = pt(2); z = pt(3);
        if z < surf.u_min || z > surf.u_max
            f = 1e9;
            return;
        end
        R_surf = surf.radius(z);   % нужно добавить метод radius (см. ниже)
        if isnan(R_surf)
            f = 1e9;
            return;
        end
        R_ray = sqrt(x^2 + y^2);
        f = R_ray - R_surf;
    end

    % Добавим метод radius в RevolutionSurface (просто публичный)
    % Но для вызова внутри trace_ray сделаем анонимную функцию, используя уже имеющийся метод.
    % Однако у нас нет публичного radius. Добавим его в класс:
    
    % Поиск интервала смены знака
    N = 200;
    dt = (t_max - t_min) / N;
    t_prev = t_min;
    f_prev = objective(t_prev);
    for i = 1:N
        t_curr = t_min + i * dt;
        f_curr = objective(t_curr);
        if abs(f_curr) < 1e-10
            t = t_curr; point = ro + t*rd; return;
        end
        if f_prev * f_curr < 0
            try
                t = fzero(@objective, [t_prev, t_curr]);
                point = ro + t*rd;
                return;
            catch
                % метод Брента не сошёлся, продолжаем
            end
        end
        t_prev = t_curr;
        f_prev = f_curr;
    end
    t = NaN; point = NaN(1,3);
end