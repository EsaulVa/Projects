classdef RevolutionSurface < handle
    % Поверхность вращения: r(u,v) = [R(u)*cos(v), R(u)*sin(v), u]
    % Задаётся кусочно: днище (полиномы phi(u) и R(phi)), цилиндр, зеркальное днище.
    
    properties (SetAccess = private)
        phi_coeffs      % коэффициенты phi(u) (степень 4)
        R_coeffs        % коэффициенты R(phi) (степень 5)
        segment_bounds  % [a, b, c, d] – границы участков
        cylinder_radius
        u_min, u_max
        v_min, v_max
    end
    
    properties (Access = private)
        phi_prime_coeffs
        phi_dprime_coeffs
        R_prime_coeffs
        R_dprime_coeffs
    end
    
    methods        
        function obj = RevolutionSurface(phi_coeffs, R_coeffs, segment_bounds, cylinder_radius)
            obj.phi_coeffs = phi_coeffs(:)';
            obj.R_coeffs = R_coeffs(:)';
            obj.segment_bounds = segment_bounds;
            obj.cylinder_radius = cylinder_radius;
            obj.u_min = segment_bounds(1);
            obj.u_max = segment_bounds(4);
            obj.v_min = 0;
            obj.v_max = 2*pi;
            
            obj.phi_prime_coeffs = polyder(obj.phi_coeffs);
            obj.phi_dprime_coeffs = polyder(obj.phi_prime_coeffs);
            obj.R_prime_coeffs = polyder(obj.R_coeffs);
            obj.R_dprime_coeffs = polyder(obj.R_prime_coeffs);
        end
        
        function [r, ru, rv] = derivatives(obj, u, v)
            % Возвращает точку и первые производные
            [R, Rp, ~] = obj.radius_and_derivs(u);
            cosv = cos(v); sinv = sin(v);
            r = [R * cosv; R * sinv; u];
            ru = [Rp * cosv; Rp * sinv; 1];
            rv = [-R * sinv; R * cosv; 0];
        end
        
        function n = normal(obj, u, v)
            % Внешняя единичная нормаль (согласована с первой и второй формами)
            [R, Rp, ~] = obj.radius_and_derivs(u);
            denom = sqrt(1 + Rp^2);
            cosv = cos(v); sinv = sin(v);
            n = [cosv / denom; sinv / denom; -Rp / denom];
            % Проверка для цилиндрической части (Rp=0): n = [cosv; sinv; 0]
        end
        
        function [E, F, G] = first_fundamental_form(obj, u, v)
            [R, Rp, ~] = obj.radius_and_derivs(u);
            E = 1 + Rp^2;
            F = 0;
            G = R^2;
        end
        
        function [L, M, N] = second_fundamental_form(obj, u, v)
            [R, Rp, Rpp] = obj.radius_and_derivs(u);
            denom = sqrt(1 + Rp^2);
            L = Rpp / denom;   % согласовано с внешней нормалью
            M = 0;
            N = -R / denom;
        end
        
        function r = position(obj, u, v)
            [R, ~, ~] = obj.radius_and_derivs(u);
            r = [R * cos(v); R * sin(v); u];
        end
        
        function [u, v] = uv_from_point(obj, point)
            % Преобразование точки на поверхности в параметры
            x = point(1); y = point(2); z = point(3);
            v = atan2(y, x);
            u = z;
            % Коррекция u, если выходит за границы (не должно быть)
            u = max(obj.u_min, min(obj.u_max, u));
        end
         function R = radius(obj, u)
        [R, ~, ~] = obj.radius_and_derivs(u);
         end
    end
       
    
    methods (Access = private)
        function [R, Rp, Rpp] = radius_and_derivs(obj, u)
            % Вычисляет радиус и его производные по u
            seg = obj.get_segment(u);
            if seg == 1
                % Нижнее днище
                phi = polyval(obj.phi_coeffs, u);
                phi_p = polyval(obj.phi_prime_coeffs, u);
                phi_pp = polyval(obj.phi_dprime_coeffs, u);
                Rphi = polyval(obj.R_coeffs, phi);
                Rp_phi = polyval(obj.R_prime_coeffs, phi);
                Rpp_phi = polyval(obj.R_dprime_coeffs, phi);
                sin_phi = sin(phi);
                cos_phi = cos(phi);
                R = Rphi * sin_phi;
                temp = Rp_phi * sin_phi + Rphi * cos_phi;
                Rp = phi_p * temp;
                temp2 = Rpp_phi * sin_phi + 2*Rp_phi*cos_phi - Rphi*sin_phi;
                Rpp = phi_pp * temp + phi_p^2 * temp2;
            elseif seg == 2
                % Цилиндр
                R = obj.cylinder_radius;
                Rp = 0;
                Rpp = 0;
            else % seg == 3
                % Верхнее днище (зеркальное отражение)
                u_sym = obj.u_min + obj.u_max - u;
                [R_sym, Rp_sym, Rpp_sym] = obj.radius_and_derivs(u_sym);
                R = R_sym;
                Rp = -Rp_sym;      % r'(u) = -r'_sym(u_sym)
                Rpp = Rpp_sym;     % r''(u) = r''_sym(u_sym)
            end
        end
        
        function seg = get_segment(obj, u)
            % Определяет сегмент по u
            a = obj.segment_bounds(1);
            b = obj.segment_bounds(2);
            c = obj.segment_bounds(3);
            d = obj.segment_bounds(4);
            if u < a || u > d
                error('u вне диапазона [%f, %f]', a, d);
            end
            if u <= b
                seg = 1;
            elseif u < c
                seg = 2;
            else
                seg = 3;
            end
        end
    end
end