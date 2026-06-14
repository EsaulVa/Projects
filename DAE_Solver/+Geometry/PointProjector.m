classdef PointProjector < handle
    %POINTPROJECTOR Ортогональное проектирование точки на поверхность вращения.
    %   Для поверхности вида r(z, v) = (r(z)*cos(v), r(z)*sin(v), z).
    %   Используется метод Ньютона для уравнения (rho - r(z)) * r'(z) + (z_p - z) = 0.
    %
    %   Пример:
    %       proj = Filter.PointProjector(E3);
    %       q = proj.project([x, y, z]);
    %       [q, z_proj, success] = proj.project_with_info([x, y, z]);

    properties (Access = private)
        surface  % объект, имеющий методы radius(z), radius_deriv(z), radius_deriv2(z), z_min, z_max
    end

    methods
        function obj = PointProjector(surf)
            % Конструктор
            % surf - объект поверхности вращения с поддержкой:
            %        radius(z), radius_deriv(z), radius_deriv2(z), z_min, z_max
            if ~(isa(surf, 'Filter.DiffusedRevolutionSurface') || ...
                 (isobject(surf) && ismethod(surf, 'radius') && ismethod(surf, 'radius_deriv')))
                error('PointProjector: поверхность должна реализовывать методы radius(z), radius_deriv(z), radius_deriv2(z)');
            end
            obj.surface = surf;
        end

        function q = project(obj, p)
            % Проекция точки p (вектор 1x3 или 3x1) на поверхность.
            % Возвращает точку q (1x3) на поверхности.
            [q, ~, ~] = obj.project_with_info(p);
        end

        function [q, z_proj, success] = project_with_info(obj, p)
            % Полная версия: возвращает точку q, найденное z и флаг успеха.
            p = p(:)'; % привести к 1x3
            rho = sqrt(p(1)^2 + p(2)^2);
            phi = atan2(p(2), p(1));
            zp = p(3);
            
            % Начальное приближение: z = zp, но обрезаем по границам
            z = max(obj.surface.z_min, min(obj.surface.z_max, zp));
            
            % Метод Ньютона
            max_iter = 20;
            tol = 1e-10;
            success = false;
            
            for iter = 1:max_iter
                r = obj.surface.radius(z);
                dr = obj.surface.radius_deriv(z);
                d2r = obj.surface.radius_deriv2(z);
                
                f = (rho - r) * dr + (zp - z);
                % Защита от нулевой производной
                df = -dr^2 + (rho - r) * d2r - 1;
                if abs(df) < 1e-12
                    df = sign(df) * 1e-12;
                end
                
                dz = -f / df;
                z_new = z + dz;
                
                % Ограничение по границам
                z_new = max(obj.surface.z_min, min(obj.surface.z_max, z_new));
                
                if abs(z_new - z) < tol
                    z = z_new;
                    success = true;
                    break;
                end
                z = z_new;
            end
            
            if ~success
                % Если не сошёлся, пробуем метод деления пополам на отрезке
                warning('PointProjector: Ньютон не сошёлся, используем бисекцию');
                z = obj.bisection(rho, zp);
            end
            
            z_proj = z;
            r_proj = obj.surface.radius(z);
            q = [r_proj * cos(phi), r_proj * sin(phi), z];
        end
    end

    methods (Access = private)
        function z = bisection(obj, rho, zp)
            % Метод деления пополам для надёжности
            a = obj.surface.z_min;
            b = obj.surface.z_max;
            fa = obj.f_bisection(a, rho, zp);
            fb = obj.f_bisection(b, rho, zp);
            
            if fa * fb > 0
                % Нет смены знака – берём ближайший конец
                if abs(fa) < abs(fb)
                    z = a;
                else
                    z = b;
                end
                return;
            end
            
            for iter = 1:50
                c = (a + b) / 2;
                fc = obj.f_bisection(c, rho, zp);
                if abs(fc) < 1e-12
                    z = c;
                    return;
                end
                if fa * fc < 0
                    b = c;
                    fb = fc;
                else
                    a = c;
                    fa = fc;
                end
            end
            z = (a + b) / 2;
        end
        
        function val = f_bisection(obj, z, rho, zp)
            r = obj.surface.radius(z);
            dr = obj.surface.radius_deriv(z);
            val = (rho - r) * dr + (zp - z);
        end
    end
end