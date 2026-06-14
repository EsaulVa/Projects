classdef TrajectorySynthesizer < handle
%TRAJECTORYSYNTHESIZER Синтез траектории раскладчика R(z) по линии укладки.
%
%   Режимы работы:
%     1. 'shadow' — трассировка луча от r(s) вдоль касательной до пересечения с E1
%     2. 'explicit_lambda' — явное задание длины свободного участка ?(s)
%
%   Пример:
%       synth = Geometry.TrajectorySynthesizer(E2, E1, 'shadow');
%       [R_traj, lambda_s] = synth.synthesize(lu_traj);

properties (Access = private)
    surf_mandrel    % поверхность оправки (E2 или E3)
    surf_safety     % поверхность безопасности (E1)
    mode            % 'shadow' или 'explicit_lambda'
    lambda_func     % функция ?(s) для режима 'explicit_lambda'
    t_min           % минимальная длина луча для трассировки
    t_max           % максимальная длина луча для трассировки
end

methods
    function obj = TrajectorySynthesizer(surf_mandrel, surf_safety, mode, varargin)
    % Конструктор
        obj.surf_mandrel = surf_mandrel;
        obj.surf_safety = surf_safety;
        obj.mode = mode;
        
        % Значения по умолчанию
        obj.t_min = 1.0;
        obj.t_max = 3000.0;
        obj.lambda_func = @(s) 1000;
        
        % Парсинг дополнительных параметров
        p = inputParser;
        addParameter(p, 'LambdaFunction', @(s) 1000, @isfunction_handle);
        addParameter(p, 'TMin', 1.0, @isnumeric);
        addParameter(p, 'TMax', 3000.0, @isnumeric);
        parse(p, varargin{:});
        
        obj.lambda_func = p.Results.LambdaFunction;
        obj.t_min = p.Results.TMin;
        obj.t_max = p.Results.TMax;
    end
    
    function [R_traj, lambda_s, s_vals] = synthesize(obj, lu_traj, num_points)
    % Синтез траектории R(z)
        if nargin < 3
            num_points = 2500;
        end
        
        L = lu_traj.totalLength();
        s_vals = linspace(0, L, num_points);
        R_points = zeros(3, num_points);
        lambda_s = zeros(num_points, 1);
        valid_mask = false(num_points, 1);
        
        fprintf('TrajectorySynthesizer: синтез %d точек в режиме ''%s''...\n', ...
            num_points, obj.mode);
        
        for i = 1:num_points
            s = s_vals(i);
            r_point = lu_traj.getPoint(s);
            r_point = r_point(:);
            tau_lu = lu_traj.getTangent(s);
            tau_lu = tau_lu(:);
            
            % Нормаль к оправке в точке r_point
            n = obj.get_normal_at(r_point);
            
            % Проекция касательной на касательную плоскость оправки
            tau_proj = obj.project_to_tangent(tau_lu, n);
            
            if strcmp(obj.mode, 'shadow')
                % Трассировка луча до поверхности безопасности
                [t, pt] = obj.trace_ray(r_point, tau_proj);
                if ~isnan(t)
                    R_points(:, i) = pt;
                    lambda_s(i) = t;
                    valid_mask(i) = true;
                end
            elseif strcmp(obj.mode, 'explicit_lambda')
                % Явное задание длины свободного участка
                lambda_val = obj.lambda_func(s);
                R_points(:, i) = r_point + lambda_val * tau_proj;
                lambda_s(i) = lambda_val;
                valid_mask(i) = true;
            end
            
            if mod(i, 500) == 0
                fprintf('  Обработано %d/%d точек\n', i, num_points);
            end
        end
        
        % Оставляем только успешные точки
        valid_idx = find(valid_mask);
        R_valid = R_points(:, valid_idx);
        s_valid = s_vals(valid_idx);
        
        fprintf('  Успешно синтезировано: %d из %d точек\n', length(valid_idx), num_points);
        
        % Перепараметризация по длине дуги z
        R_traj = obj.reparametrize_by_arc_length(R_valid, s_valid);
    end
end

methods (Access = private)
    function n = get_normal_at(obj, point)
    % Вычисление нормали к оправке в точке point
        % Для DiffusedRevolutionSurface
        if ismethod(obj.surf_mandrel, 's_from_z')
            z_coord = point(3);
            v_coord = atan2(point(2), point(1));
            s_merid = obj.surf_mandrel.s_from_z(z_coord);
            n = obj.surf_mandrel.normal(s_merid, v_coord);
        % Для DiscreteRevolutionSurface или RevolutionSurface
        elseif ismethod(obj.surf_mandrel, 'derivatives')
            z_coord = point(3);
            v_coord = atan2(point(2), point(1));
            d = obj.surf_mandrel.derivatives(z_coord, v_coord);
            n = d.normal;
        else
            error('TrajectorySynthesizer: неизвестный тип поверхности оправки');
        end
        n = n(:);
    end
    
    function tau_proj = project_to_tangent(obj, tau, n)
    % Проекция вектора tau на касательную плоскость (ортогональную n)
        tau_proj = tau - dot(tau, n) * n;
        if norm(tau_proj) < 1e-8
            % Вырожденный случай: tau коллинеарен нормали
            tau_proj = [1; 0; 0];
        else
            tau_proj = tau_proj / norm(tau_proj);
        end
    end
    
    function [t, pt] = trace_ray(obj, origin, direction)
    % Трассировка луча origin + t*direction до пересечения с поверхностью безопасности
        t = NaN;
        pt = NaN(3, 1);
        
        % Дискретизация луча
        t_vec = linspace(obj.t_min, obj.t_max, 2000);
        
        for j = 1:length(t_vec)-1
            t1 = t_vec(j);
            t2 = t_vec(j+1);
            P1 = origin + t1 * direction;
            P2 = origin + t2 * direction;
            
            % Проверка пересечения с поверхностью безопасности
            f1 = obj.safety_surface_residual(P1);
            f2 = obj.safety_surface_residual(P2);
            
            if ~isnan(f1) && ~isnan(f2) && f1 * f2 <= 0
                % Пересечение найдено, уточняем бисекцией
                t = obj.bisection(origin, direction, t1, t2, 20);
                pt = origin + t * direction;
                return;
            end
        end
    end
    
    function f = safety_surface_residual(obj, point)
% Остаток уравнения поверхности безопасности: rho - R(z)
    rho = sqrt(point(1)^2 + point(2)^2);
    z = point(3);
    
    % Определяем диапазон осевой координаты для поверхности безопасности
    % Разные классы используют разные имена свойств
    if isprop(obj.surf_safety, 'z_min') && isprop(obj.surf_safety, 'z_max')
        % DiffusedRevolutionSurface с preserve_z = true
        z_min = obj.surf_safety.z_min;
        z_max = obj.surf_safety.z_max;
    elseif isprop(obj.surf_safety, 'u_min') && isprop(obj.surf_safety, 'u_max')
        % DiscreteRevolutionSurface или RevolutionSurface (u = z)
        z_min = obj.surf_safety.u_min;
        z_max = obj.surf_safety.u_max;
    else
        error('TrajectorySynthesizer: поверхность безопасности не имеет свойств z_min/z_max или u_min/u_max');
    end
    
    % Проверяем, что точка в диапазоне
    if z < z_min || z > z_max
        f = NaN;
        return;
    end
    
    % Вычисляем радиус поверхности в точке z
    if ismethod(obj.surf_safety, 'radius')
        % DiffusedRevolutionSurface или DiscreteRevolutionSurface
        R_safety = obj.surf_safety.radius(z);
        f = rho - R_safety;
    elseif ismethod(obj.surf_safety, 'position')
        % RevolutionSurface
        pt = obj.surf_safety.position(z, 0);
        R_safety = sqrt(pt(1)^2 + pt(2)^2);
        f = rho - R_safety;
    else
        error('TrajectorySynthesizer: поверхность безопасности не имеет методов radius или position');
    end
end
    
    function t = bisection(obj, origin, direction, t1, t2, max_iter)
    % Метод бисекции для уточнения пересечения
        f1 = obj.safety_surface_residual(origin + t1 * direction);
        f2 = obj.safety_surface_residual(origin + t2 * direction);
        
        for iter = 1:max_iter
            t_mid = (t1 + t2) / 2;
            f_mid = obj.safety_surface_residual(origin + t_mid * direction);
            
            if isnan(f_mid)
                t = NaN;
                return;
            end
            
            if abs(f_mid) < 1e-6 || (t2 - t1) < 1e-6
                t = t_mid;
                return;
            end
            
            if f1 * f_mid <= 0
                t2 = t_mid;
                f2 = f_mid;
            else
                t1 = t_mid;
                f1 = f_mid;
            end
        end
        t = (t1 + t2) / 2;
    end
    
    function R_traj = reparametrize_by_arc_length(obj, R_valid, s_valid)
    % Перепараметризация траектории R по длине дуги z
        N = size(R_valid, 2);
        if N < 2
            R_traj = Geometry.Trajectory(R_valid);
            return;
        end
        
        % Вычисление длин дуги между точками
        ds = sqrt(sum(diff(R_valid, 1, 2).^2, 1));
        z_vals = [0, cumsum(ds)];
        
        % Интерполяция на равномерную сетку по z
        z_max = z_vals(end);
        z_uniform = linspace(0, z_max, N);
        
        R_uniform = zeros(3, N);
        for dim = 1:3
            R_uniform(dim, :) = interp1(z_vals, R_valid(dim, :), z_uniform, 'pchip');
        end
        
        R_traj = Geometry.Trajectory(R_uniform);
    end
end
end