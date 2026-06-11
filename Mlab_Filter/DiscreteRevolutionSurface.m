classdef DiscreteRevolutionSurface < handle
    %DISCRETEREVOLUTIONSURFACE Поверхность вращения из таблицы (z, r).
    %   Параметр u = z (осевая координата), v = угол поворота.
    %   Использует кубический сплайн с аналитическими производными.
    %
    %   Пример:
    %      z = linspace(0, 800, 1000)';
    %      r = 250 + 50*sin(z/100);
    %      S = DiscreteRevolutionSurface(z, r);
    %      pt = S.position(400.0, 0.0);
    %      [E,F,G] = S.first_fundamental_form(400.0, 0.0);
    %      [L,M,N] = S.second_fundamental_form(400.0, 0.0);

    properties (Access = private)
        pp_r      % piecewise polynomial: r(z)
        pp_dr     % r'(z)
        pp_d2r    % r''(z)
    end

    properties (SetAccess = immutable)
        z_nodes   % исходная сетка z
        r_nodes   % исходная сетка r
        u_min     % min(z)
        u_max     % max(z)
    end

    methods
        function obj = DiscreteRevolutionSurface(z_array, r_array, bc_type)
            % Конструктор. Принимает векторы z и r.
            % bc_type — опционально: 'natural' (по умолчанию) или 'not-a-knot'.

            if nargin < 3
                bc_type = 'natural';
            end

            z = double(z_array(:));
            r = double(r_array(:));

            if length(z) ~= length(r)
                error('z_array и r_array должны иметь одинаковую длину');
            end
            if length(z) < 2
                error('Минимум 2 точки');
            end

            % Удаление дублей по z (если есть)
            [z_unique, idx] = unique(z, 'stable');
            if length(z_unique) < length(z)
                z = z_unique;
                r = r(idx);
            end

            % Проверка монотонности
            if any(diff(z) <= 0)
                error('z_array должен быть строго монотонно возрастающим');
            end

            obj.z_nodes = z;
            obj.r_nodes = r;
            obj.u_min = z(1);
            obj.u_max = z(end);

            % Построение сплайна r(z)
            if strcmpi(bc_type, 'clamped') && exist('csape', 'file')
                % csape требует Curve Fitting Toolbox
                obj.pp_r = csape(z, r, 'complete'); % или 'not-a-knot'
            else
                obj.pp_r = spline(z, r);
            end

            % Производные: аналитические (fnder) или численные
            obj.pp_dr = obj.build_derivative_spline(z, r, 1);
            obj.pp_d2r = obj.build_derivative_spline(z, r, 2);
        end

        function [r_val, dr_val, d2r_val] = eval_at(obj, u)
            % Возвращает r(u), r'(u), r''(u) для u = z.
            u = double(u);
            r_val = ppval(obj.pp_r, u);
            if nargout > 1
                dr_val = ppval(obj.pp_dr, u);
            end
            if nargout > 2
                d2r_val = ppval(obj.pp_d2r, u);
            end
        end

        function pt = position(obj, u, v)
            % Радиус-вектор точки (u=z, v=angle).
            [r, ~, ~] = obj.eval_at(u);
            pt = [r * cos(v), r * sin(v), u];
        end

        function d = derivatives(obj, u, v)
            % Базисы ru, rv и нормаль.
            [r, dr, ~] = obj.eval_at(u);
            ru = [dr * cos(v), dr * sin(v), 1.0];
            rv = [-r * sin(v), r * cos(v), 0.0];
            n = cross(ru, rv);
            norm_n = norm(n);
            if norm_n < 1e-14
                n = [0.0, 0.0, 1.0];
            else
                n = n / norm_n;
            end
            d = struct('r', [r * cos(v), r * sin(v), u], ...
                       'ru', ru, ...
                       'rv', rv, ...
                       'normal', n);
        end

        function n = normal(obj, u, v)
            d = obj.derivatives(u, v);
            n = d.normal;
        end

        function [E, F, G] = first_fundamental_form(obj, u, v)
            % Метрический тензор G = [g_zz, g_zv; g_vz, g_vv].
            [r, dr, ~] = obj.eval_at(u);
            E = 1.0 + dr^2;   % g_zz
            F = 0.0;          % g_zv = g_vz
            G = r^2;          % g_vv
        end

        function [L, M, N] = second_fundamental_form(obj, u, v)
            % Матрица кривизны B.
            [r, dr, d2r] = obj.eval_at(u);
            denom = sqrt(1.0 + dr^2);
            if denom < 1e-14
                L = 0.0; M = 0.0; N = 0.0;
                return;
            end
            L = -d2r / denom;   % b_zz
            M = 0.0;            % b_zv = b_vz
            N = r / denom;      % b_vv
        end

        function r_val = radius(obj, u)
            % Радиус параллели для заданной осевой координаты.
            r_val = ppval(obj.pp_r, double(u));
        end

        function [u_out, v_out] = uv_from_point(obj, point)
            % Обратное отображение: из 3D-точки в (u=z, v=atan2(y,x)).
            x = point(1); y = point(2); z = point(3);
            v_out = atan2(y, x);
            u_out = z;  % для поверхности вращения u=z напрямую
        end
    end

    methods (Access = private)
        function pp_d = build_derivative_spline(obj, z, r, order)
            % Построение сплайна для r^(order)(z).
            % Приоритет: fnder (Curve Fitting Toolbox), иначе численно.
            if order == 1
                if exist('fnder', 'file')
                    pp_d = fnder(obj.pp_r, 1);
                else
                    % Численное дифференцирование + spline
                    dr = gradient(r) ./ gradient(z);
                    pp_d = spline(z, dr);
                end
            elseif order == 2
                if exist('fnder', 'file')
                    pp_d = fnder(obj.pp_r, 2);
                else
                    dr = gradient(r) ./ gradient(z);
                    d2r = gradient(dr) ./ gradient(z);
                    pp_d = spline(z, d2r);
                end
            else
                error('order должен быть 1 или 2');
            end
        end
    end
end


