classdef DiffusedRevolutionSurface < handle
    %DIFFUSEDREVOLUTIONSURFACE Сглаженная поверхность вращения.
    %   Диффузия профиля меридиана (r,z) с произвольными ГУ на концах.
    %   Параметр u = s (длина дуги меридиана).
    %   Опционально: сохранение параметризации по z (preserve_z_parameter).
    %
    %   Дополнительно: доступ к производным радиуса r(z) через radius_deriv, radius_deriv2.

    properties (Access = private)
        base_surface    % исходная поверхность (DiscreteRevolutionSurface)
        pp_r            % spline: r(s)
        pp_z            % spline: z(s)
        pp_dr           % r'(s)
        pp_dz           % z'(s)
        pp_d2r          % r''(s)
        pp_d2z          % z''(s)
        % Для preserve_z:
        pp_s_by_z       % s(z)
        pp_r_by_z       % r(z) на сглаженном профиле
        pp_dr_by_z      % dr/dz (сплайн)
        pp_d2r_by_z     % d2r/dz2 (сплайн)
    end

    properties (SetAccess = immutable)
        N               % число точек дискретизации
        tau             % шаг диффузии
        n_steps         % число шагов
        bc_left         % ГУ слева
        bc_right        % ГУ справа
        preserve_z      % флаг сохранения z-параметризации
        u_min           % min(s)
        u_max           % max(s)
        z_min           % min(z_smooth) (при preserve_z)
        z_max           % max(z_smooth) (при preserve_z)
        r_raw           % исходный радиус (для справки)
        z_raw           % исходная высота (для справки)
        r_smooth        % сглаженный радиус (для справки)
        z_smooth        % сглаженная высота (для справки)
        s_grid          % сетка по дуге
    end

    methods
        function obj = DiffusedRevolutionSurface(base_surface, N, tau, n_steps, ...
                                                  bc_left, bc_right, varargin)
            % Конструктор.
            % Дополнительные параметры (Name-Value):
            %   PreserveZParameter — false/true
            %   SaveMeridianPath   — строка пути к CSV (или '')

            p = inputParser;
            addRequired(p, 'base_surface');
            addRequired(p, 'N', @(x) isnumeric(x) && x >= 2);
            addRequired(p, 'tau', @isnumeric);
            addRequired(p, 'n_steps', @isnumeric);
            addRequired(p, 'bc_left');
            addRequired(p, 'bc_right');
            addParameter(p, 'PreserveZParameter', false, @islogical);
            addParameter(p, 'SaveMeridianPath', '', @ischar);
            parse(p, base_surface, N, tau, n_steps, bc_left, bc_right, varargin{:});

            obj.base_surface = base_surface;
            obj.N = N;
            obj.tau = tau;
            obj.n_steps = n_steps;
            obj.bc_left = bc_left;
            obj.bc_right = bc_right;
            obj.preserve_z = p.Results.PreserveZParameter;

            % --- 1. Извлечь меридиан в исходном параметре u = z ---
            u_raw = linspace(base_surface.u_min, base_surface.u_max, N)';
            pts = zeros(N, 3);
            for i = 1:N
                pts(i, :) = base_surface.position(u_raw(i), 0.0);
            end
            obj.r_raw = sqrt(pts(:, 1).^2 + pts(:, 2).^2);
            obj.z_raw = pts(:, 3);

            % --- 2. Перепараметризация по длине дуги s ---
            ds = sqrt(diff(obj.r_raw).^2 + diff(obj.z_raw).^2);
            s = [0; cumsum(ds)];
            obj.u_min = 0.0;
            obj.u_max = s(end);
            obj.s_grid = s;

            % --- 3. Автоподстановка Дирихле (концы не уезжают) ---
            bc_l_r = obj.auto_dirichlet(bc_left, obj.r_raw(1));
            bc_r_r = obj.auto_dirichlet(bc_right, obj.r_raw(end));
            bc_l_z = obj.auto_dirichlet(bc_left, obj.z_raw(1));
            bc_r_z = obj.auto_dirichlet(bc_right, obj.z_raw(end));

            % --- 4. Диффузия r(s) и z(s) ---
            r_smooth = obj.diffuse(s, obj.r_raw, tau, n_steps, bc_l_r, bc_r_r);
            z_smooth = obj.diffuse(s, obj.z_raw, tau, n_steps, bc_l_z, bc_r_z);
            obj.r_smooth = r_smooth;
            obj.z_smooth = z_smooth;

            % --- 5. Сплайны по s ---
            obj.pp_r = spline(s, r_smooth);
            obj.pp_z = spline(s, z_smooth);
            if exist('fnder', 'file')
                obj.pp_dr = fnder(obj.pp_r, 1);
                obj.pp_dz = fnder(obj.pp_z, 1);
                obj.pp_d2r = fnder(obj.pp_r, 2);
                obj.pp_d2z = fnder(obj.pp_z, 2);
            else
                obj.pp_dr = obj.num_derivative_spline(s, r_smooth, 1);
                obj.pp_dz = obj.num_derivative_spline(s, z_smooth, 1);
                obj.pp_d2r = obj.num_derivative_spline(s, r_smooth, 2);
                obj.pp_d2z = obj.num_derivative_spline(s, z_smooth, 2);
            end

            % --- 6. Опционально: сохранить z-параметризацию и построить сплайны r(z) ---
            if obj.preserve_z
                dz_ds = gradient(z_smooth) ./ gradient(s);
                if any(dz_ds <= 0)
                    warning('DiffusedRevolutionSurface:z_nonmonotonic', ...
                        'z(s) не монотонна. Интерполяция s(z) может быть неоднозначной.');
                end
                [z_u, idx_u] = unique(z_smooth, 'stable');
                if length(z_u) < length(z_smooth)
                    warning('DiffusedRevolutionSurface:z_duplicates', ...
                        'Удалены дубли z_smooth для интерполяции.');
                end
                obj.z_min = z_smooth(1);
                obj.z_max = z_smooth(end);
                obj.pp_s_by_z = interp1(z_u, s(idx_u), 'cubic', 'pp');
                obj.pp_r_by_z = interp1(z_u, r_smooth(idx_u), 'cubic', 'pp');
                
                % Построение производных r(z) с помощью fnder (или численно)
                if exist('fnder', 'file')
                    obj.pp_dr_by_z = fnder(obj.pp_r_by_z, 1);
                    obj.pp_d2r_by_z = fnder(obj.pp_r_by_z, 2);
                else
                    % численное дифференцирование сплайна
                    z_der = linspace(obj.z_min, obj.z_max, 1000);
                    r_der = ppval(obj.pp_r_by_z, z_der);
                    dr_num = gradient(r_der) ./ gradient(z_der);
                    obj.pp_dr_by_z = spline(z_der, dr_num);
                    d2r_num = gradient(dr_num) ./ gradient(z_der);
                    obj.pp_d2r_by_z = spline(z_der, d2r_num);
                end
            end

            % --- 7. Сохранение CSV (опционально) ---
            if ~isempty(p.Results.SaveMeridianPath)
                obj.save_meridian(p.Results.SaveMeridianPath, s);
            end
        end

        % ------------------------------------------------------------------
        % Методы для совместимости с интерфейсом Surface (для computeTangentCoeffs, recoverLayer)
        % ------------------------------------------------------------------
         function r = getPoint(obj, u, v)
            r = obj.position(u, v);
            r = r(:);
         end

        function n = getNormal(obj, u, v)
            n = obj.normal(u, v);
            n = n(:);
        end
        
        function [ru, rv] = getFirstDerivatives(obj, u, v)
            d = obj.derivatives(u, v);
            ru = d.ru; rv = d.rv;
        end
        
        function [E, F, G] = getFirstFundamental(obj, u, v)
            [E, F, G] = obj.first_fundamental_form(u, v);
        end
        
        function [L, M, N] = getSecondFundamental(obj, u, v)
            [L, M, N] = obj.second_fundamental_form(u, v);
        end
        
       
        % ------------------------------------------------------------------
        % Доступ по z (только при preserve_z = true)
        % ------------------------------------------------------------------
        function s_val = s_from_z(obj, z)
            if ~obj.preserve_z
                error('DiffusedRevolutionSurface:preserve_z_false', ...
                    'preserve_z_parameter = false. Используйте position(s,v).');
            end
            z = max(obj.z_min, min(obj.z_max, z));
            s_val = ppval(obj.pp_s_by_z, z);
        end

        function r_val = radius(obj, z)
            if ~obj.preserve_z
                error('DiffusedRevolutionSurface:preserve_z_false', ...
                    'preserve_z_parameter = false. Используйте position(s,v).');
            end
            z = max(obj.z_min, min(obj.z_max, z));
            r_val = ppval(obj.pp_r_by_z, z);
        end

        function dr_val = radius_deriv(obj, z)
            % Первая производная dr/dz
            if ~obj.preserve_z
                error('DiffusedRevolutionSurface:preserve_z_false', ...
                    'preserve_z_parameter = false для доступа к производным r(z).');
            end
            z = max(obj.z_min, min(obj.z_max, z));
            dr_val = ppval(obj.pp_dr_by_z, z);
        end

        function d2r_val = radius_deriv2(obj, z)
            % Вторая производная d2r/dz2
            if ~obj.preserve_z
                error('DiffusedRevolutionSurface:preserve_z_false', ...
                    'preserve_z_parameter = false для доступа к производным r(z).');
            end
            z = max(obj.z_min, min(obj.z_max, z));
            d2r_val = ppval(obj.pp_d2r_by_z, z);
        end

        function [r_val, dr_val, d2r_val] = eval_at(obj, z)
            % Возвращает r(z), r'(z), r''(z)
            if ~obj.preserve_z
                error('DiffusedRevolutionSurface:preserve_z_false', ...
                    'preserve_z_parameter = false. Нет сплайна r(z).');
            end
            z = max(obj.z_min, min(obj.z_max, z));
            r_val = ppval(obj.pp_r_by_z, z);
            dr_val = ppval(obj.pp_dr_by_z, z);
            d2r_val = ppval(obj.pp_d2r_by_z, z);
        end

        function pt = position_by_z(obj, z, v)
            r = obj.radius(z);
            pt = [r * cos(v), r * sin(v), z];
        end

        function d = derivatives_by_z(obj, z, v)
            [r, dr, ~] = obj.eval_at(z);
            ru = [dr * cos(v), dr * sin(v), 1.0];
            rv = [-r * sin(v), r * cos(v), 0.0];
            n = cross(ru, rv);
            norm_n = norm(n);
            if norm_n < 1e-14
                n = [0.0, 0.0, 1.0];
            else
                n = n / norm_n;
            end
            d = struct('r', [r * cos(v), r * sin(v), z], ...
                       'ru', ru, 'rv', rv, 'normal', n);
        end

        % ------------------------------------------------------------------
        % Стандартный интерфейс по длине дуги s
        % ------------------------------------------------------------------
        function pt = position(obj, s, v)
            r = ppval(obj.pp_r, s);
            z = ppval(obj.pp_z, s);
            pt = [r * cos(v), r * sin(v), z];
        end

        % function d = derivatives(obj, s, v)
        %     r = ppval(obj.pp_r, s);
        %     dr = ppval(obj.pp_dr, s);
        %     dz = ppval(obj.pp_dz, s);
        %     ru = [dr * cos(v), dr * sin(v), dz];
        %     rv = [-r * sin(v), r * cos(v), 0.0];
        %     n = cross(ru, rv);
        %     norm_n = norm(n);
        %     if norm_n < 1e-14
        %         n = [0.0, 0.0, 1.0];
        %     else
        %         n = n / norm_n;
        %     end
        %     d = struct('r', [r * cos(v), r * sin(v), z], ...
        %                'ru', ru, 'rv', rv, 'normal', n);
        % end
        function d = derivatives(obj, s, v)
            r = ppval(obj.pp_r, s);
            z = ppval(obj.pp_z, s);   % <-- добавить эту строку
            dr = ppval(obj.pp_dr, s);
            dz = ppval(obj.pp_dz, s);
            ru = [dr * cos(v), dr * sin(v), dz];
            rv = [-r * sin(v), r * cos(v), 0.0];
            n = cross(ru, rv);
            norm_n = norm(n);
            if norm_n < 1e-14
                n = [0.0, 0.0, 1.0];
            else
                n = n / norm_n;
            end
            d = struct('r', [r * cos(v), r * sin(v), z], ...
                       'ru', ru, 'rv', rv, 'normal', n);
        end
        function n = normal(obj, s, v)
            d = obj.derivatives(s, v);
            n = d.normal;
        end

        function [E, F, G] = first_fundamental_form(obj, s, v)
            dr = ppval(obj.pp_dr, s);
            dz = ppval(obj.pp_dz, s);
            r = ppval(obj.pp_r, s);
            E = dr^2 + dz^2;   % = 1, если s натуральный
            F = 0.0;
            G = r^2;
        end

        function [L, M, N] = second_fundamental_form(obj, s, v)
            dr = ppval(obj.pp_dr, s);
            dz = ppval(obj.pp_dz, s);
            d2r = ppval(obj.pp_d2r, s);
            d2z = ppval(obj.pp_d2z, s);
            r = ppval(obj.pp_r, s);
            denom = sqrt(dr^2 + dz^2);
            if denom < 1e-14
                L = 0.0; M = 0.0; N = 0.0;
                return;
            end
            d = obj.derivatives(s, v);
            n = d.normal;
            r_uu = [d2r * cos(v), d2r * sin(v), d2z];
            r_vv = [-r * cos(v), -r * sin(v), 0.0];
            L = dot(r_uu, n);
            M = dot([0, 0, 0], n); % r_uv = 0 для поверхности вращения
            N = dot(r_vv, n);
        end
    end

    methods (Access = private)
        % ------------------------------------------------------------------
        % Диффузия: неявная прогонка
        % ------------------------------------------------------------------
        function h = diffuse(obj, s, h0, tau, n_steps, bc_left, bc_right)
            N = length(s);
            s_uni = linspace(s(1), s(end), N)';
            h_uni = interp1(s, h0, s_uni, 'linear', 'extrap');

            % Радиус параллели на равномерной сетке (из исходной поверхности)
            u_uni = linspace(obj.base_surface.u_min, obj.base_surface.u_max, N)';
            r_uni = zeros(N, 1);
            for i = 1:N
                pt = obj.base_surface.position(u_uni(i), 0.0);
                r_uni(i) = sqrt(pt(1)^2 + pt(2)^2);
            end

            ds = (s_uni(end) - s_uni(1)) / (N - 1);

            % Коэффициент a(s) = r'(s) / r(s)
            r_prime = zeros(N, 1);
            r_prime(2:N-1) = (r_uni(3:N) - r_uni(1:N-2)) / (2 * ds);
            r_prime(1) = (r_uni(2) - r_uni(1)) / ds;
            r_prime(N) = (r_uni(N) - r_uni(N-1)) / ds;
            a = r_prime ./ max(r_uni, 1e-12);

            % Коэффициенты неявной схемы
            alpha = tau / ds^2 - tau * a / (2 * ds);
            beta = ones(N, 1) * (1.0 + 2.0 * tau / ds^2);
            gamma = tau / ds^2 + tau * a / (2 * ds);

            % Трёхдиагональная матрица (внутренние точки)
            lower = [-alpha(2:N); 0];      % поддиагональ
            upper = [0; -gamma(1:N-1)];    % наддиагональ
            M = spdiags([lower, beta, upper], -1:1, N, N);

            % Применение ГУ к матрице
            [M, ~] = obj.apply_bc_matrix(M, bc_left, bc_right, ds);

            % Шаги по времени
            h = h_uni;
            for step = 1:n_steps
                rhs = h;
                rhs = obj.apply_bc_rhs(rhs, bc_left, bc_right, ds);
                h = M \ rhs;
            end

            % Вернуть на исходную сетку s
            h = interp1(s_uni, h, s, 'linear', 'extrap');
        end

        % ------------------------------------------------------------------
        % Граничные условия: матрица
        % ------------------------------------------------------------------
        function [M, rhs] = apply_bc_matrix(obj, M, bc_left, bc_right, ds)
            N = size(M, 1);
            rhs = zeros(N, 1); % placeholder

            % Левая граница
            switch lower(bc_left.kind)
                case 'dirichlet'
                    M(1, 1) = 1.0; M(1, 2) = 0.0;
                case 'neumann'
                    M(1, 1) = -1.0; M(1, 2) = 1.0;
                case 'robin'
                    a = bc_left.coeff_a; b = bc_left.coeff_b;
                    M(1, 1) = a + b / ds; M(1, 2) = -b / ds;
            end

            % Правая граница
            switch lower(bc_right.kind)
                case 'dirichlet'
                    M(N, N) = 1.0; M(N, N-1) = 0.0;
                case 'neumann'
                    M(N, N-1) = -1.0; M(N, N) = 1.0;
                case 'robin'
                    a = bc_right.coeff_a; b = bc_right.coeff_b;
                    M(N, N-1) = -b / ds; M(N, N) = a + b / ds;
            end
        end

        % ------------------------------------------------------------------
        % Граничные условия: правая часть
        % ------------------------------------------------------------------
        function rhs = apply_bc_rhs(obj, rhs, bc_left, bc_right, ds)
            N = length(rhs);
            switch lower(bc_left.kind)
                case 'dirichlet'
                    rhs(1) = bc_left.value;
                case 'neumann'
                    rhs(1) = bc_left.value * ds;
                case 'robin'
                    rhs(1) = bc_left.value;
            end
            switch lower(bc_right.kind)
                case 'dirichlet'
                    rhs(N) = bc_right.value;
                case 'neumann'
                    rhs(N) = bc_right.value * ds;
                case 'robin'
                    rhs(N) = bc_right.value;
            end
        end

        % ------------------------------------------------------------------
        % Автоподстановка Дирихле
        % ------------------------------------------------------------------
        function bc = auto_dirichlet(obj, bc, end_value)
            if strcmpi(bc.kind, 'dirichlet')
                bc = bc.with_value(end_value);
            end
        end

        % ------------------------------------------------------------------
        % Численные производные сплайна (fallback без fnder)
        % ------------------------------------------------------------------
        function pp_d = num_derivative_spline(obj, x, y, order)
            if order == 1
                dy = gradient(y) ./ gradient(x);
                pp_d = spline(x, dy);
            elseif order == 2
                dy = gradient(y) ./ gradient(x);
                d2y = gradient(dy) ./ gradient(x);
                pp_d = spline(x, d2y);
            else
                error('order должен быть 1 или 2');
            end
        end

        % ------------------------------------------------------------------
        % Сохранение меридиана в CSV
        % ------------------------------------------------------------------
        function save_meridian(obj, path, s)
            % Кривизна оригинала (параметрически)
            dr = gradient(obj.r_raw) ./ gradient(s);
            dz = gradient(obj.z_raw) ./ gradient(s);
            d2r = gradient(dr) ./ gradient(s);
            d2z = gradient(dz) ./ gradient(s);
            k_orig = abs(d2r .* dz - dr .* d2z) ./ (dr.^2 + dz.^2 + 1e-12).^1.5;

            % Кривизна сглаженного (аналитически через сплайн)
            dr_s = ppval(obj.pp_dr, s);
            dz_s = ppval(obj.pp_dz, s);
            d2r_s = ppval(obj.pp_d2r, s);
            d2z_s = ppval(obj.pp_d2z, s);
            k_smooth = abs(d2r_s .* dz_s - dr_s .* d2z_s) ./ ...
                       (dr_s.^2 + dz_s.^2 + 1e-12).^1.5;

            T = table(s, obj.z_raw, obj.r_raw, obj.z_smooth, obj.r_smooth, ...
                      dr, dz, dr_s, dz_s, k_orig, k_smooth, ...
                      'VariableNames', {'s', 'z_orig', 'r_orig', 'z_smooth', 'r_smooth', ...
                                        'dr_orig', 'dz_orig', 'dr_smooth', 'dz_smooth', ...
                                        'curvature_orig', 'curvature_smooth'});
            writetable(T, path);
            fprintf('[DiffusedRevolutionSurface] Меридиан сохранён: %s\n', path);
        end
    end
end