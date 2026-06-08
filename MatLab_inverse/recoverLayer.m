function [u_hist, v_hist, s_hist] = recoverLayer(R_func, Rprime_func, z_span, ...
    u0, v0, du0, dv0, surfFunc, DeltaZ, Percentage, varargin)
    % R_func(z) возвращает точку R(3x1), Rprime_func(z) - производную
    % z_span = [z0, z_end], u0,v0 - начальные координаты на поверхности
    % du0,dv0 - начальные du/ds, dv/ds (касательная к исходной линии)
    % Возвращает массивы u(z), v(z), s(z)
    
    z0 = z_span(1); z_end = z_span(2);
    h = (z_end - z0) / 1000;   % шаг интегрирования
    z = z0:h:z_end;
    N = length(z);
    u_hist = zeros(1,N); v_hist = zeros(1,N); s_hist = zeros(1,N);
    u_hist(1) = u0; v_hist(1) = v0; s_hist(1) = 0;
    u = u0; v = v0; s = 0;
    
    for i = 1:N-1
        R = R_func(z(i)); Rprime = Rprime_func(z(i));
        % Вычисляем du/ds, dv/ds в текущей точке
        [du_s, dv_s] = computeTangentCoeffs(u, v, (R - surfFunc(u, v, varargin{:}))/norm(R - surfFunc(u, v, varargin{:})), ...
            surfFunc, varargin{:});
        % ds/dz
        dsdz = compute_dsdz(u, v, du_s, dv_s, R, Rprime, surfFunc, DeltaZ, Percentage, varargin{:});
        % Производные по z
        dudz = du_s * dsdz;
        dvdz = dv_s * dsdz;
        
        % RK4
        k1_u = dudz; k1_v = dvdz; k1_s = dsdz;
        % промежуточные значения
        u_mid = u + h/2 * k1_u; v_mid = v + h/2 * k1_v; s_mid = s + h/2 * k1_s;
        R_mid = R_func(z(i)+h/2); Rprime_mid = Rprime_func(z(i)+h/2);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs(u_mid, v_mid, (R_mid - surfFunc(u_mid, v_mid, varargin{:}))/norm(R_mid - surfFunc(u_mid, v_mid, varargin{:})), ...
            surfFunc, varargin{:});
        dsdz_mid = compute_dsdz(u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, surfFunc, DeltaZ, Percentage, varargin{:});
        k2_u = du_s_mid * dsdz_mid; k2_v = dv_s_mid * dsdz_mid; k2_s = dsdz_mid;
        
        u_mid = u + h/2 * k2_u; v_mid = v + h/2 * k2_v; s_mid = s + h/2 * k2_s;
        R_mid = R_func(z(i)+h/2); Rprime_mid = Rprime_func(z(i)+h/2);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs(u_mid, v_mid, (R_mid - surfFunc(u_mid, v_mid, varargin{:}))/norm(R_mid - surfFunc(u_mid, v_mid, varargin{:})), ...
            surfFunc, varargin{:});
        dsdz_mid = compute_dsdz(u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, surfFunc, DeltaZ, Percentage, varargin{:});
        k3_u = du_s_mid * dsdz_mid; k3_v = dv_s_mid * dsdz_mid; k3_s = dsdz_mid;
        
        u_next = u + h * k3_u; v_next = v + h * k3_v; s_next = s + h * k3_s;
        R_next = R_func(z(i)+h); Rprime_next = Rprime_func(z(i)+h);
        [du_s_next, dv_s_next] = computeTangentCoeffs(u_next, v_next, (R_next - surfFunc(u_next, v_next, varargin{:}))/norm(R_next - surfFunc(u_next, v_next, varargin{:})), ...
            surfFunc, varargin{:});
        dsdz_next = compute_dsdz(u_next, v_next, du_s_next, dv_s_next, R_next, Rprime_next, surfFunc, DeltaZ, Percentage, varargin{:});
        k4_u = du_s_next * dsdz_next; k4_v = dv_s_next * dsdz_next; k4_s = dsdz_next;
        
        u = u + h/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u);
        v = v + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v);
        s = s + h/6 * (k1_s + 2*k2_s + 2*k3_s + k4_s);
        
        u_hist(i+1) = u; v_hist(i+1) = v; s_hist(i+1) = s;
    end
end