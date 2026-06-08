function [u_hist, v_hist, s_hist, z_hist] = recoverLayer(surf, R_func, Rprime_func, z_span, ...
    u0, v0, du0, dv0, DeltaZ, Percentage)
% Восстанавливает линию укладки r(u(z),v(z)) по траектории точки схода R(z)
% Используется метод Рунге-Кутты 4-го порядка с фиксированным шагом.
% Вход:
%   surf        - объект класса Surface (методы getPoint, getNormal, getFirstDerivatives, getSecondFundamental)
%   R_func      - function_handle: R = R_func(z) возвращает вектор 3x1
%   Rprime_func - function_handle: dRdz = Rprime_func(z) возвращает вектор 3x1
%   z_span      - [z0, z_end]
%   u0, v0      - начальные криволинейные координаты на поверхности
%   du0, dv0    - начальные du/ds, dv/ds (касательная к линии укладки)
%   DeltaZ, Percentage - параметры для вычисления коэффициента k (см. диссертацию)
% Выход:
%   u_hist, v_hist, s_hist - массивы значений вдоль z
%   z_hist - массив узлов, в которых вычислено решение

    z0 = z_span(1);
    z_end = z_span(2);
    
    Nsteps = 1000;   % число шагов (можно изменить)
    z_hist = linspace(z0, z_end, Nsteps+1);
    h = z_hist(2) - z_hist(1);
    u_hist = zeros(1, Nsteps+1);
    v_hist = zeros(1, Nsteps+1);
    s_hist = zeros(1, Nsteps+1);
    
    u = u0; v = v0; s = 0;
    u_hist(1) = u; v_hist(1) = v; s_hist(1) = s;
    
    for i = 1:Nsteps
        z = z_hist(i);
        R = R_func(z);
        Rprime = Rprime_func(z);
        
        r = surf.getPoint(u, v);
        tau = (R - r) / norm(R - r);
        [du_s, dv_s] = computeTangentCoeffs(surf, u, v, tau);
        dsdz = compute_dsdz(surf, u, v, du_s, dv_s, R, Rprime, DeltaZ, Percentage);
        dudz = du_s * dsdz;
        dvdz = dv_s * dsdz;
        
        % --- RK4 ---
        k1_u = dudz; k1_v = dvdz; k1_s = dsdz;
        
        % k2
        u_mid = u + h/2 * k1_u;
        v_mid = v + h/2 * k1_v;
        s_mid = s + h/2 * k1_s;
        R_mid = R_func(z + h/2);
        Rprime_mid = Rprime_func(z + h/2);
        r_mid = surf.getPoint(u_mid, v_mid);
        tau_mid = (R_mid - r_mid) / norm(R_mid - r_mid);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs(surf, u_mid, v_mid, tau_mid);
        dsdz_mid = compute_dsdz(surf, u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, DeltaZ, Percentage);
        k2_u = du_s_mid * dsdz_mid;
        k2_v = dv_s_mid * dsdz_mid;
        k2_s = dsdz_mid;
        
        % k3
        u_mid = u + h/2 * k2_u;
        v_mid = v + h/2 * k2_v;
        s_mid = s + h/2 * k2_s;
        R_mid = R_func(z + h/2);
        Rprime_mid = Rprime_func(z + h/2);
        r_mid = surf.getPoint(u_mid, v_mid);
        tau_mid = (R_mid - r_mid) / norm(R_mid - r_mid);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs(surf, u_mid, v_mid, tau_mid);
        dsdz_mid = compute_dsdz(surf, u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, DeltaZ, Percentage);
        k3_u = du_s_mid * dsdz_mid;
        k3_v = dv_s_mid * dsdz_mid;
        k3_s = dsdz_mid;
        
        % k4
        u_next = u + h * k3_u;
        v_next = v + h * k3_v;
        s_next = s + h * k3_s;
        R_next = R_func(z + h);
        Rprime_next = Rprime_func(z + h);
        r_next = surf.getPoint(u_next, v_next);
        tau_next = (R_next - r_next) / norm(R_next - r_next);
        [du_s_next, dv_s_next] = computeTangentCoeffs(surf, u_next, v_next, tau_next);
        dsdz_next = compute_dsdz(surf, u_next, v_next, du_s_next, dv_s_next, R_next, Rprime_next, DeltaZ, Percentage);
        k4_u = du_s_next * dsdz_next;
        k4_v = dv_s_next * dsdz_next;
        k4_s = dsdz_next;
        
        u = u + h/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u);
        v = v + h/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v);
        s = s + h/6 * (k1_s + 2*k2_s + 2*k3_s + k4_s);
        
        u_hist(i+1) = u;
        v_hist(i+1) = v;
        s_hist(i+1) = s;
    end
end