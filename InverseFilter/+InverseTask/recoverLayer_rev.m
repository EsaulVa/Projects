function [u_hist, v_hist, s_hist, z_hist] = recoverLayer_rev(surf, R_traj, z_span, ...
    u0, v0, du0, dv0, DeltaZ, Percentage, Nsteps)
% Восстанавливает линию укладки на поверхности вращения методом RK4.
    z0 = z_span(1);
    z_end = z_span(2);
    z_hist = linspace(z0, z_end, Nsteps+1);
    h = z_hist(2) - z_hist(1);
    u_hist = zeros(1, Nsteps+1);
    v_hist = zeros(1, Nsteps+1);
    s_hist = zeros(1, Nsteps+1);
    
    u = u0; v = v0; s = 0;
    u_hist(1) = u; v_hist(1) = v; s_hist(1) = s;
    
    for i = 1:Nsteps
        z = z_hist(i);
        R = R_traj.getPoint(z);
        Rprime = R_traj.getTangent(z);
        
        % Текущая точка на поверхности
        r = surf.position(u, v);
        tau = (R - r) / norm(R - r);
        
        % du/ds, dv/ds
        [du_s, dv_s] = computeTangentCoeffs_rev(surf, u, v, tau);
        
        % ds/dz
        dsdz = compute_dsdz_rev(surf, u, v, du_s, dv_s, R, Rprime, DeltaZ, Percentage);
        
        
        % Производные по z
        dudz = du_s * dsdz;
        dvdz = dv_s * dsdz;
        
        % RK4
        k1_u = dudz; k1_v = dvdz; k1_s = dsdz;
        
        u_mid = u + h/2*k1_u; v_mid = v + h/2*k1_v; s_mid = s + h/2*k1_s;
        % Защита от выхода за границы u
        if u_mid < surf.u_min || u_mid > surf.u_max
            warning('u_mid=%.3f вышла за границы на z=%.3f, остановка', u_mid, z);
            u_hist = u_hist(1:i); v_hist = v_hist(1:i); s_hist = s_hist(1:i); z_hist = z_hist(1:i);
            return;
        end
        R_mid = R_traj.getPoint(z + h/2);
        Rprime_mid = R_traj.getTangent(z + h/2);
        r_mid = surf.position(u_mid, v_mid);
        tau_mid = (R_mid - r_mid) / norm(R_mid - r_mid);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs_rev(surf, u_mid, v_mid, tau_mid);
        dsdz_mid = compute_dsdz_rev(surf, u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, DeltaZ, Percentage);
        k2_u = du_s_mid * dsdz_mid; k2_v = dv_s_mid * dsdz_mid; k2_s = dsdz_mid;
        
        u_mid = u + h/2*k2_u; v_mid = v + h/2*k2_v; s_mid = s + h/2*k2_s;
        if u_mid < surf.u_min || u_mid > surf.u_max
            warning('u_mid=%.3f вышла за границы (2), остановка', u_mid);
            u_hist = u_hist(1:i); v_hist = v_hist(1:i); s_hist = s_hist(1:i); z_hist = z_hist(1:i);
            return;
        end
        R_mid = R_traj.getPoint(z + h/2);
        Rprime_mid = R_traj.getTangent(z + h/2);
        r_mid = surf.position(u_mid, v_mid);
        tau_mid = (R_mid - r_mid) / norm(R_mid - r_mid);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs_rev(surf, u_mid, v_mid, tau_mid);
        dsdz_mid = compute_dsdz_rev(surf, u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, DeltaZ, Percentage);
        k3_u = du_s_mid * dsdz_mid; k3_v = dv_s_mid * dsdz_mid; k3_s = dsdz_mid;
        
        u_next = u + h*k3_u; v_next = v + h*k3_v; s_next = s + h*k3_s;
        if u_next < surf.u_min || u_next > surf.u_max
            warning('u_next=%.3f вышла за границы, остановка', u_next);
            u_hist = u_hist(1:i); v_hist = v_hist(1:i); s_hist = s_hist(1:i); z_hist = z_hist(1:i);
            return;
        end
        R_next = R_traj.getPoint(z + h);
        Rprime_next = R_traj.getTangent(z + h);
        r_next = surf.position(u_next, v_next);
        tau_next = (R_next - r_next) / norm(R_next - r_next);
        [du_s_next, dv_s_next] = computeTangentCoeffs_rev(surf, u_next, v_next, tau_next);
        dsdz_next = compute_dsdz_rev(surf, u_next, v_next, du_s_next, dv_s_next, R_next, Rprime_next, DeltaZ, Percentage);
        k4_u = du_s_next * dsdz_next; k4_v = dv_s_next * dsdz_next; k4_s = dsdz_next;
        
        u = u + h/6*(k1_u + 2*k2_u + 2*k3_u + k4_u);
        v = v + h/6*(k1_v + 2*k2_v + 2*k3_v + k4_v);
        s = s + h/6*(k1_s + 2*k2_s + 2*k3_s + k4_s);
        
        u_hist(i+1) = u; v_hist(i+1) = v; s_hist(i+1) = s;
        fprintf('z=%.2f, dsdz=%.6f, s=%.4f\n', z, dsdz, s);
    end
end