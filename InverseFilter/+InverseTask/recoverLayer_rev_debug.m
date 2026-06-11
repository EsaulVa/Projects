function [u_hist, v_hist, s_hist, z_hist, dist_arr, bad_indices] = recoverLayer_rev_debug(...
    surf, R_traj, z_span, u0, v0, du0, dv0, DeltaZ, Percentage, Nsteps)
    z0 = z_span(1);
    z_end = z_span(2);
    z_hist = linspace(z0, z_end, Nsteps+1);
    h = z_hist(2) - z_hist(1);
    u_hist = zeros(1, Nsteps+1);
    v_hist = zeros(1, Nsteps+1);
    s_hist = zeros(1, Nsteps+1);
    dist_arr = zeros(1, Nsteps+1);
    bad_indices = [];
    
    u = u0; v = v0; s = 0;
    u_hist(1) = u; v_hist(1) = v; s_hist(1) = s;
    dist_arr(1) = 0;
    
    for i = 1:Nsteps
        z = z_hist(i);
        R = R_traj.getPoint(z);
        Rprime = R_traj.getTangent(z);
        r = surf.position(u, v);
        dist = norm(R - r);
        dist_arr(i+1) = dist;
        if dist > 500 && isempty(bad_indices)
            bad_indices = i+1;   % запоминаем номер шага, где начался разрыв
            fprintf('Расходимость на шаге %d, z=%.2f, dist=%.1f\n', i, z, dist);
        end
        
        tau = (R - r) / dist;
        [du_s, dv_s] = computeTangentCoeffs_rev(surf, u, v, tau);
        dsdz = compute_dsdz_rev(surf, u, v, du_s, dv_s, R, Rprime, DeltaZ, Percentage);
        dudz = du_s * dsdz;
        dvdz = dv_s * dsdz;
        
        % RK4 (как раньше)
        k1_u = dudz; k1_v = dvdz; k1_s = dsdz;
        u_mid = u + h/2*k1_u; v_mid = v + h/2*k1_v; s_mid = s + h/2*k1_s;
        R_mid = R_traj.getPoint(z + h/2); Rprime_mid = R_traj.getTangent(z + h/2);
        r_mid = surf.position(u_mid, v_mid);
        tau_mid = (R_mid - r_mid) / norm(R_mid - r_mid);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs_rev(surf, u_mid, v_mid, tau_mid);
        dsdz_mid = compute_dsdz_rev(surf, u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, DeltaZ, Percentage);
        k2_u = du_s_mid * dsdz_mid; k2_v = dv_s_mid * dsdz_mid; k2_s = dsdz_mid;
        
        u_mid = u + h/2*k2_u; v_mid = v + h/2*k2_v; s_mid = s + h/2*k2_s;
        R_mid = R_traj.getPoint(z + h/2); Rprime_mid = R_traj.getTangent(z + h/2);
        r_mid = surf.position(u_mid, v_mid);
        tau_mid = (R_mid - r_mid) / norm(R_mid - r_mid);
        [du_s_mid, dv_s_mid] = computeTangentCoeffs_rev(surf, u_mid, v_mid, tau_mid);
        dsdz_mid = compute_dsdz_rev(surf, u_mid, v_mid, du_s_mid, dv_s_mid, R_mid, Rprime_mid, DeltaZ, Percentage);
        k3_u = du_s_mid * dsdz_mid; k3_v = dv_s_mid * dsdz_mid; k3_s = dsdz_mid;
        
        u_next = u + h*k3_u; v_next = v + h*k3_v; s_next = s + h*k3_s;
        R_next = R_traj.getPoint(z + h); Rprime_next = R_traj.getTangent(z + h);
        r_next = surf.position(u_next, v_next);
        tau_next = (R_next - r_next) / norm(R_next - r_next);
        [du_s_next, dv_s_next] = computeTangentCoeffs_rev(surf, u_next, v_next, tau_next);
        dsdz_next = compute_dsdz_rev(surf, u_next, v_next, du_s_next, dv_s_next, R_next, Rprime_next, DeltaZ, Percentage);
        k4_u = du_s_next * dsdz_next; k4_v = dv_s_next * dsdz_next; k4_s = dsdz_next;
        
        u = u + h/6*(k1_u + 2*k2_u + 2*k3_u + k4_u);
        v = v + h/6*(k1_v + 2*k2_v + 2*k3_v + k4_v);
        s = s + h/6*(k1_s + 2*k2_s + 2*k3_s + k4_s);
        
        u_hist(i+1) = u; v_hist(i+1) = v; s_hist(i+1) = s;
        
        % Остановка при сильной расходимости (необязательно)
        if dist > 1000
            fprintf('Остановка из-за слишком большого расстояния на шаге %d\n', i);
            u_hist = u_hist(1:i+1); v_hist = v_hist(1:i+1); s_hist = s_hist(1:i+1);
            z_hist = z_hist(1:i+1); dist_arr = dist_arr(1:i+1);
            break;
        end
    end
end