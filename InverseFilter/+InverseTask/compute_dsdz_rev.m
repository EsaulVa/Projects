function dsdz = compute_dsdz_rev(surf, u, v, du, dv, R, Rprime, DeltaZ, Percentage)
    r = surf.position(u, v);
    n = surf.normal(u, v);
    tau = (R - r) / norm(R - r);
    delta = dot(tau, n);
    if abs(delta) >= 1
        delta = sign(delta) * 0.9999;
    end
    alfa = 1 / sqrt(1 - delta^2);
    
    [L, M, N] = surf.second_fundamental_form(u, v);
    II = L*du^2 + 2*M*du*dv + N*dv^2;
    if abs(II) < 1e-6
        II = sign(II) * 1e-6;
    end
    denominator = norm(R - r) * II;
    if abs(denominator) < 1e-6
        denominator = sign(denominator) * 1e-6;
    end
    
    if DeltaZ > 0 && Percentage > 0 && Percentage < 100
        k = -log(1 - Percentage/100) / DeltaZ;
    else
        k = 0;
    end
    
    term1 = dot(Rprime, n) / denominator;
    term2 = k * delta / II;
    
    % Основная формула (минус перед term1)
    dsdz = -alfa * term1 + alfa * term2;
    
    % Для надёжности на время отладки – гарантия положительности
    % dsdz = abs(dsdz);
    
    % Отладка (можно закомментировать)
    fprintf('z=%.2f: delta=%.4e, II=%.4e, denom=%.4e, Rprime_n=%.4e, term1=%.4e, term2=%.4e, dsdz=%.6f\n', ...
        norm(R-r), delta, II, denominator, dot(Rprime,n), term1, term2, dsdz);
end