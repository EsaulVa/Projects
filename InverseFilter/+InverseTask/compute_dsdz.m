function dsdz = compute_dsdz(surf, u, v, du, dv, R, Rprime, DeltaZ, Percentage)
    r = surf.getPoint(u, v);
    n = surf.getNormal(u, v);
    tau = (R - r) / norm(R - r);
    delta = dot(tau, n);
    if abs(delta) >= 1
        delta = sign(delta) * 0.9999;
    end
    alfa = 1 / sqrt(1 - delta^2);
    
    [L, M, N] = surf.getSecondFundamental(u, v);
    II = L*du^2 + 2*M*du*dv + N*dv^2;
    if II == 0, II = eps; end
    
    denominator = norm(R - r) * II;
    
    if DeltaZ > 0 && Percentage > 0 && Percentage < 100
        k = -log(1 - Percentage/100) / DeltaZ;
    else
        k = 0;
    end
    
    dsdz = alfa * (dot(Rprime, n) / denominator) + alfa * k * delta / II;
end