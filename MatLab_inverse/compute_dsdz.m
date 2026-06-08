function dsdz = compute_dsdz(u, v, du, dv, R, Rprime, surfFunc, DeltaZ, Percentage, varargin)
    % R - точка схода (3x1), Rprime - производная R по z
    [r, ru, rv, ruu, ruv, rvv] = surfFunc(u, v, varargin{:});
    n = computeNormal(ru, rv);
    
    % Касательная к линии укладки tau = (R - r) / |R-r|
    tau = (R - r) / norm(R - r);
    delta = dot(tau, n);
    if abs(delta) >= 1, delta = sign(delta)*0.9999; end
    alfa = 1 / sqrt(1 - delta^2);
    
    % Вторая квадратичная форма
    [L, M, N] = secondFundamentalForm(ru, rv, ruu, ruv, rvv, n);
    II = L*du^2 + 2*M*du*dv + N*dv^2;   % II может быть отрицательной?
    if II == 0, II = eps; end
    
    denominator = norm(R - r) * II;
    
    % Коэффициент k (по формуле из диссертации)
    if DeltaZ > 0 && Percentage > 0 && Percentage < 100
        k = -log(1 - Percentage/100) / DeltaZ;
    else
        k = 0;
    end
    
    dsdz = alfa * (dot(Rprime, n) / denominator) + alfa * k * delta / II;
end