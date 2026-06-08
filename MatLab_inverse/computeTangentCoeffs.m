function [du, dv] = computeTangentCoeffs(surf, u, v, tau)
    % tau - единичный касательный вектор в пространстве
    n = surf.getNormal(u, v);
    delta = dot(tau, n);
    if abs(delta) >= 1
        delta = sign(delta) * 0.9999;
    end
    proj = (tau - delta * n) / sqrt(1 - delta^2);
    
    [ru, rv] = surf.getFirstDerivatives(u, v);
    [E, F, G] = surf.getFirstFundamental(u, v);
    rhs = [dot(proj, ru); dot(proj, rv)];
    M = [E, F; F, G];
    uv = M \ rhs;
    du = uv(1); dv = uv(2);
end