function [du, dv] = computeTangentCoeffs_rev(surf, u, v, tau)
% tau – единичный касательный вектор в пространстве (3x1)
    [ru, rv] = surf.derivatives(u, v);
    n = surf.normal(u, v);
    delta = dot(tau, n);
    if abs(delta) >= 1
        delta = sign(delta) * 0.9999;
    end
    proj = (tau - delta*n) / sqrt(1 - delta^2);
    E = dot(ru, ru); F = dot(ru, rv); G = dot(rv, rv);
    rhs = [dot(proj, ru); dot(proj, rv)];
    M = [E, F; F, G];
    uv = M \ rhs;
    du = uv(1); dv = uv(2);
end