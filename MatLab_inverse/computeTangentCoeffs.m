function [du, dv] = computeTangentCoeffs(u, v, tau, surfFunc, varargin)
    % tau - единичный касательный вектор в пространстве (3x1)
    % вычисляем r(u,v), ru, rv, нормаль, затем проекцию
    [r, ru, rv, ~, ~, ~] = surfFunc(u, v, varargin{:});
    n = computeNormal(ru, rv);
    delta = dot(tau, n);
    if abs(delta) >= 1, delta = sign(delta)*0.9999; end  % защита
    % Проекция tau на касательную плоскость
    proj = (tau - delta*n) / sqrt(1 - delta^2);
    % Решаем систему: [E F; F G] * [du; dv] = [proj·ru, proj·rv]
    E = dot(ru, ru); F = dot(ru, rv); G = dot(rv, rv);
    rhs = [dot(proj, ru); dot(proj, rv)];
    M = [E, F; F, G];
    uv = M \ rhs;
    du = uv(1); dv = uv(2);
end