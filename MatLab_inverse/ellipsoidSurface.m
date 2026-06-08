function [r, ru, rv, ruu, ruv, rvv] = ellipsoidSurface(u, v, a, b, c)
    % Эллипсоид: x = a*cos(u)*cos(v), y = b*sin(u)*cos(v), z = c*sin(v)
    % u - долгота (0..2pi), v - широта (-pi/2..pi/2)
    cosu = cos(u); sinu = sin(u);
    cosv = cos(v); sinv = sin(v);
    
    r = [a*cosu*cosv; b*sinu*cosv; c*sinv];
    
    % Первые производные
    ru = [-a*sinu*cosv; b*cosu*cosv; 0];
    rv = [-a*cosu*sinv; -b*sinu*sinv; c*cosv];
    
    % Вторые производные
    ruu = [-a*cosu*cosv; -b*sinu*cosv; 0];
    ruv = [a*sinu*sinv; -b*cosu*sinv; 0];
    rvv = [-a*cosu*cosv; -b*sinu*cosv; -c*sinv];
end

function [E, F, G] = firstFundamentalForm(ru, rv)
    E = dot(ru, ru);
    F = dot(ru, rv);
    G = dot(rv, rv);
end

function [L, M, N] = secondFundamentalForm(ru, rv, ruu, ruv, rvv, n)
    L = dot(ruu, n);
    M = dot(ruv, n);
    N = dot(rvv, n);
end

function n = computeNormal(ru, rv)
    n = cross(ru, rv);
    n = n / norm(n);
end