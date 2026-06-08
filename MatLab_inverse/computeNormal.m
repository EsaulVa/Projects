function n = computeNormal(ru, rv)
    n = cross(ru, rv);
    n = n / norm(n);
end

