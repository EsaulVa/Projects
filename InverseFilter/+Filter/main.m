% Чтение CSV
data = readtable('meridian_E2_raw_rz.csv');
z = data.z;
r = data.r;

% Создание поверхности
E2 = DiscreteRevolutionSurface(z, r);

% Точка на поверхности
pt = E2.position(400.0, 0.0);        % [x, y, z]

% Геометрия для DAE (если потом понадобится)
d = E2.derivatives(400.0, 0.0);    % struct с ru, rv, normal
[E, F, G] = E2.first_fundamental_form(400.0, 0.0);
[L, M, N] = E2.second_fundamental_form(400.0, 0.0);

% Радиус
r400 = E2.radius(400.0);
point = [100.0, 0.0, 400.0];   % [x, y, z]
[u, v] = E2.uv_from_point(point);   % u = 400.0, v = 0.0