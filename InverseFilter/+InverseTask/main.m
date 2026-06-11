clear; clc; close all;

% Параметры эллипсоидов
a1 = 3.0; b1 = 2.5; c1 = 2.0;
scale = 0.8;
a2 = a1 * scale; b2 = b1 * scale; c2 = c1 * scale;

% 1. Строим геодезическую на E2 (эллипсоид с полуосями a2,b2,c2)
u0 = 0.0; v0 = 0.5;         % начальная точка (долгота, широта)
azimuth0 = pi/4;             % начальный азимут (45 градусов от меридиана)
s_max = 3.0;                 % длина геодезической
[u_geo, v_geo, s_geo] = geodesicOnEllipsoid(u0, v0, azimuth0, s_max, a2, b2, c2);
% Переводим в декартовы координаты на E2
geo_points = zeros(3, length(u_geo));
for i = 1:length(u_geo)
    geo_points(:,i) = ellipsoidSurface(u_geo(i), v_geo(i), a2, b2, c2);
end

% 2. Для каждой точки геодезической строим луч по касательной и находим пересечение с E1
R_points = zeros(3, length(u_geo));
for i = 1:length(u_geo)
    [r, ru, rv] = ellipsoidSurface(u_geo(i), v_geo(i), a2, b2, c2);
    % Касательный вектор к геодезической в пространстве:
    tau = ( ru * (u_geo(i+1)-u_geo(i)) + rv * (v_geo(i+1)-v_geo(i)) ) / norm(ru * (u_geo(i+1)-u_geo(i)) + rv * (v_geo(i+1)-v_geo(i)));
    % Находим пересечение луча r + t*tau с эллипсоидом E1
    % Уравнение эллипсоида: x^2/a1^2 + y^2/b1^2 + z^2/c1^2 = 1
    A = (tau(1)^2)/a1^2 + (tau(2)^2)/b1^2 + (tau(3)^2)/c1^2;
    B = 2*( r(1)*tau(1)/a1^2 + r(2)*tau(2)/b1^2 + r(3)*tau(3)/c1^2 );
    C = (r(1)^2)/a1^2 + (r(2)^2)/b1^2 + (r(3)^2)/c1^2 - 1;
    D = B^2 - 4*A*C;
    if D >= 0
        t1 = (-B + sqrt(D))/(2*A);
        t2 = (-B - sqrt(D))/(2*A);
        t = min([t1 t2]); % берем ближайшее положительное
        if t < 0, t = max([t1 t2]); end
        R_points(:,i) = r + t * tau;
    else
        R_points(:,i) = r; % запасной вариант
    end
end

% Параметризуем R_points натуральным параметром z (длина кривой)
z = cumsum([0, sqrt(sum(diff(R_points,1,2).^2,1))]);
R_func = @(zi) interp1(z, R_points', zi, 'pchip')';
% Производную R'(z) получим через сплайн
spl_x = spline(z, R_points(1,:)); spl_y = spline(z, R_points(2,:)); spl_z = spline(z, R_points(3,:));
Rprime_func = @(zi) [ppval(fnder(spl_x), zi); ppval(fnder(spl_y), zi); ppval(fnder(spl_z), zi)];

% 3. Обратная задача: восстанавливаем линию укладки на E2 по R(z)
% Начальные условия: берём первую точку геодезической как начальную точку линии укладки
u0_rec = u_geo(1); v0_rec = v_geo(1);
% Начальные du/ds, dv/ds вычисляем из начального касательного вектора
[du0_s, dv0_s] = computeTangentCoeffs(u0_rec, v0_rec, (R_points(:,1) - geo_points(:,1))/norm(R_points(:,1) - geo_points(:,1)), ...
    @ellipsoidSurface, a2, b2, c2);

DeltaZ = 0.5;       % длина участка для расчета k (метров)
Percentage = 50;    % уменьшение ошибки на 50% на длине DeltaZ
z_span = [0, z(end)];
[u_rec, v_rec, s_rec] = recoverLayer(R_func, Rprime_func, z_span, ...
    u0_rec, v0_rec, du0_s, dv0_s, @ellipsoidSurface, DeltaZ, Percentage, a2, b2, c2);

% Преобразуем восстановленную линию в декартовы координаты
recovered_points = zeros(3, length(u_rec));
for i = 1:length(u_rec)
    recovered_points(:,i) = ellipsoidSurface(u_rec(i), v_rec(i), a2, b2, c2);
end

% 4. Визуализация
figure;
% Эллипсоид E1
[X1, Y1, Z1] = ellipsoid(0,0,0, a1, b1, c1, 40);
surf(X1, Y1, Z1, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'r'); hold on;
% Эллипсоид E2
[X2, Y2, Z2] = ellipsoid(0,0,0, a2, b2, c2, 40);
surf(X2, Y2, Z2, 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'FaceColor', 'g');

% Исходная геодезическая на E2
plot3(geo_points(1,:), geo_points(2,:), geo_points(3,:), 'b-', 'LineWidth', 2);
% Восстановленная линия укладки
plot3(recovered_points(1,:), recovered_points(2,:), recovered_points(3,:), 'm--', 'LineWidth', 2);
% Точки R(z) на E1
plot3(R_points(1,:), R_points(2,:), R_points(3,:), 'ro', 'MarkerSize', 4);

legend('E1', 'E2', 'Исходная геодезическая (E2)', 'Восстановленная линия (E2)', 'R(z) на E1');
axis equal; grid on; view(3);
title('Обратная задача намотки: восстановление линии укладки');