% Создаём эллипсоид
E2 = Ellipsoid(2.4, 2.0, 1.6);   % уменьшенный на 0.8 от (3,2.5,2)

% Параметры геодезической
u0 = 0; v0 = 0.5;          % начальная точка
azimuth0 = pi/4;           % 45 градусов
s_max = 4.0;               % длина

% Вычисляем геодезическую
[u_geo, v_geo, s_geo] = geodesicOnEllipsoid(E2, u0, v0, azimuth0, s_max, 'ode45');

% Переводим в декартовы координаты
geo_pts = zeros(3, length(u_geo));
for i = 1:length(u_geo)
    geo_pts(:,i) = E2.getPoint(u_geo(i), v_geo(i));
end

% Визуализация
plot3(geo_pts(1,:), geo_pts(2,:), geo_pts(3,:), 'b-', 'LineWidth', 1.5);
axis equal; grid on;
xlabel('x'); ylabel('y'); zlabel('z');
title('Геодезическая на эллипсоиде');