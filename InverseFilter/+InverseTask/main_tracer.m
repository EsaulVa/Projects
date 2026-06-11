% ====================== ПАРАМЕТРЫ ======================
% Эллипсоид E1 (внешний, безопасность)
a1 = 3.0; b1 = 2.5; c1 = 2.0;
E1 = Ellipsoid(a1, b1, c1);

% Эллипсоид E2 (внутренний, оправка) - масштабирован
scale = 0.8;
a2 = a1 * scale; b2 = b1 * scale; c2 = c1 * scale;
E2 = Ellipsoid(a2, b2, c2);

% Параметры геодезической на E2
u0 = 0; v0 = 0.5;          % начальная точка
azimuth0 = pi/4;           % начальный азимут
s_end = 19.0;               % длина геодезической

% Построение геодезической (линии укладки)
[u_geo, v_geo, s_geo] = geodesicOnEllipsoid(E2, u0, v0, azimuth0, s_end, 'ode45');
% Переводим в декартовы координаты
N_geo = length(u_geo);
geo_points = zeros(3, N_geo);
for i = 1:N_geo
    geo_points(:,i) = E2.getPoint(u_geo(i), v_geo(i));
end

% Создаём траекторию линии укладки с натуральной параметризацией
layerTraj = Trajectory(geo_points);

% ====================== ТРАССИРОВКА ======================
step_s = 0.05;   % шаг по длине дуги s (можно менять)
tracer = RayTracer(E2, E1, layerTraj);
[R_traj, R_points, z_vals] = tracer.trace(step_s);

% ====================== ВИЗУАЛИЗАЦИЯ ======================
figure('Name', 'Трассировка лучей', 'NumberTitle', 'off');
hold on; grid on; axis equal;

% Отображаем эллипсоиды
[X1, Y1, Z1] = ellipsoid(0,0,0, a1, b1, c1, 40);
surf(X1, Y1, Z1, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'FaceColor', 'r');
[X2, Y2, Z2] = ellipsoid(0,0,0, a2, b2, c2, 40);
surf(X2, Y2, Z2, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'g');

% Линия укладки на E2
plot3(geo_points(1,:), geo_points(2,:), geo_points(3,:), 'b-', 'LineWidth', 2);

% Траектория R(z) на E1
R_curve = zeros(3, 200);
s_R = linspace(0, R_traj.totalLength(), 200);
for i = 1:200
    R_curve(:,i) = R_traj.getPoint(s_R(i));
end
plot3(R_curve(1,:), R_curve(2,:), R_curve(3,:), 'm-', 'LineWidth', 2);

% Точки R(z)
plot3(R_points(1,:), R_points(2,:), R_points(3,:), 'mo', 'MarkerSize', 4, 'MarkerFaceColor', 'm');

% Рисуем лучи (каждый 10-й для наглядности)
s_vals = layerTraj.getSValues();   % массив s-значений в узлах (длина N_geo)
skip = max(1, floor(N_geo / 50));
for i = 1:skip:N_geo
    P = geo_points(:,i);
    s = s_vals(i);
    tau = layerTraj.getTangent(s);
    Q = R_points(:, min(i, size(R_points,2)));
    plot3([P(1), Q(1)], [P(2), Q(2)], [P(3), Q(3)], 'k-', 'LineWidth', 0.5);
end

% Оформление
xlabel('x'); ylabel('y'); zlabel('z');
legend('E1 (безопасность)', 'E2 (оправка)', 'Линия укладки', 'R(z)', 'Точки R', 'Лучи');
title('Трассировка лучей от геодезической к внешнему эллипсоиду');
view(3);
hold off;

% ====================== СОХРАНЕНИЕ R(z) В ФАЙЛ ======================
% Сохраняем точки R(z) с соответствующими значениями z (длина дуги)
z_full = linspace(0, R_traj.totalLength(), size(R_points,2));
output = [z_full(:), R_points(1,:)', R_points(2,:)', R_points(3,:)'];
save('R_points.txt', 'output', '-ascii');
fprintf('Точки R(z) сохранены в R_points.txt\n');
fprintf('Полная длина линии укладки L = %.4f\n', layerTraj.totalLength());
fprintf('Полная длина траектории R(z) = %.4f\n', R_traj.totalLength());