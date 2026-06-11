% ========== 1. Параметры и поверхности ==========
a1=3; b1=2.5; c1=2; E1 = Ellipsoid(a1,b1,c1);
scale=0.8; a2=a1*scale; b2=b1*scale; c2=c1*scale; E2 = Ellipsoid(a2,b2,c2);

% ========== 2. Геодезическая на E2 (линия укладки) ==========
u0=0; v0=0.5; azimuth0=pi/4; s_end=14.0;
[u_geo, v_geo, ~] = geodesicOnEllipsoid(E2, u0, v0, azimuth0, s_end, 'ode45');
% Переводим в декартовы
N = length(u_geo);
geo_pts = zeros(3,N);
for i=1:N, geo_pts(:,i)=E2.getPoint(u_geo(i), v_geo(i)); end
layerTraj = Trajectory(geo_pts);

% ========== 3. Построение R(z) (траектория точки схода) ==========
step_s = 0.05;
tracer = RayTracer(E2, E1, layerTraj);
[R_traj, R_points, z_vals] = tracer.trace(step_s);

% ========== 4. Обратная задача: восстановление линии укладки ==========
% Начальные условия (берём из геодезической)
u0_rec = u_geo(1); v0_rec = v_geo(1);
% Начальные du/ds, dv/ds: используем первую точку и её касательный вектор
tau0 = layerTraj.getTangent(0);
[du0_s, dv0_s] = computeTangentCoeffs(E2, u0_rec, v0_rec, tau0);

% Параметры сходимости
DeltaZ = 0.5; Percentage = 50;
z_start = 0; z_end = R_traj.totalLength();
z_span = [z_start, z_end];

% % Интегрирование
% [u_rec, v_rec, s_rec, z_rec] = recoverLayer(E2, ...
%     @(z) R_traj.getPoint(z), @(z) R_traj.getTangent(z), ...
%     z_span, u0_rec, v0_rec, du0_s, dv0_s, DeltaZ, Percentage, 'ode45');
% [u_rec, v_rec, s_rec, z_rec] = recoverLayer(E2, ...
%     @(z) R_traj.getPoint(z), @(z) R_traj.getTangent(z), ...
%     [0, R_traj.totalLength()], u0_rec, v0_rec, du0_s, dv0_s, ...
%     0.5, 50, 'ode45');
% Предполагаем, что R_traj — объект ChordalTrajectory
[u_rec, v_rec, s_rec, z_rec] = recoverLayer(E2, ...
    @(z) R_traj.getPoint(z), @(z) R_traj.getTangent(z), ...
    [0, R_traj.totalLength()], u0_rec, v0_rec, du0_s, dv0_s, ...
    0.5, 50);
% Переводим восстановленную линию в декартовы
rec_pts = zeros(3, length(u_rec));
for i=1:length(u_rec)
    rec_pts(:,i) = E2.getPoint(u_rec(i), v_rec(i));
end

% ========== 5. Визуализация сравнения ==========
figure('Name','Обратная задача','Color','w'); hold on; axis equal; grid on; view(3);
% Поверхности
[X1,Y1,Z1]=ellipsoid(0,0,0,a1,b1,c1,40); surf(X1,Y1,Z1,'FaceAlpha',0.1,'EdgeColor','none','FaceColor','r');
[X2,Y2,Z2]=ellipsoid(0,0,0,a2,b2,c2,40); surf(X2,Y2,Z2,'FaceAlpha',0.2,'EdgeColor','none','FaceColor','g');
% Исходная линия укладки (синяя)
plot3(geo_pts(1,:), geo_pts(2,:), geo_pts(3,:), 'b-', 'LineWidth', 1.5);
% Восстановленная линия (пурпурная пунктир)
plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'm--', 'LineWidth', 1.5);
% Траектория R(z) (красные точки)
plot3(R_points(1,:), R_points(2,:), R_points(3,:), 'ro', 'MarkerSize', 3);
xlabel('x'); ylabel('y'); zlabel('z');
legend('E1','E2','Исходная линия','Восстановленная линия','R(z)');
title(sprintf('Сравнение исходной и восстановленной линий укладки (k = %.3f)', ...
    -log(1-Percentage/100)/DeltaZ));

% ========== 6. Вычисление и построение ошибки ==========
% Интерполируем исходную геодезическую на узлы s_rec (восстановленной длины)
% Предварительно убедимся, что s_geo и s_rec имеют перекрывающиеся диапазоны
% После получения u_geo, v_geo вычислите s_geo как кумулятивную сумму расстояний между точками
geo_pts = zeros(3, length(u_geo));
for i=1:length(u_geo)
    geo_pts(:,i) = E2.getPoint(u_geo(i), v_geo(i));
end
dist = sqrt(sum(diff(geo_pts,1,2).^2, 1));
s_geo = [0, cumsum(dist)];
if max(s_rec) > max(s_geo)
    warning('Восстановленная длина дуги больше исходной, обрезаем');
    s_rec = s_rec(s_rec <= max(s_geo));
    u_rec = u_rec(1:length(s_rec));
    v_rec = v_rec(1:length(s_rec));
end

u_geo_interp = interp1(s_geo, u_geo, s_rec, 'pchip');
v_geo_interp = interp1(s_geo, v_geo, s_rec, 'pchip');

% Ошибка в параметрических координатах (u,v)
err_u = u_rec - u_geo_interp;
err_v = v_rec - v_geo_interp;

% Декартова ошибка (евклидово расстояние между точками на поверхности)
err_xyz = zeros(size(s_rec));
for i = 1:length(s_rec)
    p_rec = E2.getPoint(u_rec(i), v_rec(i));
    p_orig = E2.getPoint(u_geo_interp(i), v_geo_interp(i));
    err_xyz(i) = norm(p_rec - p_orig);
end

% График ошибки
figure('Name','Ошибка восстановления','Color','w');
subplot(2,1,1);
semilogy(s_rec, err_u, 'b-', 'LineWidth', 1.5); grid on;
ylabel('\Delta u'); title('Ошибка в координате u');

subplot(2,1,2);
semilogy(s_rec, err_v, 'r-', 'LineWidth', 1.5); grid on;
ylabel('\Delta v'); xlabel('Восстановленная длина дуги s');
sgtitle('Ошибка в параметрических координатах');

figure('Name','Евклидова ошибка','Color','w');
semilogy(s_rec, err_xyz, 'k-', 'LineWidth', 2); grid on;
xlabel('Восстановленная длина дуги s'); ylabel('||r_{rec} - r_{orig}||');
title('Евклидово расстояние между исходной и восстановленной линиями');