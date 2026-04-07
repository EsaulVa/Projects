%% CLIENT SCRIPT: Clean Architecture Demo
clear; close all; clc;

%% 1. Подготовка данных
filename = 'data.txt';
if ~exist(filename, 'file')
    % Создадим тестовый файл, если его нет
    dlmwrite(filename, [2.646 11.007 2.98 -0.428; 9.797 26.197 1.609 -0.094; ...
        20.397 39.113 0.87 -0.051; 33.257 47.007 0.405 -0.027; 49.5 50.341 0 0], 'delimiter', ' ');
end
M = readmatrix(filename);
x_glob = M(:, 1);
y_glob = M(:, 2);

%% 2. Поиск центра (Circle Fit)
% Используем линейный МНК для уравнения окружности
A = [2*x_glob, 2*y_glob, ones(length(x_glob), 1)];
b = x_glob.^2 + y_glob.^2;
p = A \ b;
x_center = p(1);
y_center = p(2);
R_est = sqrt(p(3) + x_center^2 + y_center^2);
fprintf('Найден центр: (%.2f, %.2f), Радиус: %.2f\n', x_center, y_center, R_est);

%% 3. Переход к локальным координатам
x_loc = x_glob - x_center;
y_loc = y_glob - y_center;

%% 4. Формирование ГУ (Геометрическая гипотеза)
% Предполагаем, что кривая - дуга окружности:
% r' = 0, r'' = 0, phi' = 1/R, phi'' = 0

% Структура ГУ: NaN означает "свободная переменная"
% Вариант А: Жесткая заделка по радиусу, свобода по углу (пример)
bc_start.r = struct('value', sqrt(x_loc(1)^2+y_loc(1)^2), 'deriv1', 0, 'deriv2', 0);
% bc_start.phi = struct('value', atan2(y_loc(1), x_loc(1)), 'deriv1', 1/R_est, 'deriv2', 0);
bc_start.phi = struct('value', atan2(y_loc(1), x_loc(1)), 'deriv1', NaN, 'deriv2', 0);

bc_end.r   = struct('value', sqrt(x_loc(end)^2+y_loc(end)^2), 'deriv1', 0, 'deriv2', 0);
% bc_end.phi = struct('value', atan2(y_loc(end), x_loc(end)), 'deriv1', 1/R_est, 'deriv2', 0);
bc_end.phi = struct('value', atan2(y_loc(end), x_loc(end)), 'deriv1', 0, 'deriv2', 0);

%% 5. Создание и оптимизация сплайна
spline = Splines.PolarSmoothingSplineNew(x_loc, y_loc);
spline.setBC(bc_start, bc_end); % Устанавливаем ГУ и маски

alpha = 0.95; % Вес точности
spline.fit(alpha);

%% 6. Визуализация результатов
u_dense = linspace(spline.u(1), spline.u(end), 200)';
n = spline.n;

% Подготовка данных для графики
r_res = spline.results.r;
phi_res = spline.results.phi;

% Вычисление точек (можно вынести в метод predict, но здесь сделаем явно)
r_vals = zeros(size(u_dense)); phi_vals = zeros(size(u_dense));
r1_vals = zeros(size(u_dense)); phi1_vals = zeros(size(u_dense));

geom = spline.geom_r; % Экземпляр движка

for k = 1:length(u_dense)
    u = u_dense(k);
    idx = find(spline.u <= u, 1, 'last');
    if idx > n-1, idx = n-1; end
    t = u - spline.u(idx);
    
    % R
    cr = geom.getSegmentCoeffs(r_res.v(idx), r_res.m(idx), r_res.M(idx), ...
                               r_res.v(idx+1), r_res.m(idx+1), r_res.M(idx+1), spline.u(idx+1)-spline.u(idx));
    r_vals(k) = geom.evalValue(t, r_res.v(idx), r_res.m(idx), r_res.M(idx), cr);
    r1_vals(k) = geom.evalDeriv1(t, r_res.m(idx), r_res.M(idx), cr);
    
    % Phi
    cp = geom.getSegmentCoeffs(phi_res.v(idx), phi_res.m(idx), phi_res.M(idx), ...
                               phi_res.v(idx+1), phi_res.m(idx+1), phi_res.M(idx+1), spline.u(idx+1)-spline.u(idx));
    phi_vals(k) = geom.evalValue(t, phi_res.v(idx), phi_res.m(idx), phi_res.M(idx), cp);
    phi1_vals(k) = geom.evalDeriv1(t, phi_res.m(idx), phi_res.M(idx), cp);
end

% Обратный перевод в глобальные
x_fit = r_vals .* cos(phi_vals) + x_center;
y_fit = r_vals .* sin(phi_vals) + y_center;

% Графики
figure('Position', [100, 100, 1000, 600]);

subplot(1, 2, 1);
plot(x_glob, y_glob, 'ro', 'DisplayName', 'Данные');
hold on;
plot(x_fit, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн');
plot(x_center, y_center, 'kx', 'MarkerSize', 10, 'DisplayName', 'Центр');
axis equal; grid on; legend;
title('Декартовы координаты');

subplot(1, 2, 2);
plot(phi_vals, r_vals, 'b-', 'LineWidth', 2);
hold on;
plot(phi_vals, R_est*ones(size(phi_vals)), 'k--', 'DisplayName', 'R опт.');
yline(R_est, 'k--');
grid on;
xlabel('\phi'); ylabel('r');
title('Полярные координаты (относительно центра)');

fprintf('Готово.\n');