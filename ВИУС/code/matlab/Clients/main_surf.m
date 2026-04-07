%% Генерация данных (зашумленная полуокружность)
rng(10);
n_points = 20;
r = 5.0;
theta = linspace(0, pi, n_points)';

% Истинная окружность
clean_x = r * cos(theta);
clean_y = r * sin(theta);

% Зашумленные данные
noise_std = 0.001;
points_noisy = [clean_x + noise_std*randn(n_points, 1), ...
                clean_y + noise_std*randn(n_points, 1)];

%% Граничные условия
% Направления: [0, 1] вверх, [0, -1] вниз
% Кривизна: 1/r для окружности радиуса r
bc_start = struct('direction', [0.0, 1.0], 'curvature', 1.0/r);
bc_end = struct('direction', [0.0, -1.0], 'curvature', 1.0/r);

%% Создание и обучение сплайнов
spline = Splines.ParametricQuinticSpline(points_noisy, bc_start, bc_end);
% Основной сплайн (alpha=0.95 - сильное сглаживание)
spline.fit(0.95);

%% --- ДОБАВЛЕННЫЙ КОД (Модуль конвертации) ---

%% 1. Создание модуля конвертации поверхности
% Передаем обученный сплайн как образующую
surfaceMapper = Geometry.SurfaceRevolutionMapper(spline);

%% 2. Подготовка сетки параметров для 3D графика
% u - параметр вдоль образующей (берем весь диапазон сплайна)
u_eval = linspace(spline.u(1), spline.u(end), 50); 

% v - угол поворота вокруг оси Z (полный оборот)
v_eval = linspace(0, 2*pi, 60); 

% Создаем матрицы сетки
[U, V] = meshgrid(u_eval, v_eval);

%% 3. Вычисление координат поверхности
% Метод getPosition возвращает декартовы координаты X, Y, Z
[X_surf, Y_surf, Z_surf] = surfaceMapper.getPosition(U, V);

%% 4. Визуализация
figure('Name', 'Поверхность вращения', 'Color', 'w');

% Рисуем полупрозрачную поверхность
surf(X_surf, Y_surf, Z_surf, 'EdgeColor', 'none', 'FaceAlpha', 0.8, 'FaceColor', 'interp');
colormap(jet);
axis equal;
hold on;
grid on;

% Рисуем исходную "шумную" образующую (для сравнения, при v=0)
% Сначала получаем координаты образующей из сплайна при v=0
[X_gen, Y_gen, Z_gen] = surfaceMapper.getPosition(spline.u, 0);
plot3(X_gen, Y_gen, Z_gen, 'k.', 'MarkerSize', 8, 'DisplayName', 'Опорные точки (шум)');

% Рисуем сглаженную образующую (линия при v=0)
[U_line, V_line] = meshgrid(u_eval, 0);
[X_line, Y_line, Z_line] = surfaceMapper.getPosition(U_line, V_line);
plot3(X_line, Y_line, Z_line, 'r-', 'LineWidth', 3, 'DisplayName', 'Сглаженный сплайн');

% Оформление
title('Поверхность вращения (Сфера) по зашумленным точкам');
xlabel('X'); ylabel('Y'); zlabel('Z (Ось изделия)');
legend('show');
view(45, 30);