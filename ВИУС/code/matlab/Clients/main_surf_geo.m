%% Генерация данных (зашумленная полуокружность)
rng(10);
n_points = 10;
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

% 2. Создаем решатель траекторий
solver = Solver.GeodesicSolver(surfaceMapper);

% 3. Задаем параметры задачи
% Начало: у полюса (u_min), угол 0
u_start = spline.u(1); 
v_start = 0;
alpha_start = pi/2; % Наматываем под 45 градусов

% Цель: противоположный полюс (u_max), угол pi (пол-оборота)
u_end = spline.u(end);
v_end = pi;

% 4. Запускаем расчет
path = solver.solve(u_start, v_start, alpha_start, u_end, v_end, 0.05);

% 5. Визуализация результата
figure('Name', 'Траектория намотки', 'Color', 'w');
% 2. Подготовка сетки параметров для 3D графика
% u - параметр вдоль образующей (берем весь диапазон сплайна)
u_eval = linspace(spline.u(1), spline.u(end), 30); 

% v - угол поворота вокруг оси Z (полный оборот)
v_eval = linspace(0, 2*pi, 20); 

% Создаем матрицы сетки
[U, V] = meshgrid(u_eval, v_eval);

%% 3. Вычисление координат поверхности
% Метод getPosition возвращает декартовы координаты X, Y, Z
[X_surf, Y_surf, Z_surf] = surfaceMapper.getPosition(U, V);
% Рисуем полупрозрачную сферу
surf(X_surf, Y_surf, Z_surf, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'FaceColor', 'blue');
hold on;
axis equal; grid on;
% Рисуем исходную "шумную" образующую (для сравнения, при v=0)
% Сначала получаем координаты образующей из сплайна при v=0
[X_gen, Y_gen, Z_gen] = surfaceMapper.getPosition(spline.u, 0);
plot3(X_gen, Y_gen, Z_gen, 'k.', 'MarkerSize', 8, 'DisplayName', 'Опорные точки (шум)');
% Рисуем сглаженную образующую (линия при v=0)
[U_line, V_line] = meshgrid(u_eval, 0);
[X_line, Y_line, Z_line] = surfaceMapper.getPosition(U_line, V_line);
plot3(X_line, Y_line, Z_line, 'cyan', 'LineWidth', 3, 'DisplayName', 'Сглаженный сплайн');
% Рисуем траекторию (красная линия)
plot3(path.X, path.Y, path.Z, 'r-', 'LineWidth', 2);

% Маркеры старта и финиша
plot3(path.X(1), path.Y(1), path.Z(1), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Старт');
plot3(path.X(end), path.Y(end), path.Z(end), 'yo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Финиш');

title('Геодезическая траектория на сфере');
legend('show');
view(45, 30);