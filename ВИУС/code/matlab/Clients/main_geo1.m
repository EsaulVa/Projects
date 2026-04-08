%% Клиентский скрипт для моделирования намотки баллона давления
% Описание: Генерирует профиль "Сфера - Цилиндр - Сфера", строит сглаживающий сплайн
% и рассчитывает геодезическую траекторию.

clear; clc; close all;

%% 1. Параметры геометрии баллона
R_sphere = 3.0;              % Радиус сферических полюсов (м)
L_cyl = 2.0;                 % Длина цилиндрической части (м)
delta_theta = deg2rad(5);   % Угол среза полюсов (насколько "отрезана" вершина)

% Плотность точек для генерации исходных данных
points_per_segment = 7;

%% 2. Генерация точек профиля (Сборка "сырых" данных)
% Мы формируем точки в формате [Z, R], где Z - высота, R - радиус

% --- Левая сфера (срезанная) ---
% Центр левой сферы в Z=0. Полюс был бы в -R.
% Параметр theta меняется от -pi/2 + delta (срез) до 0 (экватор)
theta_left = linspace(-pi/2 + delta_theta, 0, points_per_segment)';
z_left = R_sphere * sin(theta_left);
r_left = R_sphere * cos(theta_left);

% --- Цилиндр ---
% От экватора (Z=0) до Z=L_cyl
z_cyl = linspace(0, L_cyl, points_per_segment)';
r_cyl = ones(size(z_cyl)) * R_sphere;

% --- Правая сфера (срезанная) ---
% Центр правой сферы в Z=L_cyl.
% Параметр theta меняется от 0 (экватор) до pi/2 - delta (срез)
theta_right = linspace(0, pi/2 - delta_theta, points_per_segment)';
z_right = L_cyl + R_sphere * sin(theta_right);
r_right = R_sphere * cos(theta_right);

% Объединяем все точки в один массив
% points_clean = [z_left, r_left; 
%                 z_cyl, r_cyl; 
%                 z_right, r_right];
% Объединение с удалением дубликатов на стыках
points_clean = [z_left(1:end-1), r_left(1:end-1);      % Берем все, кроме последней
                z_cyl(1:end-1), r_cyl(1:end-1);        % Берем все, кроме последней
                z_right, r_right];                     % Берем все, т.к. правый конец финальный
% Добавляем шум (имитация ошибок измерения), чтобы продемонстрировать сглаживание
rng(42); % Фиксируем random seed для повторяемости
noise_level = 0; 
points_noisy = points_clean + noise_level * randn(size(points_clean));

% Защита: радиус не должен быть отрицательным
points_noisy(:, 2) = max(points_noisy(:, 2), 0.01);

%% 3. Построение математической модели (Сглаживающий сплайн)
% Граничные условия:
% На полюсах касательная вертикальна (вдоль оси Z).
% Вектор направления [dz, dr] = [1, 0].
% Кривизна полюсов равна 1/R_sphere.

% bc_start = struct('direction', [1.0, 0.0], 'curvature', 1.0/R_sphere);
% --- Расчет граничных условий для левого полюса ---
% Направление: [sin(delta), cos(delta)]
% Кривизна: 1 / R
bc_start = struct( ...
    'direction', [sin(delta_theta), cos(delta_theta)], ...
    'curvature', 1.0 / R_sphere ...
);
% bc_end   = struct('direction', [1.0, 0.0], 'curvature', 1.0/R_sphere);
% --- Для правого полюса (симметрия) ---
% Для правой сферы все зеркально, но если вы считаете от экватора к полюсу,
% логика меняется. Если же у вас точки уже сгенерированы от 0 до pi/2-delta,
% то направление будет:
% theta_end = pi/2 - delta_theta
% direction = [cos(theta_end), -sin(theta_end)] -> [sin(delta), -cos(delta)]
% Кривизна та же.

bc_end = struct( ...
    'direction', [sin(delta_theta), -cos(delta_theta)], ...
    'curvature', 1.0 / R_sphere ...
);
fprintf('Создание сплайна из %d точек...\n', size(points_noisy, 1));
spline = Splines.ParametricQuinticSpline(points_noisy, bc_start, bc_end);

% Обучаем сплайн (alpha=0.9 означает сильное сглаживание шума)
spline.fit(0.9);
fprintf('Сглаживание завершено.\n');

%% 4. Создание поверхности и расчет траектории
% Инициализируем маппер поверхности
mapper = Geometry.SurfaceRevolutionMapper(spline);

% Инициализируем решатель геодезических
solver = Solver.GeodesicSolver(mapper);

% Параметры задачи траектории
u_start = spline.u(1);           % Начало левого полюса
u_end   = spline.u(end);         % Конец правого полюса
alpha_0 = deg2rad(45);           % Угол намотки 45 градусов

% Целевой угол: делаем 2.5 оборота, чтобы получилась красивая спираль
v_target = 2*pi;         
% disp u_start  u_end alpha_0 v_target
fprintf('Расчет геодезической траектории...\n');
% Запускаем решатель. Допуск 1e-2 гарантирует точное попадание в конец баллона
% path = solver.solve(u_start, 0, alpha_0, u_end, v_target, 0.01);
% Передаем u_end только как границу безопасности или для справки
% Но в eventsFunction лучше явно прописать u_end
path = solver.solve(u_start, 0, alpha_0, u_end, v_target, 1e-2);
fprintf('Траектория рассчитана. Длина пути: %.2f м\n', path.s(end));

%% 5. Визуализация результатов
figure('Name', 'Моделирование намотки баллона', 'Color', 'w', 'Position', [100, 100, 1200, 600]);

% --- График 1: Профиль (Образующая) ---
subplot(1, 2, 1);
plot(points_noisy(:,1), points_noisy(:,2), 'k.', 'MarkerSize', 4, 'DisplayName', 'Шумные точки');
hold on;
% Рисуем сглаженный сплайн
[U_line, V_line] = meshgrid(spline.u, 0);
[X_line, Y_line, Z_line] = mapper.getPosition(U_line, V_line);
plot(X_line, Z_line, 'r-', 'LineWidth', 2, 'DisplayName', 'Сглаженный сплайн');

title('Образующая баллона');
xlabel('Z (Высота), м'); ylabel('R (Радиус), м');
legend('show'); grid on; axis equal;
xlim([min(points_noisy(:,1))-0.2, max(points_noisy(:,1))+0.2]);

% --- График 2: 3D Траектория ---
subplot(1, 2, 2);
% Рисуем сетку поверхности (баллон)
u_plot = linspace(spline.u(1), spline.u(end), 50);
v_plot = linspace(0, 2*pi, 60);
[U, V] = meshgrid(u_plot, v_plot);
[X, Y, Z_cart] = mapper.getPosition(U, V);

surf(X, Y, Z_cart, 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'FaceColor', [0.85, 0.9, 1.0]);
hold on;

% Рисуем траекторию нити
plot3(path.X, path.Y, path.Z, 'r-', 'LineWidth', 2, 'DisplayName', 'Нить');

% Маркеры старта и финиша
plot3(path.X(1), path.Y(1), path.Z(1), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Старт');
plot3(path.X(end), path.Y(end), path.Z(end), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Финиш');

title('3D Модель намотки');
xlabel('X, м'); ylabel('Y, м'); zlabel('Z, м');
legend('show'); axis equal; view(45, 20);
light; lighting gouraud; % Добавляем свет для объема