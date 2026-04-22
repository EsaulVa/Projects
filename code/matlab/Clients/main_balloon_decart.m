%% Клиент для Декартова сплайна: y(x)
% Демонстрация ограничений и возможностей одномерного сглаживания
% для образующей баллона.

clear all; close all; clc;


%% 1. Параметры баллона и генерация данных
R = 3.0;            % Радиус
L = 8.0;            % Длина цилиндра
noise_std = 0*1e-3;   % Уровень шума (имитация погрешностей измерения/модели)

% Генерируем точки для всех участков сразу (единый массив)
n_sph = 5;         % Точек на полусферу
n_cyl = 3;         % Точек на цилиндр

% Левая полусфера
theta1 = linspace(0, pi/2, n_sph)';
pts1 = [-L/2 - R*cos(theta1), R*sin(theta1)];

% Цилиндр
x_cyl = linspace(-L/2, L/2, n_cyl)';
pts2 = [x_cyl, R*ones(n_cyl, 1)];

% Правая полусфера
theta3 = linspace(pi/2, 0, n_sph)';
pts3 = [L/2 + R*cos(theta3), R*sin(theta3)];

% Объединение в единую последовательность
% Исключаем дубликаты стыковочных точек (pts1(end) ~= pts2(1) с точностью до шума)
points_true = [pts1; pts2(2:end-1,:); pts3];

% Добавляем шум для реалистичности
rng(42); % Фиксируем seed для воспроизводимости
points_noisy = points_true + noise_std * randn(size(points_true));
x_data=points_noisy(:,1);
y_noisy=points_noisy(:,2);

%% 2. Граничные условия
% В декартовом случае граничные условия задаются на y' и y''.

% Левая граница (стык сферической части с "отрезанной" крышкой)
% Тут касательная идет под углом ~85 градусов -> y' велик.
% Но для корректности зададим условие на ЭКВАТОРЕ (начало данных)?
% Нет, сплайн строится по всем точкам. Границы - это края массива x_data.
% Края у нас - это сечение сфер.
% Чтобы избежать проблем с большими производными, зададим "естественные" условия
% или оценим производную аналитически.
% Для угла max_angle: y' = -x / sqrt(R^2 - x^2)
% slope_end = -(x_data(1) + L/2) / sqrt(R^2 - (x_data(1) + L/2).^2);
% curv_end = -R^2 / (R^2 - (x_data(1) + L/2).^2).^(3/2);
slope_end = 1;
curv_end = 1/R;
bc_left = struct('m', slope_end, 'M', curv_end); 
% Примечание: задание точных краевых условий на шуме может "дергать" сплайн.
% Можно поставить 0, если мы считаем, что сплайн должен выпрямиться, 
% но это исказит форму. Здесь оставим аналитику.

% Правая граница (аналогично)
% slope_end_r = -(x_data(end) - L/2) / sqrt(R^2 - (x_data(end) - L/2).^2);
% curv_end_r = -R^2 / (R^2 - (x_data(end) - L/2).^2).^(3/2);
slope_end_r = 0;
curv_end_r = 1/R;
bc_right = struct('m', slope_end_r, 'M', curv_end_r);

%% 3. Создание и обучение сплайна
fprintf('=== Обучение декартова сплайна y(x) ===\n');
fprintf('Внимание: Края (полюса) отсечены из-за сингулярности y''(x) -> inf.\n');

% Используем класс SmoothingQuinticSpline
spline_cart = Splines.SmoothingQuinticSpline(x_data, y_noisy, bc_left, bc_right);

alpha = 1; % Параметр сглаживания
spline_cart.fit(alpha);

%% 4. Расчет значений для графиков
x_dense = linspace(x_data(1), x_data(end), 1000)';
y_spline = spline_cart.predict(x_dense, 0);
dy_spline = spline_cart.predict(x_dense, 1);
d2y_spline = spline_cart.predict(x_dense, 2);

% Истинная геометрия для сравнения
y_true = zeros(size(x_dense));
for i = 1:length(x_dense)
    x = x_dense(i);
    if x < -L/2
        y_true(i) = sqrt(R^2 - (x + L/2)^2);
    elseif x > L/2
        y_true(i) = sqrt(R^2 - (x - L/2)^2);
    else
        y_true(i) = R;
    end
end

% Истинная кривизна профиля k(x)
% k = |y''| / (1 + y'^2)^(3/2)
k_true = zeros(size(x_dense));
for i = 1:length(x_dense)
    x = x_dense(i);
    if x < -L/2 || x > L/2
        % Для сферы k = 1/R
        k_true(i) = 1/R;
    else
        % Для цилиндра k = 0
        k_true(i) = 0;
    end
end

% Кривизна сплайна (вычисленная через y' и y'')
k_spline = abs(d2y_spline) ./ (1 + dy_spline.^2).^(1.5);

%% 5. Визуализация
set(groot, 'DefaultLineLineWidth', 1.5);
figure('Name', 'Декартов сплайн y(x)', 'Position', [150, 150, 1400, 800]);

% --- График 1: Геометрия ---
subplot(2, 2, 1);
hold on;
plot(x_data, y_noisy, 'ro', 'MarkerSize', 4, 'DisplayName', 'Зашумленные данные');
plot(x_dense, y_spline, 'b-', 'DisplayName', 'Сглаживающий сплайн y(x)');
plot(x_dense, y_true, 'k--', 'DisplayName', 'Истинная форма');

title('Декартов сплайн (без полюсов)');
xlabel('x'); ylabel('y');
axis equal; grid on; legend('Location', 'best');
ylim([R-2, R+1]);

% --- График 2: Первая производная y'(x) ---
subplot(2, 2, 2);
hold on;
plot(x_dense, dy_spline, 'b-', 'DisplayName', 'y''(x) сплайн');
% Теоретическая y'(x)
dy_true = zeros(size(x_dense));
idx_sph_l = x_dense < -L/2;
idx_sph_r = x_dense > L/2;
dy_true(idx_sph_l) = -(x_dense(idx_sph_l) + L/2) ./ sqrt(R^2 - (x_dense(idx_sph_l) + L/2).^2);
dy_true(idx_sph_r) = -(x_dense(idx_sph_r) - L/2) ./ sqrt(R^2 - (x_dense(idx_sph_r) - L/2).^2);
plot(x_dense, dy_true, 'k--', 'DisplayName', 'Теория');

title('Первая производная y''(x)');
xlabel('x'); ylabel('dy/dx');
grid on; legend('Location', 'best');

% --- График 3: Вторая производная y''(x) ---
subplot(2, 2, 3);
hold on;
plot(x_dense, d2y_spline, 'b-', 'DisplayName', 'y''''(x) сплайн');
% Теоретическая y''(x)
d2y_true = zeros(size(x_dense));
d2y_true(idx_sph_l) = -R^2 ./ (R^2 - (x_dense(idx_sph_l) + L/2).^2).^(1.5);
d2y_true(idx_sph_r) = -R^2 ./ (R^2 - (x_dense(idx_sph_r) - L/2).^2).^(1.5);
plot(x_dense, d2y_true, 'k--', 'DisplayName', 'Теория (разрыв)');

title('Вторая производная y''''(x)');
xlabel('x'); ylabel('d^2y/dx^2');
grid on; legend('Location', 'best');
ylim([-2, 1]); % Обрезаем возможные выбросы на краях

% --- График 4: Геометрическая кривизна k(x) ---
subplot(2, 2, 4);
hold on;
plot(x_dense, k_spline, 'b-', 'LineWidth', 2, 'DisplayName', 'Кривизна сплайна');
plot(x_dense, k_true, 'r--', 'LineWidth', 2, 'DisplayName', 'Теор. кривизна (ступенька)');

title('Геометрическая кривизна кривой');
xlabel('x'); ylabel('\kappa');
grid on; legend('Location', 'best');
yline(1/R, 'g:', 'DisplayName', sprintf('1/R=%.3f', 1/R));

% Добавляем линии стыка
for i = 1:4
    subplot(2, 2, i);
    xline(-L/2, 'k--', 'Alpha', 0.3);
    xline(L/2, 'k--', 'Alpha', 0.3);
end

fprintf('Построение завершено.\n');