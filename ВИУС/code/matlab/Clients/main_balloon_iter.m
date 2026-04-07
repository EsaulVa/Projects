%% Клиент: Итеративная линеаризация (Взвешенная кривизна)
% Тестирование на задаче баллона давления

clear all; clc;

%% 1. Генерация данных
R = 1.0; L = 1.0; noise_std = 1e-3;

n_sph = 5; n_cyl = 3;
theta1 = linspace(0, pi/2, n_sph)';
pts1 = [-L/2 - R*cos(theta1), R*sin(theta1)];
x_cyl = linspace(-L/2, L/2, n_cyl)';
pts2 = [x_cyl, R*ones(n_cyl, 1)];
theta3 = linspace(pi/2, 0, n_sph)';
pts3 = [L/2 + R*cos(theta3), R*sin(theta3)];

points_true = [pts1; pts2(2:end-1,:); pts3];
rng(42);
points_noisy = points_true + noise_std * randn(size(points_true));

%% 2. Граничные условия
bc_start = struct('direction', [0.0, 1.0], 'curvature', 1.0/R);
bc_end   = struct('direction', [0.0, -1.0], 'curvature', 1.0/R);

%% 3. Создание и обучение
spline_iter = IterSplines.ParametricQuinticSplineIterative(points_noisy, bc_start, bc_end);

alpha = 0.95;
max_iters = 3;
spline_iter.fit(alpha, max_iters);

%% 4. Подготовка данных для графиков
u_dense = linspace(0, spline_iter.u(end), 1000)';
curve_pts = spline_iter.predict(u_dense, 0);
curvature_vals = spline_iter.curvature(u_dense);

% "Сырые" производные
[xp, yp] = Splines.getRawDerivatives(spline_iter, u_dense);
deriv2 = spline_iter.predict(u_dense, 2);
xpp = deriv2(:,1); ypp = deriv2(:,2);

%% 5. Визуализация (4 графика + Геометрия)

% --- Окно 1: Геометрия XY ---
figure('Name', 'Итеративный сплайн: Геометрия', 'Position', [100, 100, 800, 600]);
plot(points_noisy(:,1), points_noisy(:,2), 'ro', 'MarkerSize', 4);
hold on;
plot(curve_pts(:,1), curve_pts(:,2), 'b-', 'LineWidth', 2);
title('Образующая баллона (Итеративная линеаризация)');
axis equal; grid on; legend('Данные', 'Сплайн');

% --- Окно 2: Кривизна ---
figure('Name', 'Кривизна', 'Position', [150, 150, 800, 400]);
plot(u_dense, curvature_vals, 'LineWidth', 2);
hold on;
yline(1/R, 'r--', 'LineWidth', 1);
title('Геометрическая кривизна \kappa(u)');
grid on;

% --- Окна 3-6: Производные (отдельно) ---
figure('Name', 'x''(u)', 'Position', [200, 500, 800, 400]);
plot(u_dense, xp, 'LineWidth', 2); grid on; title('x''(u)');

figure('Name', 'y''(u)', 'Position', [250, 450, 800, 400]);
plot(u_dense, yp, 'LineWidth', 2); grid on; title('y''(u)');

figure('Name', 'x''''(u)', 'Position', [300, 400, 800, 400]);
plot(u_dense, xpp, 'LineWidth', 2); grid on; title('x''''(u)');

figure('Name', 'y''''(u)', 'Position', [350, 350, 800, 400]);
plot(u_dense, ypp, 'LineWidth', 2); grid on; title('y''''(u)');

fprintf('Визуализация завершена.\n');

%% 5.1. Анализ явных производных y(x)
% Вычисляем y'(x) и y''(x) по параметрическим формулам
tangents=[xp  yp];
second_derivs=[xpp ypp];
% Точки, где x'(u) близко к нулю (полюса), нужно исключить для корректного графика
% В полюсах касательная вертикальна, y' -> inf
valid_idx = abs(tangents(:,1)) > 1e-3; 

% Первая производная y'(x)
y_prime = tangents(:,2) ./ tangents(:,1);

% Вторая производная y''(x)
% Формула: (x' * y'' - y' * x'') / (x')^3
numerator = tangents(:,1) .* second_derivs(:,2) - tangents(:,2) .* second_derivs(:,1);
denominator = tangents(:,1).^3;
y_double_prime = numerator ./ denominator;

% Теоретические значения для сравнения
% Сфера: y'' = -R^2 / y^3 (выводится из уравнения окружности)
% В зоне экватора (y=R): y'' = -1/R
% Цилиндр: y'' = 0

% --- Визуализация явных производных ---
figure('Name', 'Анализ явных производных y(x)', 'Position', [200, 200, 1200, 500]);

% График y'(x)
subplot(1, 2, 1);
hold on;
% Фильтруем "уши" на полюсах для красивого графика, ограничив по X
plot(curve_pts(valid_idx,1), y_prime(valid_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'y''(x) сплайн');

% Теоретический график
% Слева: от полюса (-L/2-R) до экватора (-L/2)
x_th_left = linspace(-L/2-R+0.1, -L/2, 100);
y_th_left_prime = -(x_th_left + L/2) ./ sqrt(R^2 - (x_th_left + L/2).^2);
plot(x_th_left, y_th_left_prime, 'k--', 'DisplayName', 'Теория (сфера)');

% Справа: от экватора (L/2) до полюса (L/2+R)
x_th_right = linspace(L/2, L/2+R-0.1, 100);
y_th_right_prime = -(x_th_right - L/2) ./ sqrt(R^2 - (x_th_right - L/2).^2);
plot(x_th_right, y_th_right_prime, 'k--', 'HandleVisibility', 'off');

title('Первая производная y''(x)');
xlabel('x'); ylabel('Тангенс угла наклона');
ylim([-2, 2]);
grid on; legend('Location', 'best');
% Отмечаем зону цилиндра
xline(-L/2, 'r:', 'Alpha', 0.5);
xline(L/2, 'r:', 'Alpha', 0.5);

% График y''(x)
subplot(1, 2, 2);
hold on;
plot(curve_pts(valid_idx,1), y_double_prime(valid_idx), 'b-', 'LineWidth', 2, 'DisplayName', 'y''''(x) сплайн');

% Теоретический график (ступенька)
% Сфера
plot(x_th_left, -R^2 ./ (R^2 - (x_th_left + L/2).^2).^(3/2), 'k--', 'DisplayName', 'Теория (сфера)');
% Цилиндр
plot([x_th_left(end), x_th_right(1)], [0, 0], 'k--', 'HandleVisibility', 'off');
% Сфера справа
plot(x_th_right, -R^2 ./ (R^2 - (x_th_right - L/2).^2).^(3/2), 'k--', 'HandleVisibility', 'off');

title('Вторая производная y''''(x) (Геометрическая кривизна)');
xlabel('x'); ylabel('Кривизна профиля');
ylim([-1.5/R, 0.5/R]);
grid on; legend('Location', 'best');
xline(-L/2, 'r:', 'Alpha', 0.5);
xline(L/2, 'r:', 'Alpha', 0.5);

% Аннотация
text(0, -0.2/R, 'Зона цилиндра (теор. y''''=0)', 'HorizontalAlignment', 'center');
text(-L/2-0.5, -1.2/R, 'Переход', 'Color', 'b', 'FontWeight', 'bold');