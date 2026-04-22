%% Тестирование параметрических сплайнов на образующей баллона
% Баллон состоит из двух полусфер и цилиндра
% Используется новый класс ParametricQuinticSplineNew с явными граничными условиями

clear all; close all; clc;

%% Параметры баллона
R = 3.0;            % Радиус баллона
L = 8.0;            % Длина цилиндрической части
noise_std = 0e-3;   % Уровень шума (робастность)

%% Генерация точек образующей (три участка)
% Баллон параллелен Ox, середина в (0, R), симметричен относительно Oy

% --- Участок 1: Левая полусфера ---
% Окружность с центром (-L/2, 0), движение от полюса вверх-вправо
n_sphere_left = 5;
s1 = linspace(0, pi*R/2, n_sphere_left)';  % длина дуги
theta1 = s1 / R;  % угол в радианах от полюса

% x = -L/2 - R*cos(?), но ? отсчитывается от ? (полюс слева)
% При движении от полюса: x = -L/2 - R*cos(? - ?) = -L/2 + R*cos(?) - неправильно!
% Правильно: x = -L/2 + R*cos(? - s/R) при движении от полюса вверх
% Или: x = -L/2 - R*cos(s/R), y = R*sin(s/R) для движения от левого полюса

x1_true = -L/2 - R * cos(s1/R);
y1_true = R * sin(s1/R);

% --- Участок 2: Цилиндр ---
% Горизонтальный участок y = R
n_cyl = 3;
s2 = linspace(0, L, n_cyl)';
x2_true = -L/2 + s2;
y2_true = R * ones(n_cyl, 1);

% --- Участок 3: Правая полусфера ---
% Окружность с центром (L/2, 0), движение от стыка к полюсу
n_sphere_right = 5;
s3 = linspace(0, pi*R/2, n_sphere_right)';
% Движение от верхней точки к правому полюсу
% x = L/2 + R*sin(s/R), y = R*cos(s/R)
x3_true = L/2 + R * sin(s3/R);
y3_true = R * cos(s3/R);

%% Проверка стыковки
fprintf('=== Проверка стыковки участков ===\n');
fprintf('Левая полусфера: (%.3f, %.3f) -> (%.3f, %.3f)\n', ...
    x1_true(1), y1_true(1), x1_true(end), y1_true(end));
fprintf('Цилиндр:         (%.3f, %.3f) -> (%.3f, %.3f)\n', ...
    x2_true(1), y2_true(1), x2_true(end), y2_true(end));
fprintf('Правая полусфера: (%.3f, %.3f) -> (%.3f, %.3f)\n', ...
    x3_true(1), y3_true(1), x3_true(end), y3_true(end));

%% Добавление шума (тест на робастность)
rng(42);
x1_noisy = x1_true + noise_std * randn(n_sphere_left, 1);
y1_noisy = y1_true + noise_std * randn(n_sphere_left, 1);
x2_noisy = x2_true + noise_std * randn(n_cyl, 1);
y2_noisy = y2_true + noise_std * randn(n_cyl, 1);
x3_noisy = x3_true + noise_std * randn(n_sphere_right, 1);
y3_noisy = y3_true + noise_std * randn(n_sphere_right, 1);

%% Граничные условия (явные первые и вторые производные)
% Формулы для окружности радиуса R при параметризации длиной дуги s:
% x'' = -cos(s/R)/R, y'' = -sin(s/R)/R (для окружности с центром в начале)

% === Сплайн 1 (левая полусфера): s ? [0, ?R/2] ===
% Начало (s=0, левый полюс): x = -L/2-R, y = 0
%   x' = sin(0) = 0, y' = cos(0) = 1
%   x'' = cos(0)/R = 1/R, y'' = -sin(0)/R = 0
% Конец (s=?R/2, стык с цилиндром): x = -L/2, y = R
%   x' = sin(?/2) = 1, y' = cos(?/2) = 0
%   x'' = cos(?/2)/R = 0, y'' = -sin(?/2)/R = -1/R

bc1_start = struct('m_x', 0, 'm_y', 1, 'M_x', 1/R, 'M_y', 0);
bc1_end   = struct('m_x', 1, 'm_y', 0, 'M_x', 0, 'M_y', -1/R);

% === Сплайн 2 (цилиндр): s ? [0, L] ===
% Начало: x = -L/2, y = R, прямая линия
%   x' = 1, y' = 0
%   x'' = 0, y'' = 0
% Конец: то же самое

bc2_start = struct('m_x', 1, 'm_y', 0, 'M_x', 0, 'M_y', 0);
bc2_end   = struct('m_x', 1, 'm_y', 0, 'M_x', 0, 'M_y', 0);

% === Сплайн 3 (правая полусфера): s ? [0, ?R/2] ===
% Начало (s=0, стык с цилиндром): x = L/2, y = R
%   x' = cos(0) = 1, y' = -sin(0) = 0
%   x'' = -sin(0)/R = 0, y'' = -cos(0)/R = -1/R ... 
% НЕТ! Для нашей параметризации:
% x = L/2 + R*sin(s/R), y = R*cos(s/R)
% x' = cos(s/R), y' = -sin(s/R)
% x'' = -sin(s/R)/R, y'' = -cos(s/R)/R
% При s=0: x'=1, y'=0, x''=0, y''=-1/R -- но это кривизна направлена вниз!

% ПРАВИЛЬНО для правой полусферы (движение от стыка к полюсу):
% Окружность с центром (L/2, 0), кривизна направлена к центру (влево)
% x = L/2 + R*sin(s/R), y = R*cos(s/R)
% x' = cos(s/R), y' = -sin(s/R)
% x'' = -sin(s/R)/R, y'' = -cos(s/R)/R

% При s=0 (стык): x'=1, y'=0, x''=0, y''=-1/R
% При s=?R/2 (полюс): x'=0, y'=-1, x''=-1/R, y''=0

bc3_start = struct('m_x', 1, 'm_y', 0, 'M_x', 0, 'M_y', -1/R);
bc3_end   = struct('m_x', 0, 'm_y', -1, 'M_x', -1/R, 'M_y', 0);

%% Создание и обучение сплайнов (ОДИН РАЗ для каждого alpha)
alphas = [0.99, 0.95, 0.90];
alpha_main = 0.95;

points1 = [x1_noisy, y1_noisy];
points2 = [x2_noisy, y2_noisy];
points3 = [x3_noisy, y3_noisy];

fprintf('\n=== Обучение сплайнов ===\n');
fprintf('Параметры баллона: R=%.1f, L=%.1f\n', R, L);
fprintf('Уровень шума: sigma=%.3f\n', noise_std);

% Обучаем все варианты и сохраняем результаты
n_alphas = length(alphas);
spline1_results = cell(n_alphas, 1);
spline2_results = cell(n_alphas, 1);
spline3_results = cell(n_alphas, 1);

for i = 1:n_alphas
    alpha = alphas(i);
    fprintf('\n--- alpha = %.2f ---\n', alpha);
    
    % Создаем и обучаем сплайны
    s1 = Splines.ParametricQuinticSplineNew(points1, bc1_start, bc1_end);
    s2 = Splines.ParametricQuinticSplineNew(points2, bc2_start, bc2_end);
    s3 = Splines.ParametricQuinticSplineNew(points3, bc3_start, bc3_end);
    
    s1.fit(alpha);
    s2.fit(alpha);
    s3.fit(alpha);
    
    % Сохраняем объекты сплайнов
    spline1_results{i} = s1;
    spline2_results{i} = s2;
    spline3_results{i} = s3;
end

% Индекс основного варианта
idx_main = find(alphas == alpha_main);

%% Генерация плотных данных для визуализации
n_dense = 100;

% Находим основной индекс (второй в массиве alphas = 0.95)
idx_main = 2;

% Используем сохраненные результаты
spline1 = spline1_results{idx_main};
spline2 = spline2_results{idx_main};
spline3 = spline3_results{idx_main};

t1_dense = linspace(0, spline1.u(end), n_dense)';
t2_dense = linspace(0, spline2.u(end), n_dense)';
t3_dense = linspace(0, spline3.u(end), n_dense)';

% Предсказания для основного alpha
pts1 = spline1.predict(t1_dense, 0);
pts2 = spline2.predict(t2_dense, 0);
pts3 = spline3.predict(t3_dense, 0);

deriv1_1 = spline1.predict(t1_dense, 1);
deriv1_2 = spline2.predict(t2_dense, 1);
deriv1_3 = spline3.predict(t3_dense, 1);

deriv2_1 = spline1.predict(t1_dense, 2);
deriv2_2 = spline2.predict(t2_dense, 2);
deriv2_3 = spline3.predict(t3_dense, 2);

curv1 = spline1.curvature(t1_dense);
curv2 = spline2.curvature(t2_dense);
curv3 = spline3.curvature(t3_dense);

%% Объединение данных для единых графиков
u_boundary1 = spline1.u(end);
u_boundary2 = spline1.u(end) + spline2.u(end);

u_combined = [t1_dense; u_boundary1 + t2_dense; u_boundary2 + t3_dense];

x_combined = [pts1(:,1); pts2(:,1); pts3(:,1)];
y_combined = [pts1(:,2); pts2(:,2); pts3(:,2)];

xp_combined = [deriv1_1(:,1); deriv1_2(:,1); deriv1_3(:,1)];
yp_combined = [deriv1_1(:,2); deriv1_2(:,2); deriv1_3(:,2)];

xpp_combined = [deriv2_1(:,1); deriv2_2(:,1); deriv2_3(:,1)];
ypp_combined = [deriv2_1(:,2); deriv2_2(:,2); deriv2_3(:,2)];

curv_combined = [curv1; curv2; curv3];

n1 = length(t1_dense);
n2 = length(t2_dense);

%% Вычисление y'(x) и y''(x)
% dy/dx = y'(u) / x'(u)
% d?y/dx? = (x'(u)*y''(u) - y'(u)*x''(u)) / [x'(u)]?

yprime_x = yp_combined ./ xp_combined;

denom = xp_combined.^3;
denom(abs(denom) < 1e-10) = sign(denom(abs(denom) < 1e-10)) * 1e-10;
denom(denom == 0) = 1e-10;
yprime2_x = (xp_combined .* ypp_combined - yp_combined .* xpp_combined) ./ denom;

%% ============================================
%% ВИЗУАЛИЗАЦИЯ
%% ============================================

%% Рисунок 1: Образующая кривая
figure('Name', 'Образующая баллона', 'Position', [100, 100, 1200, 600]);

subplot(1, 2, 1);
hold on;

% Истинная форма для заливки
x1_fill = linspace(-L/2-R, -L/2, 100)';
y1_fill = sqrt(R^2 - (x1_fill + L/2).^2);
x2_fill = linspace(-L/2, L/2, 100)';
y2_fill = R * ones(100, 1);
x3_fill = linspace(L/2, L/2+R, 100)';
y3_fill = sqrt(R^2 - (x3_fill - L/2).^2);

fill([x1_fill; x2_fill; x3_fill; flip(x3_fill); flip(x2_fill); flip(x1_fill)], ...
     [y1_fill; y2_fill; y3_fill; -flip(y3_fill); -flip(y2_fill); -flip(y1_fill)], ...
     [0.9, 0.9, 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3, ...
     'DisplayName', 'Истинная форма');

% Зашумленные точки
plot(x1_noisy, y1_noisy, 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r', ...
     'DisplayName', 'Участок 1');
plot(x2_noisy, y2_noisy, 'go', 'MarkerSize', 4, 'MarkerFaceColor', 'g', ...
     'DisplayName', 'Участок 2');
plot(x3_noisy, y3_noisy, 'bo', 'MarkerSize', 4, 'MarkerFaceColor', 'b', ...
     'DisplayName', 'Участок 3');

% Сплайны
plot(pts1(:,1), pts1(:,2), 'r-', 'LineWidth', 2, 'DisplayName', 'Сплайн 1');
plot(pts2(:,1), pts2(:,2), 'g-', 'LineWidth', 2, 'DisplayName', 'Сплайн 2');
plot(pts3(:,1), pts3(:,2), 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн 3');

% Точки стыка
plot([-L/2, L/2], [R, R], 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'y', ...
     'DisplayName', 'Точки стыка');

xlabel('x'); ylabel('y');
title(sprintf('Образующая баллона (\\alpha = %.2f)', alpha_main));
axis equal; grid on; legend('Location', 'best');
xlim([-L/2-R-1, L/2+R+1]);

subplot(1, 2, 2);
hold on;

% Сравнение alpha (используем сохраненные результаты)
colors = {'r', 'g', 'b', 'm'};
for i = 1:n_alphas
    s1 = spline1_results{i};
    s2 = spline2_results{i};
    s3 = spline3_results{i};
    
    t1_d = linspace(0, s1.u(end), n_dense)';
    t2_d = linspace(0, s2.u(end), n_dense)';
    t3_d = linspace(0, s3.u(end), n_dense)';
    
    p1 = s1.predict(t1_d, 0);
    p2 = s2.predict(t2_d, 0);
    p3 = s3.predict(t3_d, 0);
    
    plot([p1(:,1); p2(:,1); p3(:,1)], [p1(:,2); p2(:,2); p3(:,2)], ...
         colors{i}, 'LineWidth', 2, 'DisplayName', sprintf('\\alpha = %.2f', alphas(i)));
end

% Истинная кривая
plot([x1_fill; x2_fill; x3_fill], [y1_fill; y2_fill; y3_fill], ...
     'k--', 'LineWidth', 1.5, 'DisplayName', 'Истинная форма');

xlabel('x'); ylabel('y');
title('Сравнение уровней сглаживания');
axis equal; grid on; legend('Location', 'best');

%% Рисунок 2: x'(u)
figure('Name', 'x''(u) - первая производная', 'Position', [150, 150, 800, 500]);
hold on;

plot(u_combined(1:n1), xp_combined(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1 (левая полусфера)');
plot(u_combined(n1+1:n1+n2), xp_combined(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2 (цилиндр)');
plot(u_combined(n1+n2+1:end), xp_combined(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3 (правая полусфера)');

xline(u_boundary1, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Границы участков');
xline(u_boundary2, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xlabel('Параметр u'); ylabel('x''(u)');
title('Первая производная x по параметру u');
legend('Location', 'best'); grid on;

%% Рисунок 3: y'(u)
figure('Name', 'y''(u) - первая производная', 'Position', [200, 200, 800, 500]);
hold on;

plot(u_combined(1:n1), yp_combined(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1 (левая полусфера)');
plot(u_combined(n1+1:n1+n2), yp_combined(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2 (цилиндр)');
plot(u_combined(n1+n2+1:end), yp_combined(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3 (правая полусфера)');

xline(u_boundary1, 'k--', 'LineWidth', 1.5);
xline(u_boundary2, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);

xlabel('Параметр u'); ylabel('y''(u)');
title('Первая производная y по параметру u');
legend('Location', 'best'); grid on;

%% Рисунок 4: x''(u)
figure('Name', 'x''''(u) - вторая производная', 'Position', [250, 250, 800, 500]);
hold on;

plot(u_combined(1:n1), xpp_combined(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1 (левая полусфера)');
plot(u_combined(n1+1:n1+n2), xpp_combined(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2 (цилиндр)');
plot(u_combined(n1+n2+1:end), xpp_combined(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3 (правая полусфера)');

xline(u_boundary1, 'k--', 'LineWidth', 1.5);
xline(u_boundary2, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);

% Теоретические значения
yline(1/R, 'm--', 'LineWidth', 1, 'DisplayName', sprintf('Теор. +1/R = %.3f', 1/R));
yline(-1/R, 'c--', 'LineWidth', 1, 'DisplayName', sprintf('Теор. -1/R = %.3f', -1/R));

xlabel('Параметр u'); ylabel('x''''(u)');
title('Вторая производная x по параметру u');
legend('Location', 'best'); grid on;

%% Рисунок 5: y''(u)
figure('Name', 'y''''(u) - вторая производная', 'Position', [300, 300, 800, 500]);
hold on;

plot(u_combined(1:n1), ypp_combined(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1 (левая полусфера)');
plot(u_combined(n1+1:n1+n2), ypp_combined(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2 (цилиндр)');
plot(u_combined(n1+n2+1:end), ypp_combined(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3 (правая полусфера)');

xline(u_boundary1, 'k--', 'LineWidth', 1.5);
xline(u_boundary2, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);

yline(-1/R, 'm--', 'LineWidth', 1, 'DisplayName', sprintf('Теор. -1/R = %.3f', -1/R));

xlabel('Параметр u'); ylabel('y''''(u)');
title('Вторая производная y по параметру u');
legend('Location', 'best'); grid on;

%% Рисунок 6: y'(x)
figure('Name', 'y''(x) - производная по x', 'Position', [350, 350, 800, 500]);
hold on;

plot(x_combined(1:n1), yprime_x(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1 (левая полусфера)');
plot(x_combined(n1+1:n1+n2), yprime_x(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2 (цилиндр)');
plot(x_combined(n1+n2+1:end), yprime_x(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3 (правая полусфера)');

% Теоретические значения
x_left_th = linspace(-L/2-R+0.01, -L/2-0.01, 50)';
yprime_left_th = -(x_left_th + L/2) ./ sqrt(R^2 - (x_left_th + L/2).^2);
x_right_th = linspace(L/2+0.01, L/2+R-0.01, 50)';
yprime_right_th = -(x_right_th - L/2) ./ sqrt(R^2 - (x_right_th - L/2).^2);

plot(x_left_th, yprime_left_th, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Теоретическая');
plot(x_right_th, yprime_right_th, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xline(-L/2, 'k--', 'LineWidth', 1.5);
xline(L/2, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);

xlabel('x'); ylabel('dy/dx');
title('Первая производная y по x');
legend('Location', 'best'); grid on;

%% Рисунок 7: y''(x)
figure('Name', 'y''''(x) - вторая производная по x', 'Position', [400, 400, 800, 500]);
hold on;

plot(x_combined(1:n1), yprime2_x(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1 (левая полусфера)');
plot(x_combined(n1+1:n1+n2), yprime2_x(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2 (цилиндр)');
plot(x_combined(n1+n2+1:end), yprime2_x(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3 (правая полусфера)');

% Теоретические значения второй производной
yprime2_left_th = -R^2 ./ (R^2 - (x_left_th + L/2).^2).^(3/2);
yprime2_right_th = -R^2 ./ (R^2 - (x_right_th - L/2).^2).^(3/2);

plot(x_left_th, yprime2_left_th, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Теоретическая');
plot(x_right_th, yprime2_right_th, 'k--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

xline(-L/2, 'k--', 'LineWidth', 1.5);
xline(L/2, 'k--', 'LineWidth', 1.5);
yline(0, 'k:', 'LineWidth', 1);
yline(-1/R, 'm--', 'LineWidth', 1);

% Аннотации разрывов
text(-L/2 + 0.5, -0.3, sprintf('Разрыв: \\Deltay'''' = %.3f', 1/R), 'FontSize', 10);
text(L/2 + 0.5, -0.3, sprintf('Разрыв: \\Deltay'''' = %.3f', 1/R), 'FontSize', 10);

xlabel('x'); ylabel('d?y/dx?');
title('Вторая производная y по x (демонстрация разрыва кривизны)');
legend('Location', 'best'); grid on;

%% Рисунок 8: Кривизна
figure('Name', 'Кривизна', 'Position', [450, 450, 1200, 500]);

subplot(1, 2, 1);
hold on;
plot(u_combined(1:n1), curv_combined(1:n1), 'r-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 1');
plot(u_combined(n1+1:n1+n2), curv_combined(n1+1:n1+n2), 'g-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 2');
plot(u_combined(n1+n2+1:end), curv_combined(n1+n2+1:end), 'b-', 'LineWidth', 2, ...
     'DisplayName', 'Сплайн 3');

xline(u_boundary1, 'k--', 'LineWidth', 1.5);
xline(u_boundary2, 'k--', 'LineWidth', 1.5);
yline(1/R, 'm--', 'LineWidth', 1.5, 'DisplayName', sprintf('Теор. 1/R = %.4f', 1/R));
yline(0, 'k:', 'LineWidth', 1);

xlabel('Параметр u'); ylabel('\kappa(u)');
title(sprintf('Кривизна (\\alpha = %.2f)', alpha_main));
legend('Location', 'best'); grid on;

subplot(1, 2, 2);
hold on;

% Сравнение разных alpha (используем сохраненные результаты)
colors = {'r', 'g', 'b'};
for i = 1:n_alphas
    s1 = spline1_results{i};
    s2 = spline2_results{i};
    s3 = spline3_results{i};
    
    t1_d = linspace(0, s1.u(end), n_dense)';
    t2_d = linspace(0, s2.u(end), n_dense)';
    t3_d = linspace(0, s3.u(end), n_dense)';
    
    c1 = s1.curvature(t1_d);
    c2 = s2.curvature(t2_d);
    c3 = s3.curvature(t3_d);
    
    u_comb = [t1_d; s1.u(end) + t2_d; s1.u(end) + s2.u(end) + t3_d];
    c_comb = [c1; c2; c3];
    
    plot(u_comb, c_comb, colors{i}, 'LineWidth', 1.5, ...
         'DisplayName', sprintf('\\alpha = %.2f', alphas(i)));
end

xline(u_boundary1, 'k--', 'LineWidth', 1.5);
xline(u_boundary2, 'k--', 'LineWidth', 1.5);
yline(1/R, 'k--', 'LineWidth', 1, 'DisplayName', 'Теор. 1/R');
yline(0, 'k:', 'LineWidth', 1);

xlabel('Параметр u'); ylabel('\kappa(u)');
title('Влияние сглаживания на разрыв кривизны');
legend('Location', 'best'); grid on;

%% Рисунок 9: Отклонение от истинной формы
figure('Name', 'Отклонение', 'Position', [500, 500, 1000, 400]);

% Отклонения для основного alpha
y1_true_dense = sqrt(R^2 - (pts1(:,1) + L/2).^2);
y3_true_dense = sqrt(R^2 - (pts3(:,1) - L/2).^2);
dev1 = abs(pts1(:,2) - y1_true_dense);
dev2 = abs(pts2(:,2) - R);
dev3 = abs(pts3(:,2) - y3_true_dense);

subplot(1, 2, 1);
hold on;
plot(pts1(:,1), dev1, 'r-', 'LineWidth', 2, 'DisplayName', 'Сплайн 1');
plot(pts2(:,1), dev2, 'g-', 'LineWidth', 2, 'DisplayName', 'Сплайн 2');
plot(pts3(:,1), dev3, 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн 3');

xline(-L/2, 'k--', 'LineWidth', 1);
xline(L/2, 'k--', 'LineWidth', 1);

xlabel('x'); ylabel('|y_{сплайн} - y_{истин}|');
title('Абсолютное отклонение от истинной формы');
legend('Location', 'best'); grid on;

subplot(1, 2, 2);
hold on;

% Среднее отклонение для разных alpha (используем сохраненные результаты)
mean_devs = zeros(n_alphas, 1);
for i = 1:n_alphas
    s1 = spline1_results{i};
    s2 = spline2_results{i};
    s3 = spline3_results{i};
    
    t1_d = linspace(0, s1.u(end), n_dense)';
    t2_d = linspace(0, s2.u(end), n_dense)';
    t3_d = linspace(0, s3.u(end), n_dense)';
    
    p1 = s1.predict(t1_d, 0);
    p2 = s2.predict(t2_d, 0);
    p3 = s3.predict(t3_d, 0);
    
    y1_t = sqrt(R^2 - (p1(:,1) + L/2).^2);
    y3_t = sqrt(R^2 - (p3(:,1) - L/2).^2);
    
    dev_total = [abs(p1(:,2) - y1_t); abs(p2(:,2) - R); abs(p3(:,2) - y3_t)];
    mean_devs(i) = mean(dev_total);
end

bar(alphas, mean_devs, 'FaceColor', [0.3, 0.6, 0.9]);
xlabel('\alpha'); ylabel('Среднее отклонение');
title('Зависимость отклонения от параметра сглаживания');
grid on;
xticks(alphas);

%% Вывод статистики
fprintf('\n=== Результаты ===\n');
fprintf('Параметры: R=%.2f, L=%.2f, noise=%.3f\n', R, L, noise_std);
fprintf('Параметр сглаживания: alpha=%.2f\n', alpha_main);

fprintf('\nДлины участков:\n');
fprintf('  Левая полусфера: %.3f (теор: %.3f)\n', spline1.u(end), pi*R/2);
fprintf('  Цилиндр: %.3f (теор: %.3f)\n', spline2.u(end), L);
fprintf('  Правая полусфера: %.3f (теор: %.3f)\n', spline3.u(end), pi*R/2);

fprintf('\nКривизна (теор. для полусфер: 1/R = %.4f):\n', 1/R);
fprintf('  Сплайн 1: средняя = %.4f, std = %.4f\n', mean(curv1), std(curv1));
fprintf('  Сплайн 2: средняя = %.4f, std = %.4f\n', mean(curv2), std(curv2));
fprintf('  Сплайн 3: средняя = %.4f, std = %.4f\n', mean(curv3), std(curv3));

fprintf('\nРазрыв кривизны в стыках:\n');
fprintf('  Левый стык: |??(конец) - ??(начало)| = %.4f (теор: %.4f)\n', ...
    abs(curv1(end) - curv2(1)), 1/R);
fprintf('  Правый стык: |??(конец) - ??(начало)| = %.4f (теор: %.4f)\n', ...
    abs(curv2(end) - curv3(1)), 1/R);

fprintf('\nВторые производные на границах:\n');
fprintf('  Сплайн 1: x''''(0)=%.4f (теор: %.4f), y''''(0)=%.4f (теор: %.4f)\n', ...
    deriv2_1(1,1), 1/R, deriv2_1(1,2), 0);
fprintf('  Сплайн 1: x''''(end)=%.4f (теор: %.4f), y''''(end)=%.4f (теор: %.4f)\n', ...
    deriv2_1(end,1), 0, deriv2_1(end,2), -1/R);

fprintf('\nСреднее отклонение от истинной формы:\n');
for i = 1:n_alphas
    fprintf('  alpha=%.2f: %.6f\n', alphas(i), mean_devs(i));
end

fprintf('\n=== Тест завершён ===\n');