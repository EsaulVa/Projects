%% Клиент для "Пути 1": Единый сглаживающий сплайн
% Построение траектории намотки с приоритетом плавности (C2 непрерывность)
% Решение проблемы скачка кривизны через глобальную оптимизацию

clear all; close all; clc;

%% 1. Параметры баллона и генерация данных
R = 3.0;            % Радиус
L = 8.0;            % Длина цилиндра
noise_std = 1e-3;   % Уровень шума (имитация погрешностей измерения/модели)

% Генерируем точки для всех участков сразу (единый массив)
n_sph = 7;         % Точек на полусферу
n_cyl = 7;         % Точек на цилиндр

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

%% 2. Граничные условия (только на полюсах!)
% Мы задаем условия только на концах всего баллона.
% Внутренние узлы (включая стыки сферы и цилиндра) свободны для оптимизации.

% Левый полюс (начало)
% Направление: вверх [0, 1]. Кривизна: 1/R (выпуклость)
bc_start = struct('direction', [0.0, 1.0], 'curvature', 1.0/R);

% Правый полюс (конец)
% Направление: вниз [0, -1]. Кривизна: 1/R
bc_end   = struct('direction', [0.0, -1.0], 'curvature', 1.0/R);

%% 3. Создание и обучение единого сплайна
fprintf('=== Обучение единого сглаживающего сплайна (Путь 1) ===\n');

% Используем ваш класс ParametricQuinticSpline
% spline_unified = Splines.ParametricQuinticSpline(points_noisy, bc_start, bc_end);
spline_unified = Splines.ParametricQuinticSpline(points_noisy, bc_start, bc_end);

% Параметр сглаживания.
% alpha ~ 0.95-0.99 сохраняет геометрию, но позволяет сгладить шум.
% alpha ~ 0.85-0.90 сильнее сглаживает углы (создает более длинные переходы).
alpha = 0.99; 
spline_unified.fit(alpha);

fprintf('Оптимизация завершена.\n');

%% 4. Подготовка данных для визуализации и анализа
u_dense = linspace(0, spline_unified.u(end), 1000)';
curve_pts = spline_unified.predict(u_dense, 0);       % Точки кривой
tangents = spline_unified.predict(u_dense, 1);        % Первые производные
curvature_vals = spline_unified.curvature(u_dense);   % Кривизна

% "Истинная" геометрия для сравнения (без шума, кусочно-линейная логика)
% Восстанавливаем для графика истинную форму (куски окружностей и прямая)
u_true_idx = find(u_dense <= spline_unified.u(n_sph), 1, 'last');
u_cyl_idx  = find(u_dense <= spline_unified.u(n_sph + n_cyl - 2), 1, 'last');

% Расчет теоретической кривизны (ступенька)
% 1/R на сфере, 0 на цилиндре, 1/R на сфере
kappa_theory = zeros(size(u_dense));
kappa_theory(1:u_true_idx) = 1/R;
% Цилиндр: 0 (уже нули)
kappa_theory(u_cyl_idx+1:end) = 1/R;

% Находим индексы стыков на графике (примерно)
idx_junction1 = u_true_idx;
idx_junction2 = u_cyl_idx;

%% 5. Визуализация
set(groot, 'DefaultLineLineWidth', 1.5);
figure('Name', 'Результат: Путь 1 (Единый сплайн)', 'Position', [100, 100, 1400, 800]);

% --- График 1: Геометрия XY ---
subplot(2, 2, 1);
hold on;
plot(points_noisy(:,1), points_noisy(:,2), 'ro', 'MarkerSize', 4, ...
    'DisplayName', 'Зашумленные данные');
plot(curve_pts(:,1), curve_pts(:,2), 'b-', 'DisplayName', 'Сглаженный сплайн');
% Теоретическая форма
plot(pts1(:,1), pts1(:,2), 'k--', 'DisplayName', 'Теор. сфера');
plot(pts2(:,1), pts2(:,2), 'k--', 'HandleVisibility', 'off');
plot(pts3(:,1), pts3(:,2), 'k--', 'HandleVisibility', 'off');

title('Образующая баллона (Единый сплайн)');
xlabel('x'); ylabel('y');
axis equal; grid on; legend('Location', 'best');
xlim([min(curve_pts(:,1))-1, max(curve_pts(:,1))+1]);

% --- График 2: Кривизна (Ключевой график!) ---
subplot(2, 2, 2);
hold on;
plot(u_dense, curvature_vals, 'b-', 'LineWidth', 2, 'DisplayName', 'Кривизна сплайна');
plot(u_dense, kappa_theory, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Теор. кривизна (ступенька)');

% Подсветка зон перехода
yline(1/R, 'g:', 'DisplayName', sprintf('1/R = %.3f', 1/R));
yline(0, 'k:', 'HandleVisibility', 'off');

title('Сглаживание скачка кривизны');
xlabel('Параметр u (путь)'); ylabel('\kappa');
legend('Location', 'best'); grid on;
% Аннотации
text(u_dense(idx_junction1), 1/R*0.5, '  Зона плавного перехода', 'Color', 'b');

% --- График 3: Компоненты второй производной (M) ---
subplot(2, 2, 3);
hold on;
second_derivs = spline_unified.predict(u_dense, 2);
plot(u_dense, second_derivs(:,1), 'r-', 'DisplayName', 'x''''(u)');
plot(u_dense, second_derivs(:,2), 'b-', 'DisplayName', 'y''''(u)');
yline(0, 'k:');
title('Вторая производная (непрерывность C2)');
xlabel('Параметр u'); ylabel('Значение');
legend; grid on;

% --- График 4: Отклонение от истинной формы ---
subplot(2, 2, 4);
% Считаем отклонение как расстояние до ближайшей точки истинной геометрии
% (упрощенно: сравниваем Y сплайна с Y истинной формы для соответствующего X)
% Для цилиндра и сфер это просто |y_spline - R| или |dist_to_center - R|

% Более честно: отклонение от эталонной кривой
deviation = zeros(size(u_dense));
for i = 1:length(u_dense)
    x = curve_pts(i,1); y = curve_pts(i,2);
    
    if x < -L/2 % Левая сфера
        r_dist = sqrt((x + L/2)^2 + y^2);
        deviation(i) = abs(r_dist - R);
    elseif x > L/2 % Правая сфера
        r_dist = sqrt((x - L/2)^2 + y^2);
        deviation(i) = abs(r_dist - R);
    else % Цилиндр
        deviation(i) = abs(y - R);
    end
end

area(u_dense, deviation, 'FaceColor', [0.8, 0.9, 1], 'EdgeColor', 'b');
title('Отклонение от номинальной геометрии');
xlabel('Параметр u'); ylabel('Евклидово отклонение');
grid on;

% Добавляем маркеры стыков на все графики
for i = 1:4
    subplot(2, 2, i);
    xline(u_dense(idx_junction1), 'k--', 'Alpha', 0.3);
    xline(u_dense(idx_junction2), 'k--', 'Alpha', 0.3);
end

%% 6. Дополнительный анализ (на усмотрение)
% Оценка "длины перехода"
% Находим, где кривизна отличается от теоретической более чем на 5%
err = abs(curvature_vals - kappa_theory);
transition_indices = find(err > 0.05 * (1/R));
if ~isempty(transition_indices)
    u_trans_start = u_dense(transition_indices(1));
    u_trans_end = u_dense(transition_indices(end));
    len_transition = u_trans_end - u_trans_start;
    
    fprintf('\n=== Анализ переходной зоны ===\n');
    fprintf('Длина зоны сглаживания стыка: %.3f\n', len_transition);
    fprintf('Максимальное отклонение от формы: %.4f\n', max(deviation));
    fprintf('Примечание: Увеличение параметра alpha уменьшит зону перехода,\n');
    fprintf('но увеличит пиковые нагрузки (производные кривизны).\n');
end

%% 5.1. Анализ явных производных y(x)
% Вычисляем y'(x) и y''(x) по параметрическим формулам

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

%% 4. Расчет производных
u_dense = linspace(0, spline_unified.u(end), 1000)';

% Получаем положение
curve_pts = spline_unified.predict(u_dense, 0);

% Получаем производные.
% ВАЖНО: Метод predict(der=1) в классе нормализует вектор (делает единичным).
% Нам нужны истинные значения dx/du и dy/du.
% Используем вспомогательную функцию (написана ниже) для обхода нормализации.
[xp, yp] = Splines.getRawDerivatives(spline_unified, u_dense);

% Вторые производные (метод predict(der=2) не нормализует, можно использовать напрямую)
derivs2 = spline_unified.predict(u_dense, 2);
xpp = derivs2(:, 1);
ypp = derivs2(:, 2);

% Кривизна (для справки на графиках)
kappa = spline_unified.curvature(u_dense);

%% 5. Визуализация (4 отдельных окна)

% --- Окно 1: x'(u) ---
figure('Name', 'x''(u) - Скорость по X', 'Position', [100, 500, 800, 400]);
plot(u_dense, xp, 'b-', 'LineWidth', 2);
hold on;
grid on;
title('Первая производная x''(u) = dx/du');
xlabel('Параметр u (путь)');
ylabel('x''(u)');
% Отмечаем зоны
% Находим индексы переходов (грубо по u)
u_sep = spline_unified.u(n_sph); % конец левой сферы
xline(u_sep, 'r--', 'LineWidth', 1.5);
xline(spline_unified.u(end) - u_sep, 'r--', 'LineWidth', 1.5);
legend('x''(u)', 'Границы переходов', 'Location', 'best');

% --- Окно 2: y'(u) ---
figure('Name', 'y''(u) - Скорость по Y', 'Position', [200, 450, 800, 400]);
plot(u_dense, yp, 'b-', 'LineWidth', 2);
hold on;
grid on;
title('Первая производная y''(u) = dy/du');
xlabel('Параметр u (путь)');
ylabel('y''(u)');
xline(u_sep, 'r--', 'LineWidth', 1.5);
xline(spline_unified.u(end) - u_sep, 'r--', 'LineWidth', 1.5);
yline(0, 'k:'); % Линия нуля

% --- Окно 3: x''(u) ---
figure('Name', 'x''''(u) - Ускорение по X', 'Position', [300, 400, 800, 400]);
plot(u_dense, xpp, 'b-', 'LineWidth', 2);
hold on;
grid on;
title('Вторая производная x''''(u) = d^2x/du^2');
xlabel('Параметр u (путь)');
ylabel('x''''(u)');
xline(u_sep, 'r--', 'LineWidth', 1.5);
xline(spline_unified.u(end) - u_sep, 'r--', 'LineWidth', 1.5);
yline(0, 'k:');
% Теоретическое значение для сферы (центробежное ускорение)
% При движении по окружности со скоростью 1 (хорда ~ дуга), x'' = -cos(theta)/R
% На полюсе x'' -> -1/R (вектор направлен к центру)
% Но знак зависит от направления обхода.
% Для левой сферы (идем от полюса вверх-вправо): x растет, x'' > 0 в начале, потом падает?
% Сравним с графиком.

% --- Окно 4: y''(u) ---
figure('Name', 'y''''(u) - Ускорение по Y', 'Position', [400, 350, 800, 400]);
plot(u_dense, ypp, 'b-', 'LineWidth', 2);
hold on;
grid on;
title('Вторая производная y''''(u) = d^2y/du^2');
xlabel('Параметр u (путь)');
ylabel('y''''(u)');
xline(u_sep, 'r--', 'LineWidth', 1.5);
xline(spline_unified.u(end) - u_sep, 'r--', 'LineWidth', 1.5);
yline(0, 'k:');
% Теоретическое значение для цилиндра = 0
yline(0, 'g--', 'Теор. цилиндр (0)');

fprintf('Построение завершено.\n');

