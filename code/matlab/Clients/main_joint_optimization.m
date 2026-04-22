%% MAIN CLIENT: Joint Optimization of Center and Spline
% Поиск оптимального центра (xc, yc), минимизирующего энергию сплайна.
clear; close all; clc;

%% 1. Загрузка/Генерация данных
filename = 'data.txt';
if ~exist(filename, 'file')
    error('Файл данных не найден!');
end
M = readmatrix(filename);
x_glob = M(:, 1);
y_glob = M(:, 2);

%% 2. Начальное приближение для центра (Circle Fit)
% Используем геометрический центр как хорошую стартовую точку
A = [2*x_glob, 2*y_glob, ones(length(x_glob), 1)];
b = x_glob.^2 + y_glob.^2;
p = A \ b;
x_c_init = p(1);
y_c_init = p(2);
% x_c_init = 0;
% y_c_init = 0;
initial_center = [x_c_init, y_c_init];

fprintf('Стартовая оценка центра: (%.3f, %.3f)\n', x_c_init, y_c_init);

%% 3. Спецификация Граничных Условий (Геометрия)
% Описываем, что мы знаем о кривой на границах в ДЕКАРТОВЫХ координатах.
% Функция оптимизации сама переведет это в полярные условия для любого центра.

% Начало: Ничего не знаем (Natural BC)
spec_start.dy_dx = NaN;
spec_start.d2y_dx2 = NaN;

% Конец: Знаем, что выходит горизонтально (y'=0) и прямолинейно (y''=0)
spec_end.dy_dx = 0;
spec_end.d2y_dx2 = 0;

% Собираем спецификацию
bc_spec.start = spec_start;
bc_spec.end = spec_end;

%% 4. Запуск внешней оптимизации (Поиск центра)
alpha = 0.98; % Параметр сглаживания

% Настройки оптимизатора (fminsearch)
options = optimset('Display', 'iter', ...
                    'TolX', 1e-4, ...
                    'TolFun', 1e-4, ...
                    'MaxFunEvals', 100); % Ограничиваем для скорости

fprintf('\nЗапуск поиска оптимального центра...\n');
obj_fun = @(c) run_spline_optimization(c, x_glob, y_glob, bc_spec, alpha);

% optimal_center = fminsearch(obj_fun, initial_center, options);
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter','MaxFunEvals', 100);
[optimal_center, f_val, flag] = fminunc(obj_fun, initial_center, options);

x_c_opt = optimal_center(1);
y_c_opt = optimal_center(2);

fprintf('\n========================================\n');
fprintf('ОПТИМАЛЬНЫЙ ЦЕНТР: (%.4f, %.4f)\n', x_c_opt, y_c_opt);
fprintf('========================================\n');

%% 5. Финальное построение с найденным центром
% Запускаем сплайн еще раз в оптимальной точке, чтобы получить данные для графиков
x_loc_opt = x_glob - x_c_opt;
y_loc_opt = y_glob - y_c_opt;

% Формируем конкретные ГУ для финального прогона
[bc_start_final.r.deriv1, bc_start_final.phi.deriv1, ...
 bc_start_final.r.deriv2, bc_start_final.phi.deriv2] = update_derivatives(spec_start, x_loc_opt(1), y_loc_opt(1));
 % ... (аналогично для end, либо вызов run_spline_optimization с флагом return_spline)

% Для простоты вызовем внутренний код вручную или используем класс напрямую:
spline_final = PolarSmoothingSplineCoupled(x_loc_opt, y_loc_opt);

% Формируем BC вручную для финала (аналогично логике в run_spline_optimization)
% ... (код инициализации bc_start_final, bc_end_final) ...
% Если нужно, этот блок можно вынести в отдельную функцию finalize_spline

spline_final.setBC(bc_start_final, bc_end_final);
spline_final.fit(alpha);

%% 6. Визуализация
figure('Name', 'Optimal Result', 'Position', [100, 100, 800, 600]);

% График кривой
subplot(1, 2, 1);
% ... код визуализации XY ...
title('Сплайн с оптимальным центром');

% График изменения центра
subplot(1, 2, 2);
plot(x_c_init, y_c_init, 'ko', 'MarkerFaceColor', 'g', 'DisplayName', 'Начальная оценка');
hold on;
plot(x_c_opt, y_c_opt, 'kx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Оптимальный центр');
axis equal; grid on; legend;
title('Положение центра');

fprintf('Готово.\n');