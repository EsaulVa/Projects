%% CLIENT SCRIPT: Polar Spline with Robust Geometric BC
clear; close all; clc;

%% 1. Считывание данных
filename = 'data.txt';
if ~exist(filename, 'file')
    error('Файл data.txt не найден!');
end

M = readmatrix(filename);
x_glob = M(:, 1);
y_glob = M(:, 2);
n_points = length(x_glob);
fprintf('Считано %d точек.\n', n_points);

if n_points < 4
    error('Слишком мало точек для построения сплайна 5-го порядка. Нужно минимум 4-5.');
end

%% 2. Поиск геометрического центра (Fitting Circle)
A = [2*x_glob, 2*y_glob, ones(n_points, 1)];
b = x_glob.^2 + y_glob.^2;
p = A \ b; 

x_center = p(1);
y_center = p(2);
R_est = sqrt(p(3) + x_center^2 + y_center^2);

fprintf('Найден центр: (%.3f, %.3f), Радиус: %.3f\n', x_center, y_center, R_est);

%% 3. Переход в локальные полярные координаты
x_loc = x_glob - x_center;
y_loc = y_glob - y_center;

% Параметризация u (длина хорды)
diffs = sqrt(diff(x_loc).^2 + diff(y_loc).^2);
u_data = [0; cumsum(diffs)];

%% 4. Формирование "Геометрических" граничных условий
% Гипотеза: кривая ведет себя как окружность с найденным центром.
% Для окружности r(u) = const => r' = 0, r'' = 0.
% Это идеальные условия для минимума энергии изгиба.

% Начало
r_start = sqrt(x_loc(1)^2 + y_loc(1)^2);
phi_start = atan2(y_loc(1), x_loc(1));

% Конец
r_end = sqrt(x_loc(end)^2 + y_loc(end)^2);
phi_end = atan2(y_loc(end), x_loc(end));

% Оценка dphi/du (скорость изменения угла) через среднюю длину дуги
% Для окружности dphi/du = 1/R
avg_R = (r_start + r_end) / 2; 
dphi_approx = 1 / avg_R; 

% Заполняем BC (предполагаем окружность!)
bc_start = struct('r', r_start, 'dr', 0, 'ddr', 0, ...
                  'phi', phi_start, 'dphi', dphi_approx, 'ddphi', 0);

bc_end   = struct('r', r_end, 'dr', 0, 'ddr', 0, ...
                  'phi', phi_end, 'dphi', dphi_approx, 'ddphi', 0);

disp('Граничные условия установлены по модели окружности:');
disp(bc_start);

%% 5. Запуск оптимизации
alpha = 0.95; % Вес точности (0.95 = почти интерполяция)
fprintf('Запуск оптимизации (alpha=%.2f)...\n', alpha);

spline = Splines.PolarSmoothingSpline(x_loc, y_loc, bc_start, bc_end);
spline.fit(alpha);

%% 6. Визуализация
u_dense = linspace(spline.u(1), spline.u(end), 200)';

% Проверка сохраненных результатов
if ~isprop(spline, 'm_r_opt') || isempty(spline.m_r_opt)
    error('Оптимизация не удалась или результаты не сохранены.');
end

m_r = spline.m_r_opt; M_r = spline.M_r_opt;
m_phi = spline.m_phi_opt; M_phi = spline.M_phi_opt;

% Расчет значений
r_fit = zeros(size(u_dense)); phi_fit = zeros(size(u_dense));
r1_fit = zeros(size(u_dense)); phi1_fit = zeros(size(u_dense));
r2_fit = zeros(size(u_dense)); phi2_fit = zeros(size(u_dense));

n_seg = spline.n - 1;
for k = 1:length(u_dense)
    u = u_dense(k);
    idx = find(spline.u <= u, 1, 'last');
    if idx > n_seg, idx = n_seg; end
    t = u - spline.u(idx);
    
    % R
    cr = spline.coeffs_r{idx}; a3=cr(1); a4=cr(2); a5=cr(3);
    yi_r=spline.r_opt(idx); mi_r=m_r(idx); Mi_r=M_r(idx);
    
    r_fit(k) = yi_r + mi_r*t + 0.5*Mi_r*t^2 + a3*t^3 + a4*t^4 + a5*t^5;
    r1_fit(k) = mi_r + Mi_r*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4;
    r2_fit(k) = Mi_r + 6*a3*t + 12*a4*t^2 + 20*a5*t^3;
    
    % Phi
    cp = spline.coeffs_phi{idx}; a3=cp(1); a4=cp(2); a5=cp(3);
    yi_p=spline.phi_opt(idx); mi_p=m_phi(idx); Mi_p=M_phi(idx);
    
    phi_fit(k) = yi_p + mi_p*t + 0.5*Mi_p*t^2 + a3*t^3 + a4*t^4 + a5*t^5;
    phi1_fit(k) = mi_p + Mi_p*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4;
    phi2_fit(k) = Mi_p + 6*a3*t + 12*a4*t^2 + 20*a5*t^3;
end

% Обратный перевод в глобальные координаты
x_fit = r_fit .* cos(phi_fit) + x_center;
y_fit = r_fit .* sin(phi_fit) + y_center;

% Производные по phi
denom = phi1_fit;
denom(abs(denom) < 1e-9) = 1e-9; % защита от деления на 0
dr_dphi = r1_fit ./ denom;
d2r_dphi2 = (denom .* r2_fit - r1_fit .* phi2_fit) ./ (denom.^3);

%% Графики
figure('Name', 'Результат оптимизации', 'Position', [100, 100, 1200, 700]);

% 1. XY
subplot(2, 3, 1);
plot(x_glob, y_glob, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Данные');
hold on;
plot(x_fit, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн');
plot(x_center, y_center, 'kx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Центр');
axis equal; grid on; legend;
title('Глобальная система координат');

% 2. r(phi)
subplot(2, 3, 2);
plot(phi_fit, r_fit, 'b-', 'LineWidth', 1.5);
hold on;
plot([phi_start, phi_end], [R_est, R_est], 'k--', 'DisplayName', 'Оценка R');
grid on; xlabel('\phi'); ylabel('r');
title('r(\phi) относительно центра');

% 3. Невязка
subplot(2, 3, 3);
plot(phi_fit, r_fit - R_est, 'b-');
grid on; xlabel('\phi'); ylabel('\Delta r');
title('Отклонение от идеальной окружности');

% Остальные
subplot(2, 3, 4); plot(u_dense, r1_fit); grid on; title('r''(u) - должно быть около 0');
subplot(2, 3, 5); plot(phi_fit, dr_dphi); grid on; title('dr/d\phi');
subplot(2, 3, 6); plot(phi_fit, d2r_dphi2); grid on; title('d^2r/d\phi^2');

fprintf('Построение завершено.\n');