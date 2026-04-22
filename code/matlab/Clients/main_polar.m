%% Client Script: Polar Spline Demo
clear; close all; clc;

%% 1. Генерация данных (Четвертинка окружности с шумом)
rng(42);
R = 15.0;
n_points = 5;
theta = linspace(0, pi/2, n_points)';

% Декартовы координаты (истинные)
x_true = R * cos(theta);
y_true = R * sin(theta);

% Добавляем шум
noise_lvl = 0;
x_noisy = x_true + noise_lvl * randn(n_points, 1);
y_noisy = y_true + noise_lvl * randn(n_points, 1);

%% 2. Подготовка граничных условий
% Для четвертинки окружности: r = const, phi = theta.
% Параметр u - длина дуги: u = R*theta. => phi = u/R.
% dr/du = 0, dphi/du = 1/R.
% d^2r/du^2 = 0, d^2phi/du^2 = 0.

% Оценка длины дуги для BC
u_est = [0; cumsum(sqrt(diff(x_noisy).^2 + diff(y_noisy).^2))];
u_end = u_est(end);

bc_start = struct('r', R, 'dr', 0, 'ddr', 0, ...
                  'phi', 0, 'dphi', 1/R, 'ddphi', 0);
                  
bc_end   = struct('r', R, 'dr', 0, 'ddr', 0, ...
                  'phi', pi/2, 'dphi', 1/R, 'ddphi', 0);

%% 3. Создание и обучение сплайна
fprintf('Создание модели полярного сплайна...\n');
spline = Splines.PolarSmoothingSpline(x_noisy, y_noisy, bc_start, bc_end);

alpha = 0.9; % Вес точности (0.95 - близко к интерполяции)
spline.fit(alpha);

%% 4. Вычисление результатов для графиков
u_dense = linspace(spline.u(1), spline.u(end), 200)';

% Ручной расчет значений сплайна и производных на сетке
% (так как predict в классе упрощен для краткости)
n_seg = spline.n - 1;
r_vals = zeros(size(u_dense)); r1_vals = zeros(size(u_dense)); r2_vals = zeros(size(u_dense));
phi_vals = zeros(size(u_dense)); phi1_vals = zeros(size(u_dense)); phi2_vals = zeros(size(u_dense));

for k = 1:length(u_dense)
    u = u_dense(k);
    idx = find(spline.u <= u, 1, 'last');
    if idx > n_seg, idx = n_seg; end
    
    t = u - spline.u(idx);
    
    % --- R ---
    % Здесь нам нужны mi, Mi. В классе они не сохранены, 
    % но мы можем восстановить их из сохраненных коэффициентов и значений на концах.
    % Для упрощения демо, вытащим mi, Mi из "внутренностей" класса пересчетом.
    % В идеале класс должен хранить m и M.
    % Пропустим сложную логику и вычислим S(t) по упрощенной схеме:
    % Используем Сплайн Эрмита 3-го порядка? Нет, нужен квинтический.
    % Т.к. класс выше не возвращает производные, вычислим их численно для графиков.
end

% --- БЫСТРОЕ РЕШЕНИЕ ДЛЯ ГРАФИКОВ ---
% Чтобы не усложнять класс, используем interp1 с 'spline' для ВИЗУАЛИЗАЦИИ,
% но физический смысл останется верным.
% Для настоящих расчетов нужно расширить метод predict класса.

% Но мы хотим увидеть r'(phi). Сделаем проще:
% Восстановим m и M из решения оптимизации.
% В методе fit они есть, но как локальные переменные.
% Поскольку класс handle, модифицируем его, чтобы сохранить m и M.
% Я предполагаю, что вы добавите свойства m_opt и M_opt в класс.
% Для скрипта я сделаю "хак":
% В классе в unpack_results сохраните m_r, M_r, m_phi, M_phi!

% (Предполагаем, что в свойствах объекта spline есть m_r_opt, M_r_opt...)
% Если их нет, скрипт упадет. Добавьте их в класс:
% properties: m_r_opt, M_r_opt, m_phi_opt, M_phi_opt
% И в unpack_results: obj.m_r_opt = m_r; ...

m_r = spline.m_r_opt; M_r = spline.M_r_opt;
m_phi = spline.m_phi_opt; M_phi = spline.M_phi_opt;

for k = 1:length(u_dense)
    u = u_dense(k);
    idx = find(spline.u <= u, 1, 'last');
    if idx > n_seg, idx = n_seg; end
    
    t = u - spline.u(idx);
    
    % R(t)
    cr = spline.coeffs_r{idx};
    a3=cr(1); a4=cr(2); a5=cr(3);
    yi_r = spline.r_opt(idx); mi_r = m_r(idx); Mi_r = M_r(idx);
    
    r_vals(k) = yi_r + mi_r*t + 0.5*Mi_r*t^2 + a3*t^3 + a4*t^4 + a5*t^5;
    r1_vals(k) = mi_r + Mi_r*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4;
    r2_vals(k) = Mi_r + 6*a3*t + 12*a4*t^2 + 20*a5*t^3;
    
    % Phi(t)
    cp = spline.coeffs_phi{idx};
    a3=cp(1); a4=cp(2); a5=cp(3);
    yi_p = spline.phi_opt(idx); mi_p = m_phi(idx); Mi_p = M_phi(idx);
    
    phi_vals(k) = yi_p + mi_p*t + 0.5*Mi_p*t^2 + a3*t^3 + a4*t^4 + a5*t^5;
    phi1_vals(k) = mi_p + Mi_p*t + 3*a3*t^2 + 4*a3*t^3 + 5*a5*t^4; % Ошибка! 4*a4*t^3
    phi1_vals(k) = mi_p + Mi_p*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4;
    phi2_vals(k) = Mi_p + 6*a3*t + 12*a4*t^2 + 20*a5*t^3;
end

% Обратный перевод в Декартовы для графика 1
x_fit = r_vals .* cos(phi_vals);
y_fit = r_vals .* sin(phi_vals);

%% 5. Вычисление производных r(phi)
% dr/dphi = (dr/du) / (dphi/du)
r_prime_phi = r1_vals ./ phi1_vals;

% d^2r/dphi^2 = (phi' * r'' - r' * phi'') / (phi')^3
r_double_prime_phi = (phi1_vals .* r2_vals - r1_vals .* phi2_vals) ./ (phi1_vals.^3);

%% 6. Визуализация
figure('Position', [100, 100, 1200, 800]);

% График 1: Кривая в XY
subplot(2, 3, 1);
plot(x_noisy, y_noisy, 'ro', 'DisplayName', 'Данные');
hold on;
plot(x_fit, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн');
plot(R*cos(theta), R*sin(theta), 'k--', 'DisplayName', 'Истина');
axis equal; grid on; legend;
title('Кривая в XY');

% График 2: r(u) и phi(u)
subplot(2, 3, 2);
yyaxis left; plot(u_dense, r_vals, 'b-', 'LineWidth', 1.5); ylabel('r(u)');
yyaxis right; plot(u_dense, phi_vals, 'r-', 'LineWidth', 1.5); ylabel('\phi(u)');
grid on; title('Координаты от параметра u');

% График 3: Производные по u
subplot(2, 3, 3);
yyaxis left; plot(u_dense, r1_vals, 'b-'); ylabel('r''(u)');
yyaxis right; plot(u_dense, phi1_vals, 'r-'); ylabel('\phi''(u)');
grid on; title('Первая производная по u');

% График 4: Вторые производные по u
subplot(2, 3, 4);
yyaxis left; plot(u_dense, r2_vals, 'b-'); ylabel('r''''(u)');
yyaxis right; plot(u_dense, phi2_vals, 'r-'); ylabel('\phi''''(u)');
grid on; title('Вторая производная по u');

% График 5: r'(phi)
subplot(2, 3, 5);
plot(phi_vals, r_prime_phi, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--');
% Теоретическое значение для окружности: r=R, r'=0
plot([0 pi/2], [0 0], 'k--', 'LineWidth', 1);
grid on;
xlabel('\phi'); ylabel('dr/d\phi');
title('Производная r''(\phi)');

% График 6: r''(phi)
subplot(2, 3, 6);
plot(phi_vals, r_double_prime_phi, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--');
grid on;
xlabel('\phi'); ylabel('d^2r/d\phi^2');
title('Вторая производная r''''(\phi)');

fprintf('Готово.\n');