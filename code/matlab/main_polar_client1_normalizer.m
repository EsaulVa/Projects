%% CLIENT SCRIPT: Normalized Polar Spline Architecture
% Полный цикл: Нормализация -> Оптимизация -> Денормализация
clear; close all; clc;

%% 1. Загрузка данных
filename = 'data.txt';
if ~exist(filename, 'file')
    error('Файл data.txt не найден!');
end
M = readmatrix(filename);
x_glob = M(:, 1);
y_glob = M(:, 2);

%% 2. Геометрическая подготовка (Center Finding)
% Шаг 1: Находим центр аппроксимирующей окружности
A_mat = [2*x_glob, 2*y_glob, ones(length(x_glob), 1)];
b_vec = x_glob.^2 + y_glob.^2;
p_fit = A_mat \ b_vec;
x_center = p_fit(1);
y_center = p_fit(2);
R_est = sqrt(p_fit(3) + x_center^2 + y_center^2);

fprintf('Геометрический центр: (%.3f, %.3f), Радиус: %.3f\n', x_center, y_center, R_est);

%% 3. Нормализация данных
% Создаем нормализатор, используя найденный центр как начало координат
% Масштаб выберем как средний радиус (или R_est)
norm_unit = Services.Normalizer();
norm_unit.Center = [x_center, y_center]; % Центр жестко задаем
norm_unit.ScaleFactor = R_est;           % Масштабируем по радиусу

% Переход к безразмерным координатам
[x_norm, y_norm] = norm_unit.normalize(x_glob, y_glob);

fprintf('Данные нормализованы. Масштаб L = %.3f\n', norm_unit.ScaleFactor);

%% 4. Подготовка Граничных Условий (в нормализованной системе)
% Условия задачи: Начало - Natural, Конец - y'=0, y''=0 (прямая)

% --- START (Естественные условия) ---
% В нормализованных координатах просто фиксируем точку, производные свободны (NaN)
bc_start.r = struct('value', sqrt(x_norm(1)^2 + y_norm(1)^2));
bc_start.phi = struct('value', atan2(y_norm(1), x_norm(1)));
% deriv1, deriv2 не указаны -> автоматически NaN -> Free

% --- END (Жесткие условия) ---
% Дано: y'(x) = 0, y''(x) = 0.
% Важно: y'' нужно масштабировать! y''_norm = y''_phys * L.
k_end = 0;      % y'(x)
q_end = 0;      % y''(x) физическая
q_norm = q_end * norm_unit.ScaleFactor; % Масштабируем кривизну

x_e = x_norm(end); 
y_e = y_norm(end);

% Вычисляем связь для первых производных
link_end = compute_polar_slope_link(x_e, y_e, k_end);
C_end = link_end.C;

% Вычисляем значения вторых производных (для v=1)
bc_calc = convert_cartesian_bc_to_polar(x_e, y_e, k_end, q_norm);

% Сборка структуры bc_end для решателя
bc_end.r.value   = sqrt(x_e^2 + y_e^2);
bc_end.r.deriv1  = struct('coupling', C_end);   % Связь (геометрия)
% bc_end.r.deriv2  = bc_calc.r.deriv2;            % Фиксация (кривизна)
bc_end.r.deriv2  = NaN;            % Фиксация (кривизна)

bc_end.phi.value = atan2(y_e, x_e);
bc_end.phi.deriv1 = NaN;                        % Свободная (скорость)
% bc_end.phi.deriv2 = bc_calc.phi.deriv2;         % Фиксация (кривизна)
bc_end.phi.deriv2 = NaN;         % Фиксация (кривизна)
% Добавляем спец-условие на y''
bc_end.y_deriv2 = struct('target', 0, 'weight', 1);
%% 5. Запуск Оптимизации (в нормализованном пространстве)
alpha = 1-1e-3; 
beta=0;
gamma=1e-4;
spline = Splines.PolarSmoothingSplineCoupledNew(x_norm, y_norm);
spline.setBC(bc_start, bc_end); 

fprintf('Запуск оптимизации в нормализованном пространстве...\n');
spline.fit(alpha,beta,gamma);

%% 6. Извлечение и Денормализация результатов
u_dense = linspace(spline.u(1), spline.u(end), 200)';
n = spline.n;

% Извлекаем результаты оптимизации (они безразмерные)
r_res = spline.results.r;
phi_res = spline.results.phi;

geom = spline.geom_r;

% Массивы для безразмерных величин
r_vals_n = zeros(size(u_dense)); phi_vals_n = zeros(size(u_dense));
r1_vals_n = zeros(size(u_dense)); phi1_vals_n = zeros(size(u_dense));
r2_vals_n = zeros(size(u_dense)); phi2_vals_n = zeros(size(u_dense));

% Цикл расчета (как раньше)
for k = 1:length(u_dense)
    u = u_dense(k);
    idx = find(spline.u <= u, 1, 'last');
    if idx > n-1, idx = n-1; end
    t = u - spline.u(idx);
    h = spline.u(idx+1) - spline.u(idx);
    
    cr = geom.getSegmentCoeffs(r_res.v(idx), r_res.m(idx), r_res.M(idx), ...
                               r_res.v(idx+1), r_res.m(idx+1), r_res.M(idx+1), h);
    r_vals_n(k) = geom.evalValue(t, r_res.v(idx), r_res.m(idx), r_res.M(idx), cr);
    r1_vals_n(k) = geom.evalDeriv1(t, r_res.m(idx), r_res.M(idx), cr);
    r2_vals_n(k) = geom.evalDeriv2(t, r_res.M(idx), cr);
    
    cp = geom.getSegmentCoeffs(phi_res.v(idx), phi_res.m(idx), phi_res.M(idx), ...
                               phi_res.v(idx+1), phi_res.m(idx+1), phi_res.M(idx+1), h);
    phi_vals_n(k) = geom.evalValue(t, phi_res.v(idx), phi_res.m(idx), phi_res.M(idx), cp);
    phi1_vals_n(k) = geom.evalDeriv1(t, phi_res.m(idx), phi_res.M(idx), cp);
    phi2_vals_n(k) = geom.evalDeriv2(t, phi_res.M(idx), cp);
end

% === ДЕНОРМАЛИЗАЦИЯ ===
% 1. Координаты
r_vals = norm_unit.denormRadius(r_vals_n); % r = r_norm * L
phi_vals = phi_vals_n;                      % phi не меняется

x_fit = r_vals .* cos(phi_vals) + x_center;
y_fit = r_vals .* sin(phi_vals) + y_center;

% 2. Первые производные (инвариантны, не меняются)
r1_vals = r1_vals_n;
phi1_vals = phi1_vals_n;

% 2. Производные по U
% r'(u) - инвариантно, совпадает
r1_vals = r1_vals_n; 

% phi'(u) ~ 1/L, значит надо делить на L (было упущено)
phi1_vals = phi1_vals_n / norm_unit.ScaleFactor; 

% r''(u) ~ 1/L, делим на L (как было)
r2_vals = norm_unit.denormDeriv2(r2_vals_n);

% phi''(u) ~ 1/L^2, делим на L^2 (было упущено)
phi2_vals = phi2_vals_n / (norm_unit.ScaleFactor^2); 

% % 3. Вторые производные (обратное масштабирование)
% r2_vals = norm_unit.denormDeriv2(r2_vals_n);
% phi2_vals = norm_unit.denormDeriv2(phi2_vals_n);


% 4. Производные по phi (r' = dr/du / dphi/du -> инвариантно)
dr_dphi = r1_vals ./ phi1_vals;
d2r_dphi2 = (phi1_vals .* r2_vals - r1_vals .* phi2_vals) ./ (phi1_vals.^3);

%% 7. Визуализация
% Графики те же, что и в прошлой версии
figure('Name', 'XY Curve', 'Position', [100, 500, 600, 500]);
plot(x_glob, y_glob, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Данные');
hold on;
plot(x_fit, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн');
plot(x_center, y_center, 'kx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Центр');
axis equal; grid on; legend;
xlabel('X'); ylabel('Y');
title('Кривая в декартовых координатах (Физические величины)');

figure('Name', 'Derivatives by u', 'Position', [150, 400, 800, 500]);
subplot(2, 1, 1);
yyaxis left; plot(u_dense, r1_vals, 'b-', 'LineWidth', 1.5); ylabel('r''(u)');
yyaxis right; plot(u_dense, phi1_vals, 'r-', 'LineWidth', 1.5); ylabel('\phi''(u)');
grid on; title('Первые производные (безразмерные)');

subplot(2, 1, 2);
yyaxis left; plot(u_dense, r2_vals, 'b-', 'LineWidth', 1.5); ylabel('r''''(u) [1/L]');
yyaxis right; plot(u_dense, phi2_vals, 'r-', 'LineWidth', 1.5); ylabel('\phi''''(u) [1/L]');
grid on; title('Вторые производные (физические)');
xlabel('Параметр u');

figure('Name', 'r''(phi)', 'Position', [200, 300, 600, 500]);
plot(phi_vals, dr_dphi, 'b-', 'LineWidth', 1.5);
hold on; yline(0, 'k--');
grid on; xlabel('\phi (rad)'); ylabel('dr/d\phi');
title('Первая производная r''(\phi)');

figure('Name', 'r''''(phi)', 'Position', [250, 200, 600, 500]);
plot(phi_vals, d2r_dphi2, 'b-', 'LineWidth', 1.5);
hold on; yline(0, 'k--');
grid on; xlabel('\phi (rad)'); ylabel('d^2r/d\phi^2');
title('Вторая производная r''''(\phi)');

%% 8. Вычисление декартовых производных
[dy_dx, d2y_dx2] = get_cartesian_derivatives(r_vals, phi_vals, r1_vals, phi1_vals, r2_vals, phi2_vals);

% Визуализация y'(x)
figure('Name', 'Cartesian Derivatives', 'Position', [300, 200, 800, 500]);
subplot(2,1,1);
plot(u_dense, dy_dx, 'b-', 'LineWidth', 1.5);
grid on;
title('Наклон касательной y''(x)');
ylabel('y''(x)');
xlabel('Параметр u');

% Проверка граничных условий на конце
fprintf('Проверка на конечной точке:\n');
fprintf('  Задано y''(end) = 0. Получено: %.4f\n', dy_dx(end));
fprintf('  Задано y''''(end) = 0. Получено: %.4f\n', d2y_dx2(end));

subplot(2,1,2);
plot(u_dense, d2y_dx2, 'r-', 'LineWidth', 1.5);
grid on;
title('Кривизна профиля y''''(x)');
ylabel('y''''(x)');
xlabel('Параметр u');

fprintf('Построение завершено.\n');