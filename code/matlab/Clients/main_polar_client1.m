%% CLIENT SCRIPT: Polar Spline with Advanced Plotting
% Полный цикл: поиск центра, ГУ, оптимизация, визуализация всех производных
clear; clc;

%% 1. Подготовка данных
filename = 'data.txt';
if ~exist(filename, 'file')
    % Создадим тестовый файл, если его нет (ваши данные)
    fid = fopen(filename, 'w');
    fprintf(fid, '2.646 11.007 2.98 -0.428\n');
    fprintf(fid, '9.797 26.197 1.609 -0.094\n');
    fprintf(fid, '20.397 39.113 0.87 -0.051\n');
    fprintf(fid, '33.257 47.007 0.405 -0.027\n');
    fprintf(fid, '49.5 50.341 0 0\n');
    fclose(fid);
end
M = readmatrix(filename);
x_glob = M(:, 1);
y_glob = M(:, 2);

%% 2. Поиск центра (Circle Fit)
% Линейный МНК для уравнения окружности
A = [2*x_glob, 2*y_glob, ones(length(x_glob), 1)];
b = x_glob.^2 + y_glob.^2;
p = A \ b;
x_center = p(1);
y_center = p(2);
R_est = sqrt(p(3) + x_center^2 + y_center^2);
fprintf('Найден центр: (%.2f, %.2f), Радиус: %.2f\n', x_center, y_center, R_est);

%% 3. Переход к локальным координатам
x_loc = x_glob - x_center;
y_loc = y_glob - y_center;

%% 4. Формирование ГУ (Геометрическая гипотеза)
% Гипотеза: дуга окружности. r'=0, r''=0, phi'=1/R, phi''=0.
% NaN означает "свободная переменная" (Natural BC)

bc_start.r = struct('value', sqrt(x_loc(1)^2+y_loc(1)^2));
bc_start.phi = struct('value', atan2(y_loc(1), x_loc(1)));

%% Пример использования
x_e = x_loc(end); 
y_e = y_loc(end);
dy_dx_end = 0; % Ваше условие y'(x) на конце
dy_dx=0;
d2y_dx2=0;
% 1. Сначала вычисляем связь для КОНЕЧНОЙ точки
bc_link_end = compute_polar_slope_link(x_e, y_e, dy_dx_end);
C_end = bc_link_end.C;

% 2. Формируем структуру ГУ
% Можно либо использовать convert... и перезаписать, либо собрать вручную.
bc = convert_cartesian_bc_to_polar(x_e, y_e, dy_dx, d2y_dx2);

% 2. Настраиваем первые производные (Связь, скорость свободна)
C = bc.r.deriv1 / bc.phi.deriv1; % Коэффициент связи из геометрии
% bc.r.deriv1 = struct('coupling', C); % r' зависит от phi'
% bc.phi.deriv1 = NaN;                 % phi' свободна

% 3. Настраиваем вторые производные (Жесткая фиксация)
% Функция уже вычислила их для v=1.
% Это заставит оптимизатор "подтянуть" скорость к v=1, 
% чтобы сохранить заданную геометрическую кривизну.
bc.r.deriv2 = bc.r.deriv2;   % Это просто число
bc.phi.deriv2 = bc.phi.deriv2; % Это просто число

% Собирать вручную понятнее:

bc_end.r.value   = sqrt(x_e^2 + y_e^2);      % Фиксируем значение r
bc_end.r.deriv2  = bc.r.deriv2;                        % Фиксируем r'' (из y''=0)
bc_end.r.deriv1  = struct('coupling', C_end);% r' зависит от phi' (СВЯЗЬ)

bc_end.phi.value = atan2(y_e, x_e);          % Фиксируем значение phi
bc_end.phi.deriv2 = bc.phi.deriv2;                       % Фиксируем phi'' (из y''=0)
bc_end.phi.deriv1 = NaN;                     % phi' свободна (ИЩЕТСЯ ОПТИМИЗАТОРОМ)

% Применяем
% spline.setBC(bc_start, bc_end);
%% 5. Создание и оптимизация сплайна
spline = Splines.PolarSmoothingSplineCoupled(x_loc, y_loc);
spline.setBC(bc_start, bc_end); 

% alpha = 1; 
% beta=0.1;
alpha = 0.95; 
beta=0;
spline.fit(alpha,beta);

%% 6. Вычисление плотной сетки для графиков
u_dense = linspace(spline.u(1), spline.u(end), 200)';
n = spline.n;

% Извлекаем результаты оптимизации
r_res = spline.results.r;
phi_res = spline.results.phi;

% Инициализация массивов
r_vals = zeros(size(u_dense)); 
phi_vals = zeros(size(u_dense));
r1_vals = zeros(size(u_dense)); phi1_vals = zeros(size(u_dense)); % 1-е производные по u
r2_vals = zeros(size(u_dense)); phi2_vals = zeros(size(u_dense)); % 2-е производные по u

geom = spline.geom_r; % Экземпляр движка (он же используется для phi)

% Цикл расчета значений сплайна
for k = 1:length(u_dense)
    u = u_dense(k);
    % Поиск сегмента
    idx = find(spline.u <= u, 1, 'last');
    if idx > n-1, idx = n-1; end
    t = u - spline.u(idx);
    h = spline.u(idx+1) - spline.u(idx);
    
    % --- R Segment ---
    cr = geom.getSegmentCoeffs(r_res.v(idx), r_res.m(idx), r_res.M(idx), ...
                               r_res.v(idx+1), r_res.m(idx+1), r_res.M(idx+1), h);
    r_vals(k) = geom.evalValue(t, r_res.v(idx), r_res.m(idx), r_res.M(idx), cr);
    r1_vals(k) = geom.evalDeriv1(t, r_res.m(idx), r_res.M(idx), cr);
    r2_vals(k) = geom.evalDeriv2(t, r_res.M(idx), cr);
    
    % --- Phi Segment ---
    cp = geom.getSegmentCoeffs(phi_res.v(idx), phi_res.m(idx), phi_res.M(idx), ...
                               phi_res.v(idx+1), phi_res.m(idx+1), phi_res.M(idx+1), h);
    phi_vals(k) = geom.evalValue(t, phi_res.v(idx), phi_res.m(idx), phi_res.M(idx), cp);
    phi1_vals(k) = geom.evalDeriv1(t, phi_res.m(idx), phi_res.M(idx), cp);
    phi2_vals(k) = geom.evalDeriv2(t, phi_res.M(idx), cp);
end




% Обратный перевод в глобальные координаты
x_fit = r_vals .* cos(phi_vals) + x_center;
y_fit = r_vals .* sin(phi_vals) + y_center;

% Вычисление производных r(phi)
% dr/dphi = (dr/du) / (dphi/du)
dr_dphi = r1_vals ./ phi1_vals;

% d^2r/dphi^2 = (phi' * r'' - r' * phi'') / (phi')^3
d2r_dphi2 = (phi1_vals .* r2_vals - r1_vals .* phi2_vals) ./ (phi1_vals.^3);

%% 7. Визуализация (Каждый график в отдельном окне)

% --- Окно 1: Кривая в XY ---
figure('Name', 'XY Curve', 'Position', [100, 500, 600, 500]);
plot(x_glob, y_glob, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Данные');
hold on;
plot(x_fit, y_fit, 'b-', 'LineWidth', 2, 'DisplayName', 'Сплайн');
plot(x_center, y_center, 'kx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Центр');
axis equal; grid on; legend;
xlabel('X'); ylabel('Y');
title('Кривая в декартовых координатах');

% --- Окно 2: Производные по u (r'', phi'') ---
figure('Name', 'Derivatives by u', 'Position', [150, 400, 800, 500]);

subplot(2, 1, 1);
yyaxis left; plot(u_dense, r1_vals, 'b-', 'LineWidth', 1.5); ylabel('r''(u)');
yyaxis right; plot(u_dense, phi1_vals, 'r-', 'LineWidth', 1.5); ylabel('\phi''(u)');
grid on; title('Первые производные по u');

subplot(2, 1, 2);
yyaxis left; plot(u_dense, r2_vals, 'b-', 'LineWidth', 1.5); ylabel('r''''(u)');
yyaxis right; plot(u_dense, phi2_vals, 'r-', 'LineWidth', 1.5); ylabel('\phi''''(u)');
grid on; title('Вторые производные по u');
xlabel('Параметр u');

% --- Окно 3: r'(phi) ---
figure('Name', 'r''(phi)', 'Position', [200, 300, 600, 500]);
plot(phi_vals, dr_dphi, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('\phi (rad)'); ylabel('dr/d\phi');
title('Первая производная r''(\phi)');

% --- Окно 4: r''(phi) ---
figure('Name', 'r''''(phi)', 'Position', [250, 200, 600, 500]);
plot(phi_vals, d2r_dphi2, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 0.5);
grid on;
xlabel('\phi (rad)'); ylabel('d^2r/d\phi^2');
title('Вторая производная r''''(\phi)');
% % --- Окно 3: r'(phi) ---
% figure('Name', 'r''(phi)', 'Position', [200, 300, 600, 500]);
% polar(phi_vals, dr_dphi);
% hold on;
% yline(0, 'k--', 'LineWidth', 0.5);
% grid on;
% xlabel('\phi (rad)'); ylabel('dr/d\phi');
% title('Первая производная r''(\phi)');
% 
% % --- Окно 4: r''(phi) ---
% figure('Name', 'r''''(phi)', 'Position', [250, 200, 600, 500]);
% polar(phi_vals, d2r_dphi2);
% hold on;
% yline(0, 'k--', 'LineWidth', 0.5);
% grid on;
% xlabel('\phi (rad)'); ylabel('d^2r/d\phi^2');
% title('Вторая производная r''''(\phi)');
% fprintf('Построение завершено.\n');