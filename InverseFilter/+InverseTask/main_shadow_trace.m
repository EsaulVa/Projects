% main_shadow_trace.m
% Воспроизведение трассировки по модели тени (аналог client_corridor_3.py)
% для кусочно-полиномиальных поверхностей вращения.

clear; clc; close all;

% ======================================================================
% 1. Данные поверхностей (из Python-кода)
% ======================================================================
% Оправка (E2)
phi_c_opravka = [0.0000000005642, -0.0000003012748, 0.0000605882383, -0.0099656628535, 2.9503573330764];
R_c_opravka   = [-344.1468891010463, 3932.5139101580062, -17756.7012553763525, ...
                 39582.6812110246392, -43518.6731429065403, 19122.1758646943599];
bound_opravka = [0, 234.27, 534.27, 768.54];
cyl_r_opravka = 251.705;
E2 = RevolutionSurface(phi_c_opravka, R_c_opravka, bound_opravka, cyl_r_opravka);

% Безопасность (E1)
phi_c_safe = [0.0000000000176, -0.0000000319663, 0.0000178315076, -0.0066486075257, 2.9473869159379];
R_c_safe   = [-200.4096721343111, 2428.8709925850990, -11585.7546890810463, ...
              27152.4105364360366, -31195.5446114188999, 14397.6607910855146];
bound_safe = [0, 327.978, 627.978, 955.956];
cyl_r_safe = 352.387;
E1 = RevolutionSurface(phi_c_safe, R_c_safe, bound_safe, cyl_r_safe);

z_offset = (bound_safe(4) - bound_opravka(4)) / 2;   % = (955.956-768.54)/2

% ======================================================================
% 2. Загрузка или генерация линии укладки (ЛУ) на оправке
% ======================================================================
if exist('LU_data.mat', 'file')
    data = load('LU_data.mat');
    if isfield(data, 'r')
        r_etalon = data.r';   % матрица 3 x N
        fprintf('Загружена ЛУ: %d точек\n', size(r_etalon,2));
    else
        error('Файл LU_data.mat не содержит переменной r');
    end
else
    warning('Файл LU_data.mat не найден. Генерируем тестовую ЛУ (винтовая линия на цилиндре)');
    % Тестовая линия укладки: винтовая линия на цилиндре (сегмент 2)
    u = linspace(bound_opravka(2)+10, bound_opravka(3)-10, 200);
    v = 0.5 * u;  % закрутка
    r_etalon = zeros(3, length(u));
    for i = 1:length(u)
        r_etalon(:,i) = E2.position(u(i), v(i));
    end
end

% Преобразуем линию укладки в объект Trajectory (ChordalTrajectory)
% Предполагаем, что класс Trajectory (бывший ChordalTrajectory) уже определён.
% Если нет, можно использовать простую интерполяцию через сплайны.
% Для простоты создадим структуру с методами через анонимные функции.
% Но лучше использовать готовый класс Trajectory из предыдущих ответов.

% ======================================================================
% 3. Трассировка лучей (модель тени)
% ======================================================================
% Сначала построим объект Trajectory для линии укладки
% Для этого используем функцию, которая строит сплайны от натурального параметра.
% Воспользуемся написанным ранее классом ChordalTrajectory (переименуем в Trajectory)
% Предполагаем, что Trajectory существует. Если нет, создадим его здесь.
% Временно создадим простую структуру:
if exist('Trajectory', 'class')
    lu_traj = Trajectory(r_etalon);
else
    error('Класс Trajectory не найден. Добавьте его из предыдущих ответов.');
end

num_points = 50000;
s_vals = linspace(0, lu_traj.totalLength(), num_points);
R_points = zeros(3, num_points);
lambda_max = zeros(num_points, 1);
valid_mask = false(num_points, 1);
phi_vals = zeros(num_points, 1);

fprintf('Трассировка %d точек...\n', num_points);
for i = 1:num_points
    s = s_vals(i);
    r = lu_traj.getPoint(s); r = r(:);               % столбец 3x1
    tau = lu_traj.getTangent(s); tau = tau(:);       % столбец 3x1
    
    % Нормаль к оправке в этой точке
    [u0, v0] = E2.uv_from_point(r');
    n = E2.normal(u0, v0); n = n(:);                % столбец 3x1
    
    % Проекция касательной на касательную плоскость
    tau_proj = tau - (tau' * n) * n;
    if norm(tau_proj) < 1e-6
        radial = -r; radial(3) = 0;
        radial = radial / norm(radial);
        tau_proj = radial - (radial' * n) * n;
        if norm(tau_proj) < 1e-6
            tau_proj = [1;0;0];
        end
    end
    tau_proj = tau_proj / norm(tau_proj);
    
    % Поиск пересечения луча r + t * tau_proj с E1
    t_min = 1.0;
    t_max = 1500;
    [t, pt] = trace_ray(E1, r, tau_proj, t_min, t_max);
    
    if ~isnan(t)
        R_points(:,i) = pt(:);
        lambda_max(i) = t;
        valid_mask(i) = true;
        dr = pt(:) - r(:);
        phi_vals(i) = dr' * n;   % скалярное произведение
    else
        R_points(:,i) = NaN(3,1);
        lambda_max(i) = inf;
        phi_vals(i) = NaN;
    end
    
    if mod(i, 50) == 0
        fprintf('  Обработано %d/%d\n', i, num_points);
    end
end
% for i = 1:num_points
%     s = s_vals(i);
%     r = lu_traj.getPoint(s);          % точка на оправке (3x1)
%     tau = lu_traj.getTangent(s);      % единичный касательный вектор к ЛУ
% 
%     % Нормаль к оправке в этой точке
%     [u0, v0] = E2.uv_from_point(r);
%     n = E2.normal(u0, v0);
% 
%     % Проекция касательной на касательную плоскость
%     tau_proj = tau - dot(tau, n) * n;
%     if norm(tau_proj) < 1e-6
%         % Если проекция слишком мала, используем радиальное направление
%         radial = -r; radial(3) = 0;
%         radial = radial / norm(radial);
%         tau_proj = radial - dot(radial, n) * n;
%         if norm(tau_proj) < 1e-6
%             tau_proj = [1;0;0];
%         end
%     end
%     tau_proj = tau_proj / norm(tau_proj);
% 
%     % Поиск пересечения луча r + t * tau_proj с поверхностью безопасности E1
%     t_min = 1.0;   % минимальная длина луча (можно 0)
%     t_max = 1500;
%     [t, pt] = trace_ray(E1, r, tau_proj, t_min, t_max);
% 
%     if ~isnan(t)
%         R_points(:,i) = pt;
%         lambda_max(i) = t;
%         valid_mask(i) = true;
%         % Вычисляем невязку Φ = ⟨R - r, n⟩
%         phi_vals(i) = dot(pt - r, n);
%     else
%         R_points(:,i) = NaN;
%         lambda_max(i) = Inf;
%         phi_vals(i) = NaN;
%     end
% 
%     if mod(i, 50) == 0
%         fprintf('  Обработано %d/%d\n', i, num_points);
%     end
% end

fprintf('Успешно: %d из %d\n', sum(valid_mask), num_points);

% ======================================================================
% 4. Сохранение результатов для обратной задачи
% ======================================================================
% Сохраняем точки R(z) (только валидные) и соответствующие s
valid_idx = find(valid_mask);
R_valid = R_points(:, valid_idx);
s_valid = s_vals(valid_idx)';
% Записываем в файл R_points.mat для дальнейшего использования
save('R_points.mat', 'R_valid', 's_valid');
fprintf('Сохранено %d точек R(z) в R_points.mat\n', size(R_valid,2));

% Также сохраняем CSV для просмотра
T = table(s_valid, R_valid(1,:)', R_valid(2,:)', R_valid(3,:)', ...
    lambda_max(valid_idx), phi_vals(valid_idx), ...
    'VariableNames', {'s','X','Y','Z','lambda','phi'});
writetable(T, 'tsn_shadow.csv');
fprintf('CSV сохранён в tsn_shadow.csv\n');

% ======================================================================
% 5. Визуализация
% ======================================================================
figure('Name', 'Трассировка (модель тени)', 'Color', 'w');
hold on; grid on; axis equal; view(3);

% Поверхность оправки E2
u_plot = linspace(E2.u_min, E2.u_max, 60);
v_plot = linspace(0, 2*pi, 40);
[Xm, Ym, Zm] = meshgrid(0,0,0); % заглушка, заполним в цикле
for i = 1:length(u_plot)
    for j = 1:length(v_plot)
        p = E2.position(u_plot(i), v_plot(j));
        Xm(i,j) = p(1); Ym(i,j) = p(2); Zm(i,j) = p(3);
    end
end
surf(Xm, Ym, Zm, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 1]);
% Поверхность безопасности E1 (смещённая по Z для наглядности? не смещаем)
[Xs, Ys, Zs] = meshgrid(0,0,0);
for i = 1:length(u_plot)
    for j = 1:length(v_plot)
        p = E1.position(u_plot(i), v_plot(j));
        Xs(i,j) = p(1); Ys(i,j) = p(2); Zs(i,j) = p(3);
    end
end
surf(Xs, Ys, Zs, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'FaceColor', [1 0.5 0.5]);

% Линия укладки (синяя)
lu_pts = zeros(3, num_points);
for i = 1:num_points
    lu_pts(:,i) = lu_traj.getPoint(s_vals(i));
end
plot3(lu_pts(1,:), lu_pts(2,:), lu_pts(3,:), 'b-', 'LineWidth', 2);
% Траектория R(z) (красная, только валидные)
plot3(R_valid(1,:), R_valid(2,:), R_valid(3,:), 'r-', 'LineWidth', 2);
% Точки R(z)
scatter3(R_valid(1,:), R_valid(2,:), R_valid(3,:), 20, 'r', 'filled');
% Лучи (каждый 5-й)
skip = max(1, floor(num_points/50));
for idx = valid_idx(1:skip:end)
    p1 = lu_pts(:,idx);
    p2 = R_points(:,idx);
    if ~any(isnan(p2))
        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], 'g-', 'LineWidth', 0.8);
    end
end

xlabel('X'); ylabel('Y'); zlabel('Z');
legend('Оправка (E2)', 'Безопасность (E1)', 'Линия укладки', 'ТСН R(z)', 'Лучи');
title('Трассировка лучей по проекции касательной (модель тени)');
hold off;

% Дополнительно: график невязки Φ(s)
figure('Name', 'Невязка связи');
plot(s_valid, phi_vals(valid_idx), 'b.-', 'MarkerSize', 8);
xlabel('s (длина дуги ЛУ)'); ylabel('\Phi = \langle R-r, n \rangle');
title('Невязка \Phi(s) – должна быть близка к нулю');
grid on;