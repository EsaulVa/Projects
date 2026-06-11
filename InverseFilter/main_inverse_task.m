% main_inverse_task.m
% Обратная задача: восстановление линии укладки на сглаженной оправке E3
% по траектории точки схода R(z) на поверхности безопасности E1.

clear; clc; close all;

%% 1. Загрузка данных
% Сглаженная оправка E3
load('E3_smoothed.mat', 'E3');
fprintf('E3: z ∈ [%.3f, %.3f] мм\n', E3.z_min, E3.z_max);

% Исходная (спроецированная) линия укладки на E3 (для сравнения)
load('lu_on_E3.mat', 'lu_on_E3');
fprintf('lu_on_E3: длина дуги = %.3f мм\n', lu_on_E3.totalLength());

% Траектория точки схода R(z) на E1 (результат трассировки)
load('R_trajectory_on_E1.mat', 'R_valid', 's_valid');
% R_valid – массив 3×M (декартовы точки)
% s_valid – параметр исходной ЛУ (не нужен для обратной задачи)
fprintf('Загружено %d точек R(z)\n', size(R_valid,2));

%% 2. Построение траектории R(z) как объекта Trajectory
% R_valid может содержать точки в произвольном порядке (вдоль ЛУ они идут по возрастанию s)
% Упорядочим по s_valid (если нужно)
[s_sorted, idx] = sort(s_valid);
R_sorted = R_valid(:, idx);
R_traj = InverseTask.Trajectory(R_sorted);
fprintf('Траектория R(z): длина дуги z_max = %.3f мм\n', R_traj.totalLength());

%% 3. Начальные условия для обратной задачи
% Берем начальную точку и касательную с lu_on_E3 при s=0
% u0_geo = lu_on_E3.getPoint(0);          % 3×1
% tau0_geo = lu_on_E3.getTangent(0);      % 3×1

% Для поверхности E3 нам нужны параметры (u,v) – это (s, v), где s – длина дуги
% Но lu_on_E3 уже задана в декартовых координатах, лежащих на E3.
% Чтобы получить параметры (s0, v0) на E3 для точки u0_geo:
% z0 = u0_geo(3);
% v0 = atan2(u0_geo(2), u0_geo(1));
% s0 = E3.s_from_z(z0);   % длина дуги меридиана для этой z
% fprintf('Начальные условия: z0 = %.3f, v0 = %.3f, s0 (длина дуги) = %.3f\n', z0, v0, s0);
% 
% 
% 
% % Вычисляем du/ds, dv/ds для начальной точки (производные по натуральному параметру ЛУ)
% % Для этого используем computeTangentCoeffs (из пакета InverseTask)
% [du0_s, dv0_s] = InverseTask.computeTangentCoeffs(E3, s0, v0, tau0_geo);
% fprintf('du/ds = %.6e, dv/ds = %.6e\n', du0_s, dv0_s);
% Начальные условия
s_0=2050
u0_geo = lu_on_E3.getPoint(s_0);
u0_geo = u0_geo(:);                     % столбец 3x1
tau0_geo = lu_on_E3.getTangent(s_0);
tau0_geo = tau0_geo(:);                 % столбец 3x1

% Параметры (s0, v0) для E3
z0 = u0_geo(3);
v0 = atan2(u0_geo(2), u0_geo(1));
s0 = E3.s_from_z(z0);

% Теперь вызов computeTangentCoeffs
[du0_s, dv0_s] = InverseTask.computeTangentCoeffs(E3, s0, v0, tau0_geo);

% Параметры сходимости (подобрать экспериментально, можно взять по умолчанию)
DeltaZ = 100;          % шаг по z для демпфирования
Percentage = 0;       % процент сходимости
z_span = [0, R_traj.totalLength()];   % от 0 до полной длины R(z)

%% 4. Вызов recoverLayer (интегратор обратной задачи)
% Функция recoverLayer ожидает:
%   surf – объект поверхности с методами getPoint, getNormal, getFirstFundamental, getSecondFundamental
%   R_func – function_handle: @(z) точка на R(z) (3×1)
%   Rprime_func – function_handle: @(z) касательная dR/dz (3×1)
%   z_span – [z0, z_end]
%   u0, v0 – начальные параметры на поверхности (s0, v0)
%   du0, dv0 – начальные du/ds, dv/ds
%   DeltaZ, Percentage – параметры демпфирования

% Создаём функции доступа к R(z)
R_func = @(z) R_traj.getPoint(z);
Rprime_func = @(z) R_traj.getTangent(z);

% Запуск восстановления
fprintf('Запуск recoverLayer (обратная задача)...\n');
tic;
[u_rec, v_rec, s_rec, z_rec] = InverseTask.recoverLayer(E3, R_func, Rprime_func, ...
    z_span, s0, v0, du0_s, dv0_s, DeltaZ, Percentage);
toc;

% u_rec, v_rec – параметры восстановленной ЛУ (длина дуги s и угол v)
% s_rec – накопленная длина дуги восстановленной ЛУ
% z_rec – значения параметра z, в которых вычислено решение

%% 5. Построение восстановленной ЛУ в декартовых координатах
N_rec = length(u_rec);
rec_pts = zeros(3, N_rec);
for i = 1:N_rec
    rec_pts(:,i) = E3.position(u_rec(i), v_rec(i));   % u_rec – длина дуги, v_rec – угол
end
rec_traj = InverseTask.Trajectory(rec_pts);
fprintf('Восстановленная линия укладки: длина дуги = %.3f мм\n', rec_traj.totalLength());

%% 6. Сравнение с исходной ЛУ (lu_on_E3)
% Интерполируем исходную ЛУ на те же значения s_rec (длины дуги восстановленной)
% Предварительно вычислим длины дуги для lu_on_E3
s_orig = linspace(0, lu_on_E3.totalLength(), 3000);
pts_orig = zeros(3, length(s_orig));
for i = 1:length(s_orig)
    pts_orig(:,i) = lu_on_E3.getPoint(s_orig(i));
end
orig_traj = InverseTask.Trajectory(pts_orig);  % пересоздадим для удобства

% Интерполяция исходной ЛУ на узлы s_rec
orig_pts_at_rec = zeros(3, N_rec);
for i = 1:N_rec
    orig_pts_at_rec(:,i) = orig_traj.getPoint(s_rec(i));
end

% Евклидова ошибка
err_xyz = sqrt(sum((rec_pts - orig_pts_at_rec).^2, 1));
fprintf('Средняя ошибка: %.2e мм, макс. ошибка: %.2e мм\n', mean(err_xyz), max(err_xyz));

%% 7. Визуализация
figure('Name', 'Обратная задача', 'Color', 'w', 'Position', [100 100 1200 800]);
hold on; grid on; axis equal; view(3);

% Каркас E3
z_vis = linspace(E3.z_min, E3.z_max, 40);
v_vis = linspace(0, 2*pi, 30);
for v0 = [0, pi/2, pi, 3*pi/2]
    pts = zeros(3, length(z_vis));
    for ii = 1:length(z_vis)
        pts(:,ii) = E3.position_by_z(z_vis(ii), v0);
    end
    plot3(pts(1,:), pts(2,:), pts(3,:), 'k-', 'LineWidth', 0.5);
end

% Исходная ЛУ на E3 (синяя)
s_plot = linspace(0, lu_on_E3.totalLength(), 200);
pts_orig_plot = zeros(3, length(s_plot));
for i = 1:length(s_plot)
    pts_orig_plot(:,i) = lu_on_E3.getPoint(s_plot(i));
end
plot3(pts_orig_plot(1,:), pts_orig_plot(2,:), pts_orig_plot(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Исходная ЛУ (проецированная)');

% Восстановленная ЛУ (красная пунктир)
plot3(rec_pts(1,:), rec_pts(2,:), rec_pts(3,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Восстановленная ЛУ');

% Траектория R(z) (зелёные точки)
R_plot = zeros(3, length(z_rec));
for i = 1:length(z_rec)
    R_plot(:,i) = R_func(z_rec(i));
end
scatter3(R_plot(1,:), R_plot(2,:), R_plot(3,:), 10, 'g', 'filled', 'DisplayName', 'R(z)');

xlabel('X, мм'); ylabel('Y, мм'); zlabel('Z, мм');
title('Обратная задача: восстановление линии укладки');
legend('Location', 'best');

% График ошибки
figure('Name', 'Ошибка восстановления');
semilogy(s_rec, err_xyz, 'r-', 'LineWidth', 1.5);
xlabel('Длина дуги s, мм'); ylabel('Евклидова ошибка, мм');
title('Ошибка восстановления линии укладки');
grid on;

fprintf('Готово.\n');