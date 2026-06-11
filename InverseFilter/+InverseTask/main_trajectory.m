% Создаём несколько точек на кривой (например, винтовая линия)
t = linspace(0, 4*pi, 50);
x = cos(t);
y = sin(t);
z = t / 2;
points = [x; y; z];

% Строим траекторию с натуральной параметризацией
traj = Trajectory(points);

% Проверка: берём параметр s = половина длины
L = traj.totalLength();
s_mid = L/2;
r_mid = traj.getPoint(s_mid);
tau_mid = traj.getTangent(s_mid);
fprintf('Точка в середине: [%.3f, %.3f, %.3f]\n', r_mid);
fprintf('Касательный вектор (норма = %.6f)\n', norm(tau_mid));

% Визуализация (выборочные точки)
s_plot = linspace(0, L, 200);
xyz = zeros(3, length(s_plot));
for i = 1:length(s_plot)
    xyz(:,i) = traj.getPoint(s_plot(i));
end
plot3(xyz(1,:), xyz(2,:), xyz(3,:), 'b-', 'LineWidth', 1.5);
hold on;
plot3(points(1,:), points(2,:), points(3,:), 'ro');
axis equal; grid on;