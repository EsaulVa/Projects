%% Îñíîâíîé ñêğèïò äëÿ òåñòèğîâàíèÿ ïàğàìåòğè÷åñêèõ ñïëàéíîâ

clear all; close all; clc;

%% Ãåíåğàöèÿ äàííûõ (çàøóìëåííàÿ ïîëóîêğóæíîñòü)
rng(10);
n_points = 30;
r = 5.0;
theta = linspace(0, pi, n_points)';

% Èñòèííàÿ îêğóæíîñòü
clean_x = r * cos(theta);
clean_y = r * sin(theta);

% Çàøóìëåííûå äàííûå
noise_std = 0.001;
points_noisy = [clean_x + noise_std*randn(n_points, 1), ...
                clean_y + noise_std*randn(n_points, 1)];

%% Ãğàíè÷íûå óñëîâèÿ
bc_start = struct('direction', [0.0, 1.0], 'curvature', 1.0/r);
bc_end = struct('direction', [0.0, -1.0], 'curvature', 1.0/r);

%% Ñîçäàíèå è îáó÷åíèå ñïëàéíîâ
spline = ParametricQuinticSpline(points_noisy, bc_start, bc_end);

% Òåñòèğóåì ğàçíûå óğîâíè ñãëàæèâàíèÿ
alphas = [0.99, 0.95, 0.9];
results = cell(length(alphas), 1);

for i = 1:length(alphas)
    fprintf('\n=== alpha = %.3f ===\n', alphas(i));
    spline_temp = ParametricQuinticSpline(points_noisy, bc_start, bc_end);
    spline_temp.fit(alphas(i));
    results{i} = spline_temp;
end

% Îñíîâíîé ñïëàéí
spline.fit(0.95);

%% Äàííûå äëÿ ãğàôèêîâ
u_dense = linspace(0, spline.u(end), 300)';
curve_pts = spline.predict(u_dense, 0);
tangents = spline.predict(u_dense, 1);
second_derivs = spline.predict(u_dense, 2);
curvature_vals = spline.curvature(u_dense);

%% Âèçóàëèçàöèÿ
figure('Position', [100, 100, 1400, 900]);

% 1. Êğèâàÿ â ïğîñòğàíñòâå
subplot(3, 2, 1);
plot(points_noisy(:, 1), points_noisy(:, 2), 'ro', ...
    'MarkerSize', 6, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', ...
    'DisplayName', 'Çàøóìëåííûå óçëû');
hold on;
plot(clean_x, clean_y, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Èñòèííàÿ îêğóæíîñòü');

% Ğèñóåì ğàçíûå óğîâíè ñãëàæèâàíèÿ
colors = ['b', 'g', 'm'];
for i = 1:length(alphas)
    curve_temp = results{i}.predict(u_dense, 0);
    plot(curve_temp(:, 1), curve_temp(:, 2), colors(i), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('?=%.2f', alphas(i)));
end

title('Ïàğàìåòğè÷åñêàÿ êğèâàÿ (XY ïëîñêîñòü)');
axis equal; grid on; legend('Location', 'best');

% 2. Êàñàòåëüíûå âåêòîğû (ôğàãìåíò)
subplot(3, 2, 2);
mask = (u_dense > spline.u(end)*0.4) & (u_dense < spline.u(end)*0.6);
u_segment = u_dense(mask);
curve_segment = curve_pts(mask, :);
tangents_segment = tangents(mask, :);

plot(curve_segment(:, 1), curve_segment(:, 2), 'b-', 'LineWidth', 2);
hold on;

step = max(1, floor(length(u_segment)/10));
quiver(curve_segment(1:step:end, 1), curve_segment(1:step:end, 2), ...
       tangents_segment(1:step:end, 1), tangents_segment(1:step:end, 2), ...
       0.5, 'g', 'LineWidth', 1, 'MaxHeadSize', 0.5);

title('Êàñàòåëüíûå âåêòîğû (ôğàãìåíò)');
axis equal; grid on;

% 3. Ïåğâàÿ ïğîèçâîäíàÿ
subplot(3, 2, 3);
plot(u_dense, tangents(:, 1), 'b-', 'LineWidth', 1.5, 'DisplayName', '$$\dot{x}(u)$$');
hold on;
plot(u_dense, tangents(:, 2), 'r-', 'LineWidth', 1.5, 'DisplayName', '$$\dot{y}(u)$$');
yline(0, 'k--', 'Alpha', 0.3);
title('Ïåğâàÿ ïğîèçâîäíàÿ (íîğìàëèçîâàííûå êîìïîíåíòû)');
xlabel('Ïàğàìåòğ u'); ylabel('Çíà÷åíèå');
legend('Interpreter', 'latex'); grid on;
ylim([-1.5, 1.5]);

% 4. Âòîğàÿ ïğîèçâîäíàÿ
subplot(3, 2, 4);
plot(u_dense, second_derivs(:, 1), 'b-', 'LineWidth', 1.5, 'DisplayName', '$$\ddot{x}(u)$$');
hold on;
plot(u_dense, second_derivs(:, 2), 'r-', 'LineWidth', 1.5, 'DisplayName', '$$\ddot{y}(u)$$');
yline(0, 'k--', 'Alpha', 0.3);
title('Âòîğàÿ ïğîèçâîäíàÿ (êîìïîíåíòû)');
xlabel('Ïàğàìåòğ u'); ylabel('Çíà÷åíèå');
legend('Interpreter', 'latex'); grid on;

% 5. Êğèâèçíà
subplot(3, 2, 5);
plot(u_dense, curvature_vals, 'b-', 'LineWidth', 2, 'DisplayName', 'Êğèâèçíà ñïëàéíà');
hold on;
yline(1/r, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('Èñòèííàÿ êğèâèçíà (1/r=%.3f)', 1/r));
title('Êğèâèçíà êğèâîé');
xlabel('Ïàğàìåòğ u'); ylabel('?');
legend('Location', 'best'); grid on;

% 6. Îòêëîíåíèå îò èñòèííîé êğèâîé
subplot(3, 2, 6);
theta_interp = interp1(spline.u, theta, u_dense);
true_curve = [r * cos(theta_interp), r * sin(theta_interp)];
deviation = sqrt(sum((curve_pts - true_curve).^2, 2));

plot(u_dense, deviation, 'r-', 'LineWidth', 2);
hold on;
fill([u_dense; flipud(u_dense)], [zeros(size(deviation)); flipud(deviation)], ...
    'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

title('Îòêëîíåíèå îò èñòèííîé êğèâîé');
xlabel('Ïàğàìåòğ u'); ylabel('Îòêëîíåíèå');
grid on;

%% Âûâîä èíôîğìàöèè
fprintf('\n=== Ğåçóëüòàòû ===\n');
fprintf('Äëèíà êğèâîé: %.3f\n', spline.length());
fprintf('Èñòèííàÿ äëèíà (ïîëóîêğóæíîñòü): %.3f\n', pi * r);
fprintf('Ñğåäíÿÿ êğèâèçíà: %.3f\n', mean(abs(curvature_vals(~isnan(curvature_vals)))));
fprintf('Îæèäàåìàÿ êğèâèçíà: %.3f\n', 1/r);

fprintf('\n=== Ñğàâíåíèå ğàçíûõ alpha ===\n');
for i = 1:length(alphas)
    curve_temp = results{i}.predict(u_dense, 0);
    deviation_temp = mean(sqrt(sum((curve_temp - true_curve).^2, 2)));
    curvature_temp = results{i}.curvature(u_dense);
    curvature_var = var(curvature_temp(~isnan(curvature_temp)));
    
    fprintf('?=%.2f: ñğåäíåå îòêëîíåíèå=%.4f, âàğèàöèÿ êğèâèçíû=%.6f\n', ...
        alphas(i), deviation_temp, curvature_var);
end