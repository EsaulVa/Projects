function vivod_3D_ballon_PB(u_min,u_max,func_surface,u_max_b,func_surface_b, N_steps)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Модель оправки - кусочно-полиномиальный баллон                       %
%    + Выводим оправку и поверхность безопасности                         %
%    Версия 4.0 от 1.04.2026 г.                                           %
%                                                                         %
%    Входные параметры:                                                   %
%        u_min          - начало отрисовки поверхности безопасности       %
%        u_max          - максимальная длина оправки                      %
%        func_surface   - функция модели оправки                          %
%        u_max          - максимальная длина поверхности безопасности     %
%        func_surface_b - функция модели поверхности безопасности         %
%        N_steps        - число точек по U и V                            %
%                                                                         %
%    Пример вызова:                                                       %
%      vivod_3D_ballon_PB(0,768.54,@surface_r,955.956, @surface_r_b,500)  %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Параметры для сетки отрисовки
            beas_u = (u_max_b - u_max)/2;
            N_steps_b = idivide(N_steps, int32(10));

            u = linspace(u_min, u_max, N_steps); % долгота
            v = linspace(0, 2*pi, N_steps);      % широта
            [U, V] = meshgrid(u, v);

            u_b = linspace(u_min, u_max_b, N_steps_b); % долгота
            v_b = linspace(0, 2*pi, N_steps_b);        % широта
            [U_b, V_b] = meshgrid(u_b, v_b);            
    
            % Преобразование в массив UV (с двумя длинными столбцами - U и V)
            UV = [U(:), V(:)]';
            UV_b = [U_b(:), V_b(:)]';
            
            % Получение декартовых координат поверхности и поверхности безопасности
            [pnt_surf,   ~, ~, ~, ~, ~] = func_surface(UV(1,:),UV(2,:));
            pnt_surf(3,:) = pnt_surf(3,:) + beas_u;
            [pnt_surf_b, ~, ~, ~, ~, ~] = func_surface_b(UV_b(1,:),UV_b(2,:));
            
            % Извлечение глобальных координат
            X = reshape(pnt_surf(1,:), N_steps, N_steps);
            Y = reshape(pnt_surf(2,:), N_steps, N_steps);
            Z = reshape(pnt_surf(3,:), N_steps, N_steps);
            Xb = reshape(pnt_surf_b(1,:), N_steps_b, N_steps_b);
            Yb = reshape(pnt_surf_b(2,:), N_steps_b, N_steps_b);
            Zb = reshape(pnt_surf_b(3,:), N_steps_b, N_steps_b);
    
            % Вывод поверхности на Fif
            % figure
            hold on
            
            s = surf(X, Y, Z, 'FaceAlpha', 0.3, 'EdgeAlpha', 0.05, 'EdgeColor', 'k', 'FaceColor', 'none');
            s_b = surf(Xb, Yb, Zb, 'FaceAlpha', 0.3, 'EdgeAlpha', 0.2, 'EdgeColor', 'k', 'FaceColor', 'none');

            material(s, 'shiny');
            material(s_b, 'shiny');

            xlabel('Z','FontSize',24);
            ylabel('Y','FontSize',24);
            zlabel('X','FontSize',24);
            
            % % Вывод точки (0,0)
            % [pnt, ~, ~, ~, ~, ~] = func_surface(u_min,pi/180);
            % plot3(pnt(1), pnt(2), pnt(3),'mo-', 'LineWidth', 2);
            
            grid on
            axis equal
            axis vis3d;        % фиксировать соотношение осей при вращении
            view(-35, 20);
end

