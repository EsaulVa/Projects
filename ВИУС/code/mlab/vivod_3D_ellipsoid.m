%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Скрипт вывода эллипсоитда вращения на 3D график                      %
%    08.04.2025                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            N_steps = 50; % Число по U и V
            u = linspace(0, 2*pi, N_steps);     % долгота
            v = linspace(-pi/2, pi/2, N_steps); % широта
            [U, V] = meshgrid(u, v);
    
            % Преобразование в массив UV (с двумя длинными столбцами - U и V)
            UV = [U(:), V(:)]';
            
            % Получение декартовых координат поверхности
            pnt_surf = surface_e(UV(1,:),UV(2,:));
            
            % Параметрические формулы
            X = reshape(pnt_surf(1,:), size(U));
            Y = reshape(pnt_surf(2,:), size(U));
            Z = reshape(pnt_surf(3,:), size(U));

            % Преобразование в однородные координаты
            P_local = [X(:), Y(:), Z(:)]';  % 3xN

            X_gl = reshape(P_local(1,:), N_steps, N_steps);
            Y_gl = reshape(P_local(2,:), N_steps, N_steps);
            Z_gl = reshape(P_local(3,:), N_steps, N_steps);
    
            % Вывод поверхности эллипсоида на Fif
            surf(X_gl, Y_gl, Z_gl, 'FaceAlpha', 0.3, 'EdgeAlpha', 0.1);

            grid on
            axis equal
            axis vis3d;        % фиксировать соотношение осей при вращении
