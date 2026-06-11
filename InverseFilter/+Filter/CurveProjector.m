classdef CurveProjector < handle
    %CURVEPROJECTOR Проектирование пространственной кривой (Trajectory) на поверхность вращения.
    %   Использует поточечную проекцию каждой опорной точки кривой с последующей
    %   интерполяцией сплайном и построением новой Trajectory.
    %
    %   Пример:
    %       proj = Filter.CurveProjector(E3);
    %       new_traj = proj.project(lu_traj);
    %       % new_traj лежит на E3

    properties (Access = private)
        point_projector  % экземпляр PointProjector
    end

    methods
        function obj = CurveProjector(surf)
            % Конструктор
            obj.point_projector = Filter.PointProjector(surf);
        end

        function new_traj = project(obj, traj)
            % Проектирует траекторию (объект класса InverseTask.Trajectory) на поверхность.
            % Возвращает новую Trajectory, точки которой лежат на поверхности.
            % Вход: traj - объект с методами getPoint(s) и totalLength() (как у InverseTask.Trajectory)
            % Выход: new_traj - объект InverseTask.Trajectory (или Trajectory, если доступен)
            
            % Получаем опорные точки исходной траектории (можно взять её узловые точки)
            % Для точности лучше взять исходные точки, по которым строился сплайн.
            % В Trajectory есть метод getSValues()? Если нет, сделаем равномерную выборку.
            
            if ismethod(traj, 'getSValues')
                s_vals = traj.getSValues();  % массив длин дуг в узлах
                N = length(s_vals);
                points_orig = zeros(3, N);
                for i = 1:N
                    points_orig(:, i) = traj.getPoint(s_vals(i));
                end
            else
                % Если нет узлов, берём равномерную выборку
                L = traj.totalLength();
                N = max(100, round(L / 1.0)); % шаг ~1 мм
                s_vals = linspace(0, L, N);
                points_orig = zeros(3, N);
                for i = 1:N
                    points_orig(:, i) = traj.getPoint(s_vals(i));
                end
            end
            
            % Проецируем каждую точку
            points_proj = zeros(3, size(points_orig, 2));
            for i = 1:size(points_orig, 2)
                q = obj.point_projector.project(points_orig(:, i)');
                points_proj(:, i) = q(:);
            end
            
            % Строим новую траекторию (используем доступный класс Trajectory)
            % Предполагаем, что InverseTask.Trajectory доступен и принимает 3×N матрицу
            new_traj = InverseTask.Trajectory(points_proj);
        end
        
        function points_proj = project_points(obj, points_xyz)
            % Проектирование набора точек (матрица 3×N) без построения Trajectory.
            points_xyz = double(points_xyz);
            if size(points_xyz,1) ~= 3
                points_xyz = points_xyz';
            end
            N = size(points_xyz,2);
            points_proj = zeros(3, N);
            for i = 1:N
                q = obj.point_projector.project(points_xyz(:, i)');
                points_proj(:, i) = q(:);
            end
        end
    end
end