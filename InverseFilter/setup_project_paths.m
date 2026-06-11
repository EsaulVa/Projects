function setup_project_paths()
    % Добавляет корневую папку проекта в путь MATLAB.
    % Проверяет наличие пакетов +Filter и +InverseTask.

    root = fileparts(mfilename('fullpath'));
    filter_pkg = fullfile(root, '+Filter');
    invtask_pkg = fullfile(root, '+InverseTask');

    if ~exist(filter_pkg, 'dir')
        error('Папка +Filter не найдена в %s', root);
    end
    if ~exist(invtask_pkg, 'dir')
        error('Папка +InverseTask не найдена в %s', root);
    end

    if ~contains(path, root)
        addpath(root);
        fprintf('Добавлен путь: %s\n', root);
    else
        fprintf('Путь уже присутствует: %s\n', root);
    end

    % Проверка наличия ключевых файлов классов
    bc_file = fullfile(filter_pkg, 'BoundaryCondition.m');
    if exist(bc_file, 'file')
        fprintf('Найден Filter.BoundaryCondition\n');
    else
        error('Файл BoundaryCondition.m отсутствует в +Filter');
    end

    % Дополнительно: проверить наличие DiffusedRevolutionSurface (опционально)
    drs_file = fullfile(filter_pkg, 'DiffusedRevolutionSurface.m');
    if exist(drs_file, 'file')
        fprintf('Найден Filter.DiffusedRevolutionSurface\n');
    else
        error('Файл DiffusedRevolutionSurface.m отсутствует в +Filter');
    end

    fprintf('Пакеты Filter и InverseTask готовы к использованию.\n');
end