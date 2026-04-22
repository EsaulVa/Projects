% Очистка переменных, командной строки и закрытие окон
clear; clc; close all;

%% Шаг 1. Считывание исходных данных
    FileName = 'sector_points.txt';
    sector_points = readmatrix(FileName);

%% Шаг 2. Расчёт оптимального смещения по x для полярной системы координат
    xl = abs(floor(sector_points(end,1)) - ceil(sector_points(1,1)));
    dx_D = -xl:1:xl;
    L = length(dx_D);
    e_max = zeros(L, 1);

    for i = 1:1:length(dx_D)
        [~,~,~,~,~,~,~,~,~,~,e_max(i),~] = create_polynom(FileName, 5, dx_D(i));
    end

    [e_min, idxD] = min(e_max);  % минимальная погрешность и индекс в dx_D

    D = dx_D(idxD);              % лучшее смещение полюса по x (min ошибки)
    
    [RO_phi,ROp_phi,ROpp_phi,... % коэфф. интерп. полинома r(φ) и его производных
     b,db,d2b,...                % коэфф. апрокс. полинома φ(x) и его производных
     x_min,x_max,...             % границы диапазона изменения x (донышко)
     y_min,y_max,...             % границы диапазона изменения y (донышко)
     ~,~] = create_polynom(FileName, 5, D);