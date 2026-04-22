% Загружаем s, r и tau
load('LU_data.mat')
% Зпускаем процедуру расчёта ТСН
TSN = create_TSN_C(r, tau);