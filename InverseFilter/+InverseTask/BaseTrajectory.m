classdef BaseTrajectory < handle
    % Абстрактный класс кривой, параметризованной натуральным параметром s
    methods (Abstract)
        r = getPoint(obj, s)       % точка на кривой
        tau = getTangent(obj, s)   % единичный касательный вектор
        L = totalLength(obj)       % полная длина кривой
    end
end

