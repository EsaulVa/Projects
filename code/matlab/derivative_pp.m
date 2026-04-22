% Вспомогательная функция дифференцирования PP-формы
    function dpp = derivative_pp(pp)
        coefs = pp.coefs;
        breaks = pp.breaks;
        % Для ax^3 + bx^2 + cx + d -> 3ax^2 + 2bx + c
        dcoefs = coefs(:,1:3) .* [3 2 1]; 
        dpp = mkpp(breaks, dcoefs);
    end

