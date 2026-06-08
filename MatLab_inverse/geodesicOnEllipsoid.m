function [u_geo, v_geo, s_geo] = geodesicOnEllipsoid(u0, v0, azimuth0, s_end, a, b, c)
    % u0,v0 - начальная точка, azimuth0 - начальный азимут (угол от меридиана)
    % s_end - длина кривой, возвращаем точки (u(s), v(s), натуральный параметр s)
    % Используем стандартные уравнения геодезических:
    % du/ds = p, dv/ds = q,
    % dp/ds = -Γ^u_uu p^2 - 2Γ^u_uv p q - Γ^u_vv q^2,
    % dq/ds = -Γ^v_uu p^2 - 2Γ^v_uv p q - Γ^v_vv q^2.
    
    % Сначала вычислим начальные p0, q0 из условия, что касательный вектор
    % в пространстве: dr/ds = ru * p + rv * q, его длина = 1, и угол azimuth
    % между ru и dr/ds известен. azimuth = angle between dr/ds and ru.
    % В эллипсоидальных координатах ru = ∂r/∂u, rv = ∂r/∂v.
    [~, ru0, rv0] = ellipsoidSurface(u0, v0, a, b, c);
    E0 = dot(ru0, ru0); F0 = dot(ru0, rv0); G0 = dot(rv0, rv0);
    % Начальный вектор в касательной плоскости: (p,q) такие, что
    % E p^2 + 2F p q + G q^2 = 1  и  cos(azimuth) = (E p + F q)/sqrt(E) ??? Упростим:
    % Сначала найдём единичный вектор в касательной плоскости, направленный
    % под углом azimuth относительно ru. Базис ru, rv не ортогонален.
    % Используем факт, что метрика задана. Ортогонализируем ru и перпендикулярную компоненту.
    e1 = ru0 / sqrt(E0);
    e2 = (rv0 - F0/sqrt(E0) * e1) / sqrt(G0 - F0^2/E0);
    % Тогда касательный вектор t = cos(azimuth)*e1 + sin(azimuth)*e2.
    t = cos(azimuth0)*e1 + sin(azimuth0)*e2;
    % Разложение t по ru, rv: t = A*ru + B*rv.
    % Решаем систему: [E F; F G][A; B] = [dot(t,ru); dot(t,rv)]
    M = [E0, F0; F0, G0];
    rhs = [dot(t, ru0); dot(t, rv0)];
    AB = M \ rhs;
    p0 = AB(1); q0 = AB(2);
    
    % Интегрируем ОДУ от s=0 до s_end (метод Рунге-Кутты)
    h = s_end / 1000;
    s = 0:h:s_end;
    N = length(s);
    u = zeros(1,N); v = zeros(1,N); p = zeros(1,N); q = zeros(1,N);
    u(1) = u0; v(1) = v0; p(1) = p0; q(1) = q0;
    
    for i = 1:N-1
        % Вычисляем производные в текущей точке
        [du_ds, dv_ds, dp_ds, dq_ds] = geodesicRHS(u(i), v(i), p(i), q(i), a, b, c);
        % RK4
        k1_u = du_ds; k1_v = dv_ds; k1_p = dp_ds; k1_q = dq_ds;
        
        u_mid = u(i) + h/2 * k1_u; v_mid = v(i) + h/2 * k1_v;
        p_mid = p(i) + h/2 * k1_p; q_mid = q(i) + h/2 * k1_q;
        [du_ds, dv_ds, dp_ds, dq_ds] = geodesicRHS(u_mid, v_mid, p_mid, q_mid, a, b, c);
        k2_u = du_ds; k2_v = dv_ds; k2_p = dp_ds; k2_q = dq_ds;
        
        u_mid = u(i) + h/2 * k2_u; v_mid = v(i) + h/2 * k2_v;
        p_mid = p(i) + h/2 * k2_p; q_mid = q(i) + h/2 * k2_q;
        [du_ds, dv_ds, dp_ds, dq_ds] = geodesicRHS(u_mid, v_mid, p_mid, q_mid, a, b, c);
        k3_u = du_ds; k3_v = dv_ds; k3_p = dp_ds; k3_q = dq_ds;
        
        u_next = u(i) + h * k3_u; v_next = v(i) + h * k3_v;
        p_next = p(i) + h * k3_p; q_next = q(i) + h * k3_q;
        [du_ds, dv_ds, dp_ds, dq_ds] = geodesicRHS(u_next, v_next, p_next, q_next, a, b, c);
        k4_u = du_ds; k4_v = dv_ds; k4_p = dp_ds; k4_q = dq_ds;
        
        u(i+1) = u(i) + h/6*(k1_u + 2*k2_u + 2*k3_u + k4_u);
        v(i+1) = v(i) + h/6*(k1_v + 2*k2_v + 2*k3_v + k4_v);
        p(i+1) = p(i) + h/6*(k1_p + 2*k2_p + 2*k3_p + k4_p);
        q(i+1) = q(i) + h/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
    end
    u_geo = u; v_geo = v; s_geo = s;
end

function [du_ds, dv_ds, dp_ds, dq_ds] = geodesicRHS(u, v, p, q, a, b, c)
    % Вычисляет правые части уравнений геодезических
    [~, ru, rv, ruu, ruv, rvv] = ellipsoidSurface(u, v, a, b, c);
    n = computeNormal(ru, rv);
    [E, F, G] = firstFundamentalForm(ru, rv);
    [L, M, N] = secondFundamentalForm(ru, rv, ruu, ruv, rvv, n);
    % Символы Кристоффеля
    det = E*G - F^2;
    %Gu = (G - F)/? Нет, нужны производные E, F, G по u,v.
    % Лучше вычислить символы через частные производные ruu, ruv, rvv.
    % Γ^u_uu = (G*ruu·ru - 2F*ruu·rv + E*ruu·? )/det? 
    % Вместо этого используем стандартные формулы:
    % Γ^1_11 = (G*E_u - 2F*F_u + F*E_v) / (2det)
    % и т.д. Упростим, считая, что u=долгота, v=широта, но эллипсоид не сфера,
    % поэтому символы не нулевые. Реализуем через производные метрики.
    % Сначала вычислим производные ru, rv по u и v.
    % Для эллипсоида частные производные E,F,G по u,v можно получить аналитически.
    % Но для общности, аппроксимируем численно:
    eps = 1e-6;
    [~, ru_du, rv_du] = ellipsoidSurface(u+eps, v, a, b, c);
    [E_du, F_du, G_du] = firstFundamentalForm(ru_du, rv_du);
    [E_u, F_u, G_u] = firstFundamentalForm(ru, rv);
    E_u = (E_du - E)/eps; F_u = (F_du - F)/eps; G_u = (G_du - G)/eps;
    
    [~, ru_dv, rv_dv] = ellipsoidSurface(u, v+eps, a, b, c);
    [E_dv, F_dv, G_dv] = firstFundamentalForm(ru_dv, rv_dv);
    E_v = (E_dv - E)/eps; F_v = (F_dv - F)/eps; G_v = (G_dv - G)/eps;
    
    det = E*G - F^2;
    % Γ^u_uu = (G*E_u - 2F*F_u + F*E_v) / (2det)
    Gammau_uu = (G*E_u - 2*F*F_u + F*E_v) / (2*det);
    Gammau_uv = (G*E_v - F*G_u) / (2*det);
    Gammau_vv = (2*G*F_v - G*G_u - F*G_v) / (2*det);
    Gammav_uu = (2*E*F_u - E*E_v - F*E_u) / (2*det);
    Gammav_uv = (E*G_u - F*E_v) / (2*det);
    Gammav_vv = (E*G_v - 2*F*F_v + F*G_u) / (2*det);
    
    du_ds = p;
    dv_ds = q;
    dp_ds = -Gammau_uu * p^2 - 2*Gammau_uv * p*q - Gammau_vv * q^2;
    dq_ds = -Gammav_uu * p^2 - 2*Gammav_uv * p*q - Gammav_vv * q^2;
end