import numpy as np
def make_uv_bounds_events(u_min=-np.inf, u_max=np.inf, v_min=-np.inf, v_max=np.inf):
    events = []
    
    if np.isfinite(u_max):
        def u_limit(t, y):
            return u_max - y[0]  # y[0] = u
        u_limit.terminal = True
        u_limit.direction = -1  # срабатывает при пересечении сверху вниз
        events.append(u_limit)
    
    if np.isfinite(u_min):
        def u_min_limit(t, y):
            return y[0] - u_min
        u_min_limit.terminal = True
        u_min_limit.direction = 1
        events.append(u_min_limit)
    
    # Аналогично для v (y[2] в прямой задаче)
    if np.isfinite(v_max):
        def v_limit(t, y):
            return v_max - y[2]
        v_limit.terminal = True
        v_limit.direction = -1
        events.append(v_limit)
    
    if np.isfinite(v_min):
        def v_min_limit(t, y):
            return y[2] - v_min
        v_min_limit.terminal = True
        v_min_limit.direction = 1
        events.append(v_min_limit)
    
    return events