import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. Сеть (Conv1d residual, как в вашей статье)
# ============================================================

class WindingNet(nn.Module):
    """
    Вход:  (batch, 2, N) — каналы [u, v] на сетке z
    Выход: (batch, 2, N) — Y = X + delta
    """
    def __init__(self, hidden=64):
        super().__init__()
        # Параллельные ветви (как в статье: ядра 3 и 5)
        self.conv1 = nn.Conv1d(2, hidden, kernel_size=3,
                               padding='same', padding_mode='replicate')
        self.conv2 = nn.Conv1d(2, hidden, kernel_size=5,
                               padding='same', padding_mode='replicate')
        # Финальная свертка (ядро 7)
        self.conv3 = nn.Conv1d(hidden * 2, 2, kernel_size=7,
                               padding='same', padding_mode='replicate')
        self.act = nn.Tanh()

    def forward(self, x):
        # x: (batch, 2, N)
        o1 = self.act(self.conv1(x))
        o2 = self.act(self.conv2(x))
        o = torch.cat([o1, o2], dim=1)          # (batch, 2*hidden, N)
        delta = self.conv3(o)                   # (batch, 2, N)
        return x + delta                        # ResNet-связь


# ============================================================
# 2. Вычисление J[Y] через numpy (ваша геометрия)
# ============================================================

def compute_J_numpy(Y_np, surface, traj, z_eval, w_Phi=1.0, w_diff=1.0, w_smooth=0.05):
    """
    Y_np: (batch, N, 2) — физические u, v
    Возвращает средний loss по батчу (scalar).
    """
    batch, N, _ = Y_np.shape
    dz = z_eval[1] - z_eval[0] if N > 1 else 1.0
    total = 0.0

    for b in range(batch):
        u = Y_np[b, :, 0]
        v = Y_np[b, :, 1]

        # --- A. Связь Phi ---
        loss = 0.0
        for k in range(N):
            r = surface.position(u[k], v[k])
            m = surface.normal(u[k], v[k])
            Phi = np.dot(traj.R(z_eval[k]) - r, m)
            loss += w_Phi * Phi ** 2

        # --- B. Динамика ---
        for k in range(N - 1):
            du = (u[k + 1] - u[k]) / dz
            dv = (v[k + 1] - v[k]) / dz
            loss += w_diff * (du ** 2 + dv ** 2)

        # --- C. Гладкость ---
        for k in range(N - 1):
            E, F, G = surface.first_fundamental_form(u[k], v[k])
            du = u[k + 1] - u[k]
            dv = v[k + 1] - v[k]
            ds_sq = max(E * du ** 2 + 2 * F * du * dv + G * dv ** 2, 0.0)
            loss += w_smooth * ds_sq

        total += loss

    return total / batch


# ============================================================
# 3. SPSA-обертка для PyTorch autograd
# ============================================================

class JFunction(torch.autograd.Function):
    """
    Forward:  вычисляет J(Y) через numpy.
    Backward: оценивает dJ/dY методом SPSA (O(1) по размерности).
    """
    @staticmethod
    def forward(ctx, Y_torch, compute_J_fn, delta=1e-3, n_samples=2):
        ctx.compute_J_fn = compute_J_fn
        ctx.delta = delta
        ctx.n_samples = n_samples
        Y_np = Y_torch.detach().cpu().numpy()
        J_val = compute_J_fn(Y_np)
        ctx.save_for_backward(Y_torch)
        return torch.tensor(J_val, dtype=Y_torch.dtype, device=Y_torch.device)

    @staticmethod
    def backward(ctx, grad_output):
        Y_torch, = ctx.saved_tensors
        Y_np = Y_torch.detach().cpu().numpy()
        delta = ctx.delta
        n_samples = ctx.n_samples

        grad_est = torch.zeros_like(Y_torch)
        for _ in range(n_samples):
            # Случайное направление (антитетическая выборка)
            pert = torch.randn_like(Y_torch)
            pert_np = pert.cpu().numpy()

            J_plus = ctx.compute_J_fn(Y_np + delta * pert_np)
            J_minus = ctx.compute_J_fn(Y_np - delta * pert_np)

            # SPSA: градиент = (J+ - J-) / (2*delta) * perturbation
            grad_est += (J_plus - J_minus) / (2.0 * delta) * pert

        grad_est = grad_est / n_samples
        return grad_est * grad_output, None, None, None


class JLoss(nn.Module):
    def __init__(self, surface, traj, z_eval,
                 w_Phi=1.0, w_diff=1.0, w_smooth=0.05,
                 spsa_delta=1e-3, spsa_samples=2):
        super().__init__()
        self.z_eval = z_eval
        self.w_Phi = w_Phi
        self.w_diff = w_diff
        self.w_smooth = w_smooth
        self.spsa_delta = spsa_delta
        self.spsa_samples = spsa_samples

        # Closure для передачи в JFunction
        self._compute_J = lambda Y_np: compute_J_numpy(
            Y_np, surface, traj, z_eval,
            w_Phi=w_Phi, w_diff=w_diff, w_smooth=w_smooth
        )

    def forward(self, Y_torch):
        return JFunction.apply(Y_torch, self._compute_J,
                                self.spsa_delta, self.spsa_samples)


# ============================================================
# 4. Генерация случайных начальных приближений
# ============================================================

def generate_batch(base_u, base_v, batch_size, noise_u=10.0, noise_v=0.1):
    """
    base_u, base_v: (N,) — эталонная или radial-init траектория
    Возвращает X: (batch, N, 2) тензор
    """
    N = len(base_u)
    X = np.zeros((batch_size, N, 2))
    for b in range(batch_size):
        X[b, :, 0] = base_u + np.random.normal(0, noise_u, N)
        X[b, :, 1] = base_v + np.random.normal(0, noise_v, N)
    # unwrap v для непрерывности
    X[:, :, 1] = np.unwrap(X[:, :, 1], axis=1)
    return torch.tensor(X, dtype=torch.float32)


# ============================================================
# 5. Обучение
# ============================================================

def train(net, jloss, base_u, base_v,
          epochs=100, batch_size=16, lr=1e-3,
          noise_u=10.0, noise_v=0.1,
          verbose=True):
    """
    net:   WindingNet
    jloss: JLoss
    base_u, base_v: (N,) — центр облака начальных приближений
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epochs):
        net.train()
        X = generate_batch(base_u, base_v, batch_size, noise_u, noise_v)

        # Перестановка осей для Conv1d: (batch, N, 2) -> (batch, 2, N)
        X = X.permute(0, 2, 1)

        optimizer.zero_grad()
        Y = net(X)                      # (batch, 2, N)

        # Функция потерь: J[Y] + проксимальный член ||Y - X||^2
        loss_main = jloss(Y)
        loss_prox = 0.01 * ((Y - X.detach()) ** 2).mean()
        loss = loss_main + loss_prox

        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                J_val = jloss(Y).item()
                print(f"Epoch {epoch:4d} | J[Y]={J_val:.3e} | "
                      f"loss={loss.item():.3e} | lr={scheduler.get_last_lr()[0]:.2e}")

    return net