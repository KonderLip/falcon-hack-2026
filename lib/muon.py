import math

import torch

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor) -> torch.Tensor:
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    for a, b, c in polar_express_coeffs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def adjust_lr(lr: float, param_shape: torch.Size) -> float:
    A, B = param_shape[:2]
    adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    return lr * adjusted_ratio


class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 1e-3)
                group["momentum"] = group.get("momentum", 0.95)
            else:
                # defaults
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.999))
                group["eps"] = group.get("eps", 1e-8)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            if group["use_muon"]:
                params = group["params"]
                lr = group["lr"]
                momentum = group["momentum"]

                for p in params:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum)
                    u = polar_express(g)
                    adjusted_lr = adjust_lr(lr, p.shape)
                    p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

            ############################
            #        Adam backup       #
            ############################

            if not group["use_muon"]:
                params = group["params"]
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                for p in params:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["moment1"] = torch.zeros_like(g)
                        state["moment2"] = torch.zeros_like(g)
                    state["step"] += 1
                    step = state["step"]
                    buf1 = state["moment1"]
                    buf2 = state["moment2"]
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)
                    g = buf1 / (eps + buf2.sqrt())
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.add_(g, alpha=-lr / scale)

        return loss
