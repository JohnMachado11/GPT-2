import torch


class AdamW:
    def __init__(
        self,
        parameters,
        learning_rate=1e-3,
        momentum_decay=0.9,       # a.k.a. "beta1": how much old momentum to keep each step
        bumpiness_decay=0.999,    # a.k.a. "beta2": how much old bumpiness to keep each step
        epsilon=1e-8,             # tiny guard so we never divide by zero
        weight_decay=0.01,        # strength of the gentle pull toward zero
    ):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.bumpiness_decay = bumpiness_decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step_count = 0
        # one running-average per weight tensor, all starting at zeros
        self.momentum = [torch.zeros_like(p) for p in self.parameters]      # average of recent gradients
        self.bumpiness = [torch.zeros_like(p) for p in self.parameters]     # average of recent gradients squared

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad

            # 1. gentle pull toward zero (weight decay)
            parameter -= self.learning_rate * self.weight_decay * parameter

            # 2. update the running averages (momentum uses the gradient; bumpiness uses it SQUARED)
            self.momentum[i] = self.momentum_decay * self.momentum[i] + (1 - self.momentum_decay) * gradient
            self.bumpiness[i] = self.bumpiness_decay * self.bumpiness[i] + (1 - self.bumpiness_decay) * gradient ** 2

            # 3. bias-correct them (they started at zero, so early steps read to small)
            corrected_momentum = self.momentum[i] / (1 - self.momentum_decay ** self.step_count)
            corrected_bumpiness = self.bumpiness[i] / (1 - self.bumpiness_decay ** self.step_count)

            # 4. the adaptive step: big for clam directions, small for bumpy ones
            parameter -= self.learning_rate * corrected_momentum / (corrected_bumpiness.sqrt() + self.epsilon)