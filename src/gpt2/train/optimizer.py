import torch


class AdamW:
    def __init__(
        self,
        parameters,
        learning_rate=1e-3,
        momentum_decay=0.9,       # a.k.a. "beta1": how much old momentum to keep each step
        bumpiness_decay=0.999,    # a.k.a. "beta2": how much old bumpiness to keep each step
        epsilon=1e-8,             # tiny guard so we never divide by zero
        weight_decay=0.1,         # pull toward zero; 0.1 from the GPT-3 paper + nanoGPT's GPT-2 retrain
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
        # Which tensors actually get weight decay. Standard GPT practice (nanoGPT does
        # this) is to decay ONLY the 2-D weight matrices (embeddings, attention and
        # feed-forward weights). Biases and LayerNorm scales are 1-D -- pulling those
        # toward zero hurts, so they're left alone. True = decay it, False = skip it.
        self.decayed = [parameter.dim() >= 2 for parameter in self.parameters]

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

            # 1. gentle pull toward zero (weight decay) -- 2-D weights only
            if self.decayed[i]:
                parameter -= self.learning_rate * self.weight_decay * parameter

            # 2. update the running averages (momentum uses the gradient; bumpiness uses it SQUARED)
            self.momentum[i] = self.momentum_decay * self.momentum[i] + (1 - self.momentum_decay) * gradient
            self.bumpiness[i] = self.bumpiness_decay * self.bumpiness[i] + (1 - self.bumpiness_decay) * gradient ** 2

            # 3. bias-correct them (they started at zero, so early steps read too small)
            corrected_momentum = self.momentum[i] / (1 - self.momentum_decay ** self.step_count)
            corrected_bumpiness = self.bumpiness[i] / (1 - self.bumpiness_decay ** self.step_count)

            # 4. the adaptive step: big for calm directions, small for bumpy ones
            parameter -= self.learning_rate * corrected_momentum / (corrected_bumpiness.sqrt() + self.epsilon)

    def hyperparameters(self):
        # the settings that decide HOW this optimizer moves the weights. (The learning rate isn't one:
        # the schedule sets it fresh every step.)
        # (plain Python numbers: a numpy number would save fine and then fail to load at resume time)
        return {
            "momentum_decay": float(self.momentum_decay),
            "bumpiness_decay": float(self.bumpiness_decay),
            "epsilon": float(self.epsilon),
            "weight_decay": float(self.weight_decay),
        }

    def state_dict(self):
        # everything AdamW has accumulated so far -- a checkpoint saves this, so a resumed
        # run carries on with the same momentum instead of restarting cold
        return {
            "step_count": self.step_count,
            "momentum": list(self.momentum),
            "bumpiness": list(self.bumpiness),
            "hyperparameters": self.hyperparameters(),
        }

    def check_state_dict(self, state):
        # raise if this saved state can't be loaded into THIS optimizer -- without changing anything
        if len(state["momentum"]) != len(self.parameters):
            raise ValueError(
                f"optimizer state has {len(state['momentum'])} tensors, but this model has "
                f"{len(self.parameters)} -- was it saved from a different model?"
            )
        if len(state["bumpiness"]) != len(self.parameters):
            raise ValueError(f"optimizer state has {len(state['bumpiness'])} bumpiness tensors, but this model "
                             f"has {len(self.parameters)} -- the saved state is damaged or from a different model")
        for saved_momentum, saved_bumpiness, parameter in zip(state["momentum"], state["bumpiness"], self.parameters):
            for saved in (saved_momentum, saved_bumpiness):
                if saved.shape != parameter.shape:
                    raise ValueError(f"optimizer state holds a {tuple(saved.shape)} tensor where this model has "
                                     f"{tuple(parameter.shape)} -- was it saved from a different model?")
        saved_hyperparameters = state.get("hyperparameters")      # missing in checkpoints from before this check
        if saved_hyperparameters is not None and saved_hyperparameters != self.hyperparameters():
            raise ValueError(f"optimizer was saved with {saved_hyperparameters}, but this one has "
                             f"{self.hyperparameters()} -- resuming would quietly change how the weights move")

    def load_state_dict(self, state):
        self.check_state_dict(state)
        self.step_count = state["step_count"]
        # move each saved tensor onto the same device as its weight (a run saved on a GPU can resume anywhere)
        self.momentum = [m.to(p.device) for m, p in zip(state["momentum"], self.parameters)]
        self.bumpiness = [b.to(p.device) for b, p in zip(state["bumpiness"], self.parameters)]