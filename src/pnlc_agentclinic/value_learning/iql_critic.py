from copy import deepcopy

import torch
from torch import nn


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared loss used to fit IQL's value function."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        output_dim: int = 1,
    ):
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend((nn.Linear(in_dim, out_dim), nn.ReLU()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class GoalConditionedIQLCritic(nn.Module):
    """Q(s, thought, goal) and V(s, goal) networks for offline IQL."""

    def __init__(
        self,
        state_dim: int,
        thought_dim: int,
        hidden_dims: tuple[int, ...] = (512, 512),
        normalize_inputs: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.thought_dim = thought_dim
        self.hidden_dims = tuple(hidden_dims)
        self.normalize_inputs = normalize_inputs

        self.q_network = MLP(
            input_dim=(2 * state_dim) + thought_dim,
            hidden_dims=self.hidden_dims,
        )
        self.value_network = MLP(
            input_dim=2 * state_dim,
            hidden_dims=self.hidden_dims,
        )
        self.target_value_network = deepcopy(self.value_network)
        self.target_value_network.requires_grad_(False)

        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("thought_mean", torch.zeros(thought_dim))
        self.register_buffer("thought_std", torch.ones(thought_dim))

    @torch.no_grad()
    def set_normalization_stats(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        thought_mean: torch.Tensor,
        thought_std: torch.Tensor,
    ) -> None:
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std.clamp_min(1e-6))
        self.thought_mean.copy_(thought_mean)
        self.thought_std.copy_(thought_std.clamp_min(1e-6))

    def _normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return value
        return (value - self.state_mean) / self.state_std

    def _normalize_thought(self, value: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return value
        return (value - self.thought_mean) / self.thought_std

    def q(
        self,
        state: torch.Tensor,
        thought: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (
                self._normalize_state(state),
                self._normalize_thought(thought),
                self._normalize_state(goal),
            ),
            dim=-1,
        )
        return self.q_network(inputs).squeeze(-1)

    def value(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat(
            (self._normalize_state(state), self._normalize_state(goal)),
            dim=-1,
        )
        return self.value_network(inputs).squeeze(-1)

    def target_value(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat(
            (self._normalize_state(state), self._normalize_state(goal)),
            dim=-1,
        )
        return self.target_value_network(inputs).squeeze(-1)

    @torch.no_grad()
    def update_target(self, rate: float) -> None:
        for target, online in zip(
            self.target_value_network.parameters(),
            self.value_network.parameters(),
        ):
            target.lerp_(online, rate)

    @torch.no_grad()
    def score(
        self,
        state: torch.Tensor,
        thought: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Return Q scores used to rank candidate thoughts at inference time."""
        return self.q(state, thought, goal)


def load_critic_checkpoint(
    path: str,
    device: str | torch.device = "cpu",
) -> GoalConditionedIQLCritic:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = GoalConditionedIQLCritic(
        state_dim=config["state_dim"],
        thought_dim=config["thought_dim"],
        hidden_dims=tuple(config["hidden_dims"]),
        normalize_inputs=config["normalize_inputs"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
