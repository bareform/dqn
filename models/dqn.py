import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        depth: int,
        W: int
    ) -> None:
        super().__init__()
        layers = [
            nn.Linear(state_dim, W),
            nn.ReLU()
        ]
        for _ in range(depth - 1):
            layers.append(nn.Linear(W, W))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(W, action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.layers(input)
        return out