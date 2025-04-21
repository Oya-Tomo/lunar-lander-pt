from dataclasses import dataclass
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.nn import functional as F


@dataclass
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.01


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNet, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        mean = self.mean_linear(x.clone())
        log_std = self.log_std_linear(x.clone())
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        norm = Normal(mean, std)
        action = F.tanh(norm.rsample())
        mean = F.tanh(mean)
        return action, mean


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNet, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class ActorCriticAgent:
    def __init__(self, device, config: ModelConfig):
        self.device = device
        self.config = config

        self.actor_net = ActorNet(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
        ).to(device)
        self.critic_net = CriticNet(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
        ).to(device)

        self.critic_net_target = CriticNet(
            config.state_dim,
            config.action_dim,
            config.hidden_dim,
        ).to(device)
        self.critic_net_target.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(),
            lr=config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(),
            lr=config.learning_rate,
        )

    def state_dict(self) -> dict:
        return {
            "actor_net": self.actor_net.state_dict(),
            "critic_net": self.critic_net.state_dict(),
            "critic_net_target": self.critic_net_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def save_state_dict(self, path: str):
        torch.save(self.state_dict(), path)

    def load_state_dict(self, path: str):
        state_dict = torch.load(path)
        self.actor_net.load_state_dict(state_dict["actor_net"])
        self.critic_net.load_state_dict(state_dict["critic_net"])
        self.critic_net_target.load_state_dict(state_dict["critic_net_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

    def select_action(self, state: torch.Tensor, train: bool) -> torch.Tensor:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if train:
            action, _ = self.actor_net.sample(state)
        else:
            _, action = self.actor_net.sample(state)
        return action.squeeze(0).cpu().detach()

    def soft_update(self):
        for target_param, source_param in zip(
            self.critic_net_target.parameters(), self.critic_net.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * source_param.data
                + (1.0 - self.config.tau) * target_param.data
            )

    def update_parameters(self, dataloader: DataLoader) -> tuple[float, float]:
        critic_loss_list = []
        actor_loss_list = []

        for state, action, reward, next_state, done in dataloader:
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)

            # Update critic
            with torch.no_grad():
                next_action, _ = self.actor_net.sample(next_state)
                next_q_value = self.critic_net_target(next_state, next_action)
                target_q = reward + (1.0 - done) * self.config.gamma * next_q_value

            current_q = self.critic_net(state, action)
            critic_loss = F.mse_loss(current_q, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update actor
            action, _ = self.actor_net.sample(state)
            q_value = self.critic_net(state, action)
            actor_loss: torch.Tensor = -q_value.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_loss_list.append(critic_loss.item())
            actor_loss_list.append(actor_loss.item())

        critic_loss_avg = sum(critic_loss_list) / len(critic_loss_list)
        actor_loss_avg = sum(actor_loss_list) / len(actor_loss_list)
        return critic_loss_avg, actor_loss_avg
