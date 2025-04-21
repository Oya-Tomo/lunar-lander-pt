import os
from dataclasses import dataclass

import torch
import wandb

from env import LunarLanderV3
from actor_critic import ActorCriticAgent, ModelConfig
from dataset import ReplayBufferDataset, DatasetConfig


@dataclass
class TrainConfig:
    epochs: int = 100000
    episodes_per_epoch: int = 40
    max_steps_per_episode: int = 10000
    soft_update_per_epoch: int = 5
    seed: int = 42
    random_start_steps: int = 70
    epoch_per_save: int = 10


def train():
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    if WANDB_API_KEY is None:
        raise ValueError("WANDB_API_KEY environment variable not set.")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="lunar-lander-pt",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_config = TrainConfig()

    env = LunarLanderV3(gui=False)

    agent = ActorCriticAgent(
        device=device,
        config=ModelConfig(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=256,
            learning_rate=1e-3,
            gamma=0.99,
            tau=0.005,
        ),
    )

    dataset = ReplayBufferDataset(
        config=DatasetConfig(
            max_size=100000,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    print("Filling replay buffer with random actions...")
    while len(dataset) < 100000:
        state, _ = env.reset()
        for _ in range(train_config.max_steps_per_episode):
            action = env.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            dataset.add(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    print("Replay buffer filled.")
    print(f"Buffer size: {len(dataset)}")
    print("Training started.")

    for epoch in range(train_config.epochs):
        rewards = []

        for episode in range(train_config.episodes_per_epoch):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(train_config.max_steps_per_episode):
                if step < train_config.random_start_steps:
                    action = env.sample()
                else:
                    action = agent.select_action(state, True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                dataset.add(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            rewards.append(episode_reward)

        dataloader = dataset.create_dataloader()
        critic_loss, actor_loss = agent.update_parameters(dataloader)

        if epoch % train_config.soft_update_per_epoch == 0:
            agent.soft_update()

        wandb.log(
            {
                "epoch": epoch,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "average_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
            }
        )
        print(f"Epoch {epoch} completed.")

        if epoch % train_config.epoch_per_save == train_config.epoch_per_save - 1:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")

            agent.save_state_dict(
                f"checkpoints/checkpoint{epoch}.pth",
            )
            print(f"Models saved at epoch {epoch}.")


if __name__ == "__main__":
    train()
