import torch

from env import LunarLanderV3
from actor_critic import ActorCriticAgent, ModelConfig


def eval(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = LunarLanderV3(gui=True)

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
    agent.load_state_dict(path)

    for episode in range(1000):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(10000):
            if step < 70:
                action = env.sample()
            else:
                action = agent.select_action(state, True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            if done:
                break
        print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}")


if __name__ == "__main__":
    eval("checkpoints/checkpoint5819.pth")
