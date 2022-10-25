import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

import argparse


parser = argparse.ArgumentParser(
    description="Generate a trained DQN network for a gym environment"
)
parser.add_argument(
    type=str,
    dest="file",
    action="store",
    help="destination file for the model",
)
parser.add_argument(
    "-e",
    "--env",
    type=str,
    default="LunarLander-v2",
    help="name of the environment (default: LunarLander-v2)",
)
parser.add_argument(
    "-t",
    "--timesteps",
    type=int,
    default=int(1e5),
    help="number of learning timesteps (default: 1e5)",
)


parameters = parser.parse_args()
file: str = parameters.file
env: str = parameters.env
timesteps: int = parameters.timesteps


model = model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    exploration_final_eps=0.1,
    target_update_interval=250,
)


# Separate env for evaluation
eval_env = gym.make(env)


# Random Agent, before training
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"Before training mean_reward={mean_reward:.2f} +/- {std_reward}")


# Train the agent
model.learn(total_timesteps=timesteps)
# Save the agent
model.save(file)

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"After training mean_reward={mean_reward:.2f} +/- {std_reward}")
