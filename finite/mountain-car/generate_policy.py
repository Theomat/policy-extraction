from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from env import make_env

import argparse


parser = argparse.ArgumentParser(
    description="Generate a trained DQN network on a gym environment"
)
parser.add_argument(
    type=str,
    dest="file",
    action="store",
    help="destination file for the model",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=2410,
    help="seed (default: 2410)",
)
parser.add_argument(
    "-v",
    "--verbose",
    default=False,
    action="store_true",
    help="verbose training",
)

parameters = parser.parse_args()
file: str = parameters.file
seed: int = parameters.seed
verbose: bool = parameters.verbose

# Parameters from https://github.com/DLR-RM/rl-trained-agents/blob/1e2a45e5d06efd6cc15da6cf2d1939d72dcbdf87/dqn/MountainCar-v0_1/MountainCar-v0/config.yml
model = DQN(
    "MlpPolicy",
    make_env(),
    batch_size=128,
    buffer_size=10000,
    exploration_final_eps=0.07,
    exploration_fraction=0.2,
    gamma=0.98,
    gradient_steps=8,
    learning_rate=0.004,
    learning_starts=1000,
    policy_kwargs={"net_arch": [256, 256]},
    target_update_interval=600,
    train_freq=16,
    verbose=int(verbose),
    seed=seed,
)

# Separate env for evaluation
eval_env = make_env()


# Random Agent, before training
bmean_reward, bstd_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)


# Train the agent
model.learn(total_timesteps=120000, progress_bar=True)
# Save the agent
model.save(file)

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"Before training mean_reward={bmean_reward:.2f} +/- {bstd_reward}")
print(f"After training mean_reward={mean_reward:.2f} +/- {std_reward}")
