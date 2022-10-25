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
    default="ALE/Pong-ram-v5",
    help="name of the environment (default: ALE/Pong-ram-v5)",
)
parser.add_argument(
    "-t",
    "--timesteps",
    type=int,
    default=int(5e5),
    help="number of learning timesteps (default: 5e5)",
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
env_id: str = parameters.env
timesteps: int = parameters.timesteps
seed: int = parameters.seed
verbose: bool = parameters.verbose


model = DQN(
    "MlpPolicy",
    env_id,
    verbose=int(verbose),
    exploration_final_eps=0.1,
    target_update_interval=250,
    seed=seed,
)

# Separate env for evaluation
eval_env = gym.make(env_id)


# Random Agent, before training
bmean_reward, bstd_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)


# Train the agent
model.learn(total_timesteps=timesteps)
# Save the agent
model.save(file)

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"Before training mean_reward={bmean_reward:.2f} +/- {bstd_reward}")
print(f"After training mean_reward={mean_reward:.2f} +/- {std_reward}")

try:
    from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

    video_folder = "videos/"
    video_length = 100

    env = DummyVecEnv([lambda: gym.make(env_id)])

    obs = env.reset()

    # Record the video starting at the first step
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=f"dqn-{env_id}",
    )

    obs = env.reset()
    for _ in range(video_length + 1):
        action = [model.predict(obs)]
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()
except NameError:
    print(
        "Failed to record episode, check that you have ffmpeg installed if you would like a video."
    )
