import gym
from stable_baselines3 import DQN

import argparse


parser = argparse.ArgumentParser(
    description="Record one episode of an agent interacting with a gym environment"
)
parser.add_argument(
    type=str,
    dest="model_file",
    action="store",
    help="file for the model",
)
parser.add_argument(
    type=str,
    dest="env_id",
    help="name of the environment",
)

parameters = parser.parse_args()
file: str = parameters.model_file
env_id: str = parameters.env_id


model = DQN(
    "MlpPolicy",
    env_id,
    verbose=0,
    exploration_final_eps=0,
    target_update_interval=250,
).load(file)

try:
    from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

    video_folder = "videos/"
    video_length = int(1e8)

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
    done = False
    while not done:
        action = [model.predict(obs)[0][0]]
        obs, _, done, _ = env.step(action)
    # Save the video
    env.close()
except NameError:
    print(
        "Failed to record episode, check that you have ffmpeg installed if you would like a video."
    )
