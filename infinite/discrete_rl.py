import importlib
from typing import Tuple

import numpy as np

import gym
from gym.spaces import MultiBinary

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvWrapper,
    VecEnv,
    VecEnvStepReturn,
    VecEnvObs,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="discrete_rl.py",
        description="Run the classical RL algorithm on the discrete space induced by predicates",
    )
    parser.add_argument(
        type=str,
        dest="script_path",
        help="path to python script that defines the environment and the preidcates",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.csv",
        help="destination CSV file",
    )

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    episodes: int = parameters.episodes
    seed: int = parameters.seed
    nenvs: int = parameters.n
    output: str = parameters.output

    module = importlib.import_module(script_path.replace(".py", "").replace("/", "."))
    predicates = module.__getattribute__("predicates")
    env_fn = module.__getattribute__("make_env")

    env = env_fn()

    def wrap(state) -> np.ndarray:
        out = np.zeros(len(predicates))
        for i, p in enumerate(predicates):
            out[i] = int(p(state))
        return out

    if isinstance(env, VecEnv):

        class EquivWrapper(VecEnvWrapper):
            """
            Vectorize the action

            :param env: the environment to wrap
            """

            def __init__(self, env: VecEnv) -> None:
                super().__init__(env)
                self.observation_space = MultiBinary([env.num_envs, len(predicates)])

            def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
                obs, r, done, info = self.venv.step(action)
                return np.array([wrap(s) for s in obs]), r, done, info

            def step_wait(self) -> VecEnvStepReturn:
                return self.venv.step_wait()

            def reset(self) -> VecEnvObs:
                obs = self.venv.reset()
                return np.array([wrap(s) for s in obs])

    else:

        class EquivWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = MultiBinary(len(predicates))

            def observation(self, obs):
                return wrap(obs)

    train_env = EquivWrapper(env)

    model = DQN("MlpPolicy", train_env).learn(total_timesteps=1000)
