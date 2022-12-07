import importlib
import os
import sys
from typing import Tuple
from collections import OrderedDict

import yaml
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
        help="path to python script that defines the environment and the predicates",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="discrete_model.zip",
        help="destination model file (default: discrete_model.zip)",
    )

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    output: str = parameters.output

    # Find existing config folder
    env_name = script_path.split("/")[-2]
    trained_dir = "./logs/dqn/"
    env_dir = None

    for file in os.listdir(trained_dir):
        if os.path.isdir(os.path.join(trained_dir, file)) and env_name in file.lower():
            env_dir = os.path.join(trained_dir, file, file[: file.rfind("_")])
            break
    if env_dir is None:
        print(
            "Could not find logs of already trained DQN which is needed to get parameters, please train a DQN first!"
        )
        sys.exit(1)
    print("Found existing environment directory:", env_dir)

    # Load parameters
    def ord_constr(loader: yaml.FullLoader, node):
        seq_nodes = node.value[0]
        out = OrderedDict()
        for seq_node in seq_nodes.value:
            objs = [loader.construct_object(x) for x in seq_node.value]
            out[objs[0]] = objs[1]
        return out

    yaml.add_constructor(
        "tag:yaml.org,2002:python/object/apply:collections.OrderedDict", ord_constr
    )
    with open(os.path.join(env_dir, "args.yml")) as fd:
        args = yaml.full_load(fd)
    with open(os.path.join(env_dir, "config.yml")) as fd:
        config = yaml.full_load(fd)

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

    n_timesteps = config["n_timesteps"]
    del config["n_timesteps"]
    del config["policy"]
    config["policy_kwargs"] = eval(config["policy_kwargs"])
    for elem in ["seed", "verbose", "device"]:
        config[elem] = args[elem]

    model = DQN("MlpPolicy", train_env, **config)
    model.learn(
        n_timesteps,
        progress_bar=args["progress"],
        n_eval_episodes=args["eval_episodes"],
        eval_freq=args["eval_freq"],
        log_interval=args["log_interval"],
        eval_log_path=os.path.join(args["log_folder"], "discrete"),
    )
    model.save(output)
