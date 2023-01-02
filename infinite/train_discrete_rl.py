import importlib
import os
import sys
from typing import Optional, Tuple
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
        "-f",
        "--folder",
        type=str,
        default="logs",
        help="logs folder from rl_zoo (default: logs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="discrete_model.zip",
        help="destination model file (default: discrete_model.zip)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed used when RNG is used",
    )

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    folder: str = parameters.folder
    output: str = parameters.output
    seed: Optional[int] = parameters.seed

    # Find existing config folder
    env_name = script_path.split("/")[-2]
    trained_dir = os.path.join(".", folder, "dqn")
    env_dir = None

    for file in os.listdir(trained_dir):
        if (
            os.path.isdir(os.path.join(trained_dir, file))
            and env_name.replace("-", "") in file.lower()
        ):
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

    module = importlib.import_module(
        script_path.replace(".py", "").replace("./", "").replace("/", ".")
    )
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

            def __init__(self, venv: VecEnv) -> None:
                super().__init__(
                    venv,
                    observation_space=MultiBinary(len(predicates)),
                )

            def step_wait(self) -> VecEnvStepReturn:
                obs, r, dones, infos = self.venv.step_wait()
                for i, done in enumerate(dones):
                    if done:
                        infos[i]["terminal_observation"] = wrap(
                            infos[i]["terminal_observation"]
                        )
                return np.array([wrap(s) for s in obs]), r, dones, infos

            def reset(self) -> VecEnvObs:
                obs = self.venv.reset()
                return np.array([wrap(s) for s in obs])

            def close(self) -> None:
                return self.venv.close()

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
    if "policy_kwargs" in config:
        config["policy_kwargs"] = eval(config["policy_kwargs"])
    for elem in ["seed", "verbose", "device"]:
        config[elem] = args[elem]

    if seed is not None:
        config["seed"] = seed

    if "env_wrapper" in config:
        del config["env_wrapper"]
    if "frame_stack" in config:
        del config["frame_stack"]

    model = DQN("MlpPolicy", train_env, **config)
    model.learn(
        n_timesteps,
        progress_bar=args["progress"],
        n_eval_episodes=args["eval_episodes"],
        eval_freq=args["eval_freq"],
        log_interval=args["log_interval"],
        eval_log_path=os.path.join(os.path.split(env_dir)[0], "discrete"),
    )
    model.save(output)
