import importlib
import os
import sys
from typing import Optional
from collections import OrderedDict

import yaml
import numpy as np

import torch

import gym
from gym.spaces import MultiBinary

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvWrapper,
    VecEnv,
    VecEnvStepReturn,
    VecEnvObs,
)

from rich import print
from rich.text import Text

REWARD_STYLE = "gold1"
FINITE_LOSS_STYLE = "bright_magenta"


def print_reward(eval_episodes: int, mean: float, diff: float):
    print(
        f"95% of rewards over {eval_episodes} episodes fall into: ",
        Text.assemble(
            "[",
            (f"{mean- diff:.2f}", REWARD_STYLE),
            ";",
            (f"{mean + diff:.2f}", REWARD_STYLE),
            f"] (mean=",
            (f"{mean}", REWARD_STYLE),
            ")",
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="eval_discrete_rl.py",
        description="Run the classical RL algorithm on the discrete space induced by predicates",
    )
    parser.add_argument(
        type=str,
        dest="script_path",
        help="path to python script that defines the environment and the predicates",
    )
    parser.add_argument(
        type=str,
        dest="model_path",
        help="path to model",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="logs",
        help="logs folder from rl_zoo (default: logs)",
    )
    parser.add_argument(
        type=int,
        dest="episodes",
        help="number of episodes used for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=127,
        help="seed used when RNG is used",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=32,
        help="number of envs to run simultaneously (i.e. batch size)",
    )
    parser.add_argument(
        "-y",
        dest="autoyes",
        action="store_true",
        help="say yes automatically when asking if help from solver is wanted",
    )

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    folder: str = parameters.folder
    episodes: int = parameters.episodes
    nenvs: int = parameters.n
    model_path: str = parameters.model_path
    auto_yes: bool = parameters.autoyes
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
    Q_builder = module.__getattribute__("Q_builder")
    original_model_file = os.path.join(os.path.split(env_dir)[0], "best_model.zip")
    Qfun = Q_builder(original_model_file)
    print("classic model loaded!")
    env = env_fn()

    def wrap(state) -> tuple[int, ...]:
        return tuple(int(p(state)) for p in predicates)

    n_timesteps = config["n_timesteps"]
    del config["n_timesteps"]
    del config["policy"]
    config["policy_kwargs"] = eval(config["policy_kwargs"])
    for elem in ["seed", "verbose", "device"]:
        config[elem] = args[elem]

    if seed is not None:
        config["seed"] = seed
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
    model = DQN("MlpPolicy", train_env, **config)
    model = model.load(model_path, buffer_size=0)
    print("discrete model loaded!")

    from polext.interaction_helper import vec_interact

    def qfun(observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation, device=model.device)
        batched = len(observation.shape) == 2
        if not batched:
            observation.unsqueeze_(0)
        # print(observation)
        # assert False
        with torch.no_grad():
            q_values = model.q_net(observation).cpu().numpy()
        if batched:
            return q_values
        else:
            return q_values[0]

    score = {}
    good_states = {}
    bad_states = {}
    action = {}

    def step(b, episode_num: int, state, Qval, *args):
        wstate = wrap(state)
        if wstate not in score:
            score[wstate] = 0
            good_states[wstate] = []
            bad_states[wstate] = []
        best = np.argmax(Qval)
        if action.get(wstate, best + 1) == best:
            good_states[wstate].append(state)
            return b

        qdis = qfun(np.array(wstate))
        chosen = np.argmax(qdis)
        action[wstate] = chosen
        if best == chosen:
            good_states[wstate].append(state)
            return b
        regret = Qval[best] - qdis[chosen]
        score[wstate] += regret
        bad_states[wstate].append((regret, best, tuple(state)))
        return b

    vec_interact(
        Qfun,
        episodes,
        env_fn,
        nenvs,
        step,
        0,
        seed=seed,
    )

    ranked = sorted(score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(3):
        print(f"Most valuable split n°{i+1}:")
        key = ranked[i][0]
        visits = len(good_states[key]) + len(bad_states[key])
        print("\tlost:", ranked[i][1], "visits:", visits)
        print("\tlost on average:", ranked[i][1] / visits)
        print(f"\tstates are saved in best_{i}.txt")
        with open(f"best_{i}.txt", "w") as fd:
            for j, pred in enumerate(predicates):
                if key[j] == 1:
                    fd.write(f"sat: {pred.name}\n")
            fd.write(f"\nAction Played: {action[key]}\n")
            for lost_q, best, state in sorted(bad_states[key], reverse=True):
                fd.write(f"lost:{lost_q:.2e} best:{best} state:{state}\n")
            fd.write("=" * 60 + "\n")
            for state in good_states[key]:
                fd.write(f"good state:{state}\n")
        if len(good_states[key]) > 0:
            should_try = (
                auto_yes
                or (
                    input(
                        "Do you want to try using linear programming to find a separator?[y/*]"
                    )
                    .strip()
                    .lower()
                )
                == "y"
            )
            if should_try:
                from docplex.mp.solution import SolveSolution
                from docplex.mp.model import Model

                m = Model(name="separator")
                shape = env.observation_space.shape[0]
                space_var = m.continuous_var(name="space", lb=0, ub=1)
                vars = [m.continuous_var(name=f"s[{i}]") for i in range(shape)]
                squared_vars = [m.continuous_var(name=f"s[{i}]**2") for i in range(shape)]
                mixed_vars = [
                    m.continuous_var(name=f"s[{i}] * s[{j}]")
                    for j in range(i + 1, shape)
                    for i in range(shape)
                ]
                all_vars = vars + squared_vars + mixed_vars
                comb2idx = {
                    tp: idx
                    for idx, tp in enumerate(
                        (i, j) for j in range(i + 1, shape) for i in range(shape)
                    )
                }
                for state in good_states[key][:500]:
                    linear = sum(state[i] * vars[i] for i in range(shape))
                    square = sum(
                        state[i] * state[i] * squared_vars[i] for i in range(shape)
                    )
                    mixed = sum(
                        state[i] * state[j] * mixed_vars[comb2idx[(i, j)]]
                        for j in range(i + 1, shape)
                        for i in range(shape)
                    )
                    m.add_constraint(linear + square + mixed >= space_var + 1e-6)
                for a, b, state in bad_states[key][:500]:
                    linear = sum(state[i] * vars[i] for i in range(shape))
                    square = sum(
                        state[i] * state[i] * squared_vars[i] for i in range(shape)
                    )
                    mixed = sum(
                        state[i] * state[j] * mixed_vars[comb2idx[(i, j)]]
                        for j in range(i + 1, shape)
                        for i in range(shape)
                    )
                    m.add_constraint(linear + square + mixed <= -space_var - 1e-6)
                m.set_objective("max", space_var)
                solution: SolveSolution = m.solve()
                if solution is not None:
                    solution.display()
                    predicate = "Predicate(\""
                    elements = []
                    for v in all_vars:
                        coeff = solution[v]
                        if abs(coeff) > 1:
                            elements.append(f"{v.name} * {coeff}")
                    expr = " + ".join(elements)
                    predicate +=  expr + " < 0\",\n\tlambda s:" + expr + " < 0)"
                    with open(f"best_{i}_separator.py", "w") as fd:
                        fd.write(predicate) 
                else:
                    print("\tcould not find a good separator...")
