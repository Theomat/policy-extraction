import os
import subprocess
from typing import List
import json
import sys

import tqdm

envs = ["acrobot", "cart-pole", "pong", "lunar-lander", "mountain-car"]


def exec_cmd(command_line: List[str]) -> str:
    try:
        return subprocess.check_output(command_line, text=True)
    except:
        print("Failed when executing:\n", " ".join(command_line), file=sys.stderr)
        sys.exit(1)


def train_base_dqn(env_id: str, seed: int) -> None:
    if os.path.exists(f"./logs_{seed}/dqn/{env_id}_1/best_model.zip"):
        return
    cmd = f"python -m rl_zoo3.train --algo dqn --env {env_id} -f logs_{seed}/ --seed {seed}".split(
        " "
    )
    exec_cmd(cmd)


def train_discrete_dqn(env_path: str, seed: int) -> None:
    file = f"{os.path.split(env_path)[-2]}_{seed}_discrete_rl.zip"
    if os.path.exists(file):
        return
    cmd = f"python infinite/train_discrete_rl.py {env_path} -f logs_{seed}/ --seed {seed} -o {file}".split(
        " "
    )
    exec_cmd(cmd)


def eval_discrete_dqn(env_path: str, seed: int, episodes: int, nenvs: int) -> dict:
    file = f"{os.path.split(env_path)[-2]}_{seed}_discrete_rl.zip"
    cmd = f"python infinite/eval_discrete_rl.py {env_path} {file} {episodes} -n {nenvs} --seed {seed} -f logs_{seed}/".split(
        " "
    )
    output = exec_cmd(cmd)
    # TODO: extract output
    return {f"discrete-dqn": {f"{seed}": 0}}


def run_trees(
    env_path: str,
    methods: str,
    seed: int,
    episodes: int,
    env_id: str,
    depth: int,
    iterations: int,
    nenvs: int,
    trees: int = 1,
) -> dict:
    cmd = f"python -m polext {env_path} logs_{seed}/dqn/{env_id}_1/best_model.zip {methods} {episodes} --depth {depth} --iterations {iterations} --forest {trees} -n {nenvs} --seed {seed}".split(
        " "
    )
    output = exec_cmd(cmd)
    # TODO: extract output
    data = {f"method-d={depth}-it={iterations}": {f"{seed}": 0}}


def find_env_path(env_name: str) -> str:
    for file in os.listdir("./finite/"):
        if os.path.isdir(os.path.join("./finite/", file)) and env_name in file.lower():
            return os.path.join("./finite/", file, "env.py")
    for file in os.listdir("./infinite/"):
        if (
            os.path.isdir(os.path.join("./infinite/", file))
            and env_name in file.lower()
        ):
            return os.path.join("./infinite/", file, "env.py")
    assert False, f"could not find an environment with name:{env_name}"


def find_env_id(env_path: str) -> str:
    readme_path = os.path.join(os.path.split(env_path)[0], "README.md")
    with open(readme_path) as fd:
        for line in fd.readlines():
            if "`" in line:
                return line[line.find("`") + 1 : line.rfind("`")]
    assert False, f"could not find an env_id with README file:{readme_path}"


def parse_set(text: str) -> List[int]:
    if "-" in text:
        index = text.find("-")
        return list(range(int(text[:index]), int(text[index + 1 :])))
    else:
        return list(map(int, text.split(",")))


def union(data: dict, score_dict: dict) -> None:
    for key, value in score_dict.items():
        if key in data:
            for vkey, vval in value.items():
                data[key][vkey] = vval
        else:
            data[key] = value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_experiment.py", description="Run experiment for a given environment"
    )
    parser.add_argument(
        type=str,
        dest="env_name",
        help=f"environment name, any of {', '.join(envs)}",
    )
    parser.add_argument(
        type=str,
        dest="methods",
        help="tree methods to use",
    )
    parser.add_argument(
        type=int,
        dest="episodes",
        help="number of episodes used for evaluation",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of seeds to use",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=32,
        help="number of envs to run simultaneously (i.e. batch size)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model.csv",
        help="destination CSV file",
    )
    parser.add_argument(
        "-d",
        "--depths",
        type=str,
        default="5-10",
        help="depths to test 5-10=[5;10] 7,8={7,8}",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default="1",
        help="iterations to test 5-10=[5;10] 7,8={7,8}",
    )

    parameters = parser.parse_args()
    env_name: str = parameters.env_name
    methods: str = parameters.methods
    episodes: int = parameters.episodes
    seeds: int = parameters.seeds
    nenvs: int = parameters.n
    output: str = parameters.output

    depths = parse_set(parameters.depths)
    iterations = parse_set(parameters.iterations)

    env_path = find_env_path(env_name)
    env_id = find_env_id(env_path)

    all_data = {}
    # If existing data, load it
    if os.path.exists(output):
        with open(output) as fd:
            all_data = json.load(fd)
    # Capture data
    pbar = tqdm.trange(seeds, unit="seed")
    for i in pbar:
        seed = 2410 * i + 17 * i + i
        # If already done => skip it
        if all_data.get(seed, False):
            continue
        all_data[seed] = False
        # This part is recoverable from files
        pbar.set_postfix_str("RL training")
        train_base_dqn(env_id, seed)
        pbar.set_postfix_str("discrete RL training")
        train_discrete_dqn(env_path, seed)
        pbar.set_postfix_str("discrete RL eval")
        score_dict = eval_discrete_dqn(env_path, seed, episodes, nenvs)
        union(all_data, score_dict)
        for depth in depths:
            for iteration in iterations:
                pbar.set_postfix_str(f"Trees d={depth} it={iteration}")
                score_dict = run_trees(
                    env_path, methods, seed, episodes, env_id, depth, iteration, nenvs
                )
                union(all_data, score_dict)
        all_data[seed] = True
    # Save data
    with open(output, "w") as fd:
        json.dump(all_data, fd)
