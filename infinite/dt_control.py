import importlib

import numpy as np

from polext.predicate_space import PredicateSpace
from polext.interaction_helper import vec_interact


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="dt_control.py", description="Extract data from a DQN for dtControl"
    )
    parser.add_argument(
        type=str,
        dest="script_path",
        help="path to python script that defines the environment and the preidcates",
    )
    parser.add_argument(
        type=str,
        dest="model_path",
        help="DQN model file to load",
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
    model_path: str = parameters.model_path
    nenvs: int = parameters.n
    output: str = parameters.output

    module = importlib.import_module(script_path.replace(".py", "").replace("/", "."))
    predicates = module.__getattribute__("predicates")
    Q_builder = module.__getattribute__("Q_builder")
    env_fn = module.__getattribute__("make_env")
    Q = Q_builder(model_path)

    space = PredicateSpace(predicates, True)
    # Empirical evaluation of Q-values
    def my_step(
        val: float,
        nep: int,
        state: np.ndarray,
        Qvalues: np.ndarray,
        r: float,
        stp1: np.ndarray,
        done: bool,
    ) -> float:
        space.visit_state(state, Qvalues)
        return val + r

    rewards = vec_interact(Q, episodes, env_fn, nenvs, my_step, 0, seed)
    eq_states = space.predicates_states()

    with open(output, "w") as fd:
        fd.write("#NON-PERMISSIVE\n")
        N = len(eq_states)
        fd.write(f"#BEGIN {N} 1\n")
        lines = []
        for state in eq_states:
            q_values: np.ndarray = space.predicate_state_Q(state)
            action = np.argmax(q_values)
            line = ",".join(map(str, state)) + "," + str(action) + "\n"
            lines.append(line)
        fd.writelines(lines)
