import importlib
import sys
from typing import Any, Callable, Tuple, Union

from rich import print
from rich.text import Text

import numpy as np

from polext.decision_tree import DecisionTree
from polext.forest import Forest
from polext.predicate_space import PredicateSpace
from polext.interaction_helper import vec_eval_policy, vec_interact
from polext.tree_builder import build_tree, tree_loss, list_registered_algorithms
from polext.q_values_learner import QValuesLearner
from polext.algos import *


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


def run_method(
    space: PredicateSpace,
    Qtable: QValuesLearner,
    max_depth: int,
    method: str,
    seed: int,
    builder: Callable,
    callback: Callable,
):
    if method not in list_registered_algorithms():
        print(
            Text.assemble(
                ('Method:"', "red"),
                method,
                ('" is unknown, available methods are:', "red"),
                ", ".join(list_registered_algorithms()),
            ),
            file=sys.stderr,
        )
        return
    i = 0
    for val in builder(space, Qtable, max_depth, method, seed=seed):
        print("Method:", Text.assemble((method, "bold")), f"iteration {i+1}")
        tree = callback(val)
        if isinstance(tree, DecisionTree):
            tree.print()
        i += 1
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="polext", description="Extract a Decision Tree from a DQN"
    )
    parser.add_argument(
        type=str,
        dest="script_path",
        help="path to python script that defines the environment and the predicates",
    )
    parser.add_argument(
        type=str,
        dest="model_path",
        help="DQN model file to load",
    )
    parser.add_argument(
        type=str,
        dest="method",
        help=f"methods to be used any of {list_registered_algorithms()}",
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
        "--iterations",
        type=int,
        default=1,
        help="number of iterations of the state space distribution evaluation",
    )

    group = parser.add_argument_group(
        "Model parameters", "parameters that change the model"
    )
    group.add_argument(
        "--forest",
        type=int,
        help="train a random forest with the given number of trees",
        default=0,
    )
    group.add_argument("--depth", type=int, default=5, help="max decision tree depth")

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path
    method: str = parameters.method
    episodes: int = parameters.episodes
    iterations: int = parameters.iterations
    seed: int = parameters.seed
    max_depth: int = parameters.depth
    ntrees: int = parameters.forest
    nenvs: int = parameters.n

    # Parameters check
    if episodes < 1:
        print(
            Text.assemble(("You must evaluate on at least one episode!", "red")),
            file=sys.stderr,
        )
        sys.exit(1)
    if iterations < 1:
        print(
            Text.assemble(("You cannot do less than one iteration!", "red")),
            file=sys.stderr,
        )
        sys.exit(1)
    # Load module
    module = importlib.import_module(
        script_path.replace(".py", "").replace("./", "").replace("/", ".")
    )
    predicates = module.__getattribute__("predicates")
    Q_builder = module.__getattribute__("Q_builder")
    env_fn = module.__getattribute__("make_env")

    Q = Q_builder(model_path)

    # Find out methods to run
    methods_todo = []
    if method == "all":
        methods_todo = list_registered_algorithms()
    elif "," in method:
        methods_todo = method.split(",")
    else:
        methods_todo = [method]

    # Setup predicate space
    space = PredicateSpace(predicates)
    Qtable = QValuesLearner()
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
        s = space.get_representative(state)
        Qtable.add_one_visit(s, Qvalues)
        return val + r

    rewards = vec_interact(Q, episodes, env_fn, nenvs, my_step, 0, seed=seed)

    print("Baseline Q-table:")
    print_reward(episodes, np.mean(rewards), 2 * np.std(rewards))
    print()

    eval_fn = vec_eval_policy
    loss = tree_loss
    base_builder = build_tree
    # TODO:
    # if ntrees > 1:
    #     base_builder = lambda *args, **kwargs: build_forest(
    #         *args, trees=ntrees, **kwargs
    #     )
    #     loss = forest_loss
    builder = lambda *args, **kwargs: base_builder(
        *args,
        Qfun=Q,
        env_fn=env_fn,
        nenvs=nenvs,
        episodes=episodes,
        iterations=iterations,
        **kwargs,
    )

    def callback(
        out: Tuple[Union[DecisionTree, Forest], Any]
    ) -> Union[DecisionTree, Forest]:
        tree, score = out
        print(
            "Lost Q-Values:",
            Text.assemble((str(loss(tree, space, Qtable)), FINITE_LOSS_STYLE)),
        )
        print_reward(episodes, score[0], score[1])
        return tree

    for method in methods_todo:
        run_method(
            space,
            Qtable,
            max_depth,
            method,
            seed,
            builder,
            callback,
        )
