import importlib
import sys
from typing import Any, Callable, List, Tuple, Union

from rich import print
from rich.text import Text

import numpy as np

from polext.decision_tree import DecisionTree
from polext.finite import FINITE_METHODS
from polext.forest import Forest, majority_vote
from polext.predicate_space import PredicateSpace, enumerated_space
from polext.interaction_helper import vec_eval_policy, vec_interact


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
    max_depth: int,
    method: str,
    seed: int,
    allowed_methods: List[str],
    builder: Callable,
    callback: Callable,
):
    if method not in allowed_methods:
        print(
            Text.assemble(
                ('Method:"', "red"),
                method,
                ('" is unknown, available methods are:', "red"),
                ", ".join(allowed_methods),
            ),
            file=sys.stderr,
        )
        return
    print("Method:", Text.assemble((method, "bold")))
    out = builder(space, max_depth, method, seed=seed)
    tree = callback(out)
    if isinstance(tree, DecisionTree):
        tree.print()
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
        help="methods to be used",
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

    group = parser.add_argument_group(
        "State space", "parameters that change the state space"
    )
    group.add_argument(
        "--finite",
        action="store_true",
        help="finite state space and state space is not sampled",
    )
    group.add_argument(
        "--default-space",
        action="store_true",
        help="use state space and not predicate space",
    )
    group.add_argument(
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
    predicate_space: bool = not parameters.default_space
    iterations: int = parameters.iterations
    seed: int = parameters.seed
    finite: bool = parameters.finite
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
    if finite and iterations > 1:
        print(
            Text.assemble(
                ("You cannot do mutliple iterations with the finite flag!", "red")
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    # Load module
    module = importlib.import_module(script_path.replace(".py", "").replace("/", "."))
    predicates = module.__getattribute__("predicates")
    Q_builder = module.__getattribute__("Q_builder")
    env_fn = module.__getattribute__("make_env")

    Q = Q_builder(model_path)

    # Find out methods to run
    methods_todo = []
    if method == "all":
        methods_todo = FINITE_METHODS
    elif "," in method:
        methods_todo = method.split(",")
    else:
        methods_todo = [method]

    # Guess if environment is finite or infinite
    is_infinite = True
    try:
        states = module.__getattribute__("states")
        is_infinite = False
    except:
        pass
    if finite and is_infinite:
        print(
            Text.assemble(
                ("You cannot use the finite flag with an infinite state space!", "red")
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    # Setup predicate space
    should_sample_space = True
    if finite:
        states = module.__getattribute__("states")
        space = enumerated_space(states, Q, predicates, predicate_space)
        should_sample_space = False
    else:
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
        if should_sample_space:
            space.visit_state(state, Qvalues)
        return val + r

    rewards = vec_interact(Q, episodes, env_fn, nenvs, my_step, 0, seed)
    env = env_fn()

    print("Baseline Q-table:")
    print_reward(episodes, np.mean(rewards), 2 * np.std(rewards))
    print()

    # Setup eval function and building function
    from polext.finite import build_forest, tree_loss, forest_loss

    if is_infinite:
        from polext.infinite import build_tree
    else:
        from polext.finite import build_tree

    eval_fn = vec_eval_policy
    loss = tree_loss
    base_builder = build_tree
    if ntrees > 1:
        eval_fn = lambda f, *args, **kwargs: vec_eval_policy(
            f.policy(majority_vote), *args, **kwargs
        )
        base_builder = lambda *args, **kwargs: build_forest(
            *args, trees=ntrees, **kwargs
        )
        loss = forest_loss
    builder = lambda *args, **kwargs: base_builder(
        *args, Qfun=Q, env=env, episodes=episodes, iterations=iterations, **kwargs
    )

    def callback(
        out: Tuple[Union[DecisionTree, Forest], Any]
    ) -> Union[DecisionTree, Forest]:
        tree, score = out
        if not is_infinite:
            print("Lost Q-Values:", Text.assemble((str(score), FINITE_LOSS_STYLE)))
            score = eval_fn(tree, episodes, env_fn, nenvs, space.nactions, seed)
        else:
            print(
                "Lost Q-Values:",
                Text.assemble((str(loss(tree, space)), FINITE_LOSS_STYLE)),
            )
        print_reward(episodes, score[0], score[1])
        return tree

    for method in methods_todo:
        run_method(
            space,
            max_depth,
            method,
            seed,
            FINITE_METHODS,
            builder,
            callback,
        )
