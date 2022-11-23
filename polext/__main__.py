import importlib
import sys
from typing import Callable, List, Optional, Set, Tuple, Union
import numpy as np

from rich import print
from rich.text import Text

from polext.decision_tree import DecisionTree, Leaf
from polext.forest import Forest, majority_vote
from polext.predicate_space import PredicateSpace, enumerated_space, sampled_space


def eval_policy(
    policy: Callable[[np.ndarray], int], episodes: int, env
) -> tuple[float, float]:
    total_rewards = []
    for _ in range(episodes):
        total_reward = 0
        done = False
        state = env.reset()
        while not done:
            state, reward, done, _ = env.step(policy(state))
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), 2 * np.std(total_rewards)


def eval_forest(forest: Forest, episodes: int, env) -> tuple[float, float]:
    return eval_policy(lambda state: majority_vote(forest(state)), episodes, env)


def eval_q(Q, episodes: int, env) -> tuple[float, float]:
    def f(state) -> int:
        action = np.argmax(Q(state))
        if isinstance(action, np.ndarray):
            action = action[0]
        return action

    return eval_policy(f, episodes, env)


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


def configure_tree_or_forest(
    forest: Optional[List[int]], build_tree: Callable, build_forest: Callable
) -> Tuple[Callable, Callable]:
    eval_fn = eval_policy
    builder = build_tree
    if forest is not None:
        n_trees = forest[0]
        if n_trees <= 1:
            print(
                Text.assemble(
                    ("Number of trees for a random forest must be >1!", "red")
                ),
                file=sys.stderr,
            )
            sys.exit(1)
        eval_fn = eval_forest
        builder = lambda *args, **kwargs: build_forest(*args, trees=n_trees, **kwargs)
    return eval_fn, builder


def run_method(
    space: PredicateSpace,
    max_depth: int,
    method: str,
    seed: int,
    allowed_methods: List[str],
    eval_fn: Callable,
    builder: Callable,
    callback: Callable,
):
    if method not in allowed_methods:
        print(
            Text.assemble(
                ('Method:"', "red"),
                method,
                ('" is unkown, available methods are:', "red"),
                ", ".join(allowed_methods),
            ),
            file=sys.stderr,
        )
        return
    print("Method:", Text.assemble((method, "bold")))
    out = builder(space, max_depth, method, seed=seed)
    tree = callback(out)
    if eval_episodes > 0:
        env = module.__getattribute__("make_env")()
        mean, diff = eval_fn(tree, eval_episodes, env)
        print_reward(eval_episodes, mean, diff)
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
        "--finite",
        type=str,
        nargs="?",
        help="finite state space, argument is the name  of the method to use",
        default="none",
    )
    parser.add_argument(
        "--infinite",
        type=str,
        nargs="?",
        help="infinite state space, argument is the name of the method to use",
        default="none",
    )
    parser.add_argument(
        "--forest",
        type=int,
        nargs=1,
        help="train a random forest with the given number of trees",
    )
    parser.add_argument(
        "--sampled",
        type=int,
        nargs=1,
        help="sample state space parameter is number of episodes to sample for",
    )
    parser.add_argument(
        "--default-space",
        action="store_true",
        help="use state space and not predicate space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs=1,
        default=127,
        help="seed used when RNG is used",
    )
    parser.add_argument(
        "--eval",
        type=int,
        nargs="?",
        help="number of evaluation episodes",
        default=0,
    )
    parser.add_argument("--depth", type=int, default=5, help="max decision tree depth")

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path
    finite_method: str = parameters.finite
    infinite_method: str = parameters.infinite
    eval_episodes: int = parameters.eval
    predicate_space: bool = not parameters.default_space
    seed: int = parameters.seed
    if isinstance(seed, List):
        seed = seed[0]
    max_depth: bool = parameters.depth
    forest: Optional[List[int]] = parameters.forest
    sampled: Optional[List[int]] = parameters.sampled

    module = importlib.import_module(script_path.replace(".py", "").replace("/", "."))
    predicates = module.__getattribute__("predicates")
    Q_builder = module.__getattribute__("Q_builder")
    Q = Q_builder(model_path)

    # Eval Q-values if required
    if eval_episodes > 0:
        env = module.__getattribute__("make_env")()
        mean, diff = eval_q(Q, eval_episodes, env)
        print("Baseline Q-table:")
        print_reward(eval_episodes, mean, diff)
        print()

    if finite_method != "none":
        from polext.finite import build_tree, FINITE_METHODS, build_forest

        # Setup predicate space
        states = module.__getattribute__("states")
        space = enumerated_space(states, Q, predicates, predicate_space)
        if sampled is not None:
            env = module.__getattribute__("make_env")()
            space = sampled_space(env, sampled[0], Q, predicates, predicate_space)
        # Setup eval function and building function
        eval_fn, builder = configure_tree_or_forest(forest, build_tree, build_forest)
        # Find out methods to run
        methods_todo = []
        if finite_method == "all":
            methods_todo = FINITE_METHODS
        elif "," in finite_method:
            methods_todo = finite_method.split(",")
        else:
            methods_todo = [finite_method]

        def callback(
            out: Tuple[Union[DecisionTree, Forest], float]
        ) -> Union[DecisionTree, Forest]:
            tree, score = out
            print("Lost Q-Values:", Text.assemble((str(score), FINITE_LOSS_STYLE)))
            return tree

        # Run the methods
        for method in methods_todo:
            run_method(
                space,
                max_depth,
                method,
                seed,
                FINITE_METHODS,
                eval_fn,
                builder,
                callback,
            )

    else:
        from polext.finite import build_tree, FINITE_METHODS, build_forest

        assert sampled is not None
        env = module.__getattribute__("make_env")()
        space = sampled_space(env, sampled[0], Q, predicates, predicate_space)

        # Setup eval function and building function
        eval_fn, builder = configure_tree_or_forest(forest, build_tree, build_forest)
        # Find out methods to run
        methods_todo = []
        if infinite_method == "all":
            methods_todo = FINITE_METHODS
        elif "," in infinite_method:
            methods_todo = infinite_method.split(",")
        else:
            methods_todo = [infinite_method]

        def callback(
            out: Tuple[Union[DecisionTree, Forest], float]
        ) -> Union[DecisionTree, Forest]:
            tree, score = out
            print("Lost Q-Values:", Text.assemble((str(score), FINITE_LOSS_STYLE)))
            return tree

        # Run the methods
        for method in methods_todo:
            run_method(
                space,
                max_depth,
                method,
                seed,
                FINITE_METHODS,
                eval_fn,
                builder,
                callback,
            )
