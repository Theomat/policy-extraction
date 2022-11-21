import importlib
import sys
from typing import Callable, List, Optional
import numpy as np

from rich import print
from rich.text import Text

from polext.decision_tree import DecisionTree, Leaf
from polext.finite.tree_builder import build_forest, easy_space, interactive_space
from polext.forest import Forest, majority_vote


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="polext", description="Extract a Decision Tree from a DQN"
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
        "--finite",
        type=str,
        nargs="?",
        help="finite state space and name of the method to use",
        default="none",
    )
    parser.add_argument(
        "--forest",
        type=int,
        nargs=1,
        help="train a random forest with the given number of trees",
    )
    parser.add_argument(
        "--interactive",
        type=int,
        nargs=1,
        help="interactive space, number of episodes",
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
    eval_episodes: int = parameters.eval
    predicate_space: bool = not parameters.default_space
    seed: int = parameters.seed
    if isinstance(seed, List):
        seed = seed[0]
    max_depth: bool = parameters.depth
    forest: Optional[List[int]] = parameters.forest
    interactive: Optional[List[int]] = parameters.interactive

    if finite_method != "none":
        from polext.finite import build_tree, FINITE_METHODS

        module = importlib.import_module(
            script_path.replace(".py", "").replace("/", ".")
        )
        states = module.__getattribute__("states")
        predicates = module.__getattribute__("predicates")
        Q_builder = module.__getattribute__("Q_builder")
        Q = Q_builder(model_path)

        space = easy_space(states, Q, predicates, predicate_space)
        if interactive is not None:
            env = module.__getattribute__("make_env")()
            space = interactive_space(
                states, Q, predicates, env, interactive[0], predicate_space
            )

        tree = Leaf(0)
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
            builder = lambda *args, **kwargs: build_forest(
                *args, trees=n_trees, **kwargs
            )

        if eval_episodes > 0:
            env = module.__getattribute__("make_env")()
            mean, diff = eval_q(Q, eval_episodes, env)
            print("Baseline Q-table:")
            print_reward(eval_episodes, mean, diff)
            print()
        methods_todo = []
        if finite_method == "all":
            methods_todo = FINITE_METHODS
        elif "," in finite_method:
            methods_todo = finite_method.split(",")
        else:
            methods_todo = [finite_method]

        for method in methods_todo:
            if method not in FINITE_METHODS:
                print(
                    Text.assemble(
                        ('Finite tree method:"', "red"),
                        method,
                        ('" is unkown, available methods are:', "red"),
                        ", ".join(FINITE_METHODS),
                    ),
                    file=sys.stderr,
                )
                continue
            tree, score = builder(space, max_depth, method, seed=seed)
            print("Method:", Text.assemble((method, "bold")))
            print("Lost Q-Values:", Text.assemble((str(score), FINITE_LOSS_STYLE)))
            if eval_episodes > 0:
                env = module.__getattribute__("make_env")()
                mean, diff = eval_fn(tree, eval_episodes, env)
                print_reward(eval_episodes, mean, diff)
            if isinstance(tree, DecisionTree):
                tree.print()
            print()
    else:
        # TODO: infinite case
        pass
