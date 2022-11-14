import importlib
import numpy as np
from polext.decision_tree import Leaf


def eval_tree(tree, episodes: int, env) -> tuple[float, float]:
    total_rewards = []
    for _ in range(episodes):
        total_reward = 0
        done = False
        state = env.reset()
        while not done:
            state, reward, done, _ = env.step(tree(state))
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), 2 * np.std(total_rewards)


def eval_q(Q, episodes: int, env) -> tuple[float, float]:
    total_rewards = []
    for _ in range(episodes):
        total_reward = 0
        done = False
        state = env.reset()
        while not done:
            action = np.argmax(Q(state))
            if isinstance(action, np.ndarray):
                action = action[0]
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), 2 * np.std(total_rewards)


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
    max_depth: bool = parameters.depth

    if finite_method != "none":
        from polext.finite import build_tree, FINITE_METHODS

        module = importlib.import_module(
            script_path.replace(".py", "").replace("/", ".")
        )
        states = module.__getattribute__("states")
        predicates = module.__getattribute__("predicates")
        Q_builder = module.__getattribute__("Q_builder")
        Q = Q_builder(model_path)

        tree = Leaf(0)
        if eval_episodes > 0:
            env = module.__getattribute__("make_env")()
            mean, diff = eval_q(Q, eval_episodes, env)
            print("Baseline Q-table:")
            print(
                f"95% of rewards over {eval_episodes} episodes fall into: [{mean- diff:.2f};{mean + diff:.2f}] (mean={mean})"
            )
            print()
        if finite_method == "all":
            for method in FINITE_METHODS:
                tree, score = build_tree(states, Q, predicates, max_depth, method)
                print("Method:", method)
                print("Lost Q-Values:", score)
                if eval_episodes > 0:
                    env = module.__getattribute__("make_env")()
                    mean, diff = eval_tree(tree, eval_episodes, env)
                    print(
                        f"95% of rewards over {eval_episodes} episodes fall into: [{mean- diff:.2f};{mean + diff:.2f}] (mean={mean})"
                    )
                tree.print()
                print()
        else:
            tree, score = build_tree(states, Q, predicates, max_depth, finite_method)
            print("Lost Q-Values:", score)
            tree.print()

            if eval_episodes > 0:
                env = module.__getattribute__("make_env")()
                mean, diff = eval_tree(tree, eval_episodes, env)
                print(
                    f"95% of rewards over {eval_episodes} episodes fall into: [{mean- diff:.2f};{mean + diff:.2f}] (mean={mean})"
                )
    else:
        # TODO: infinite case
        pass
