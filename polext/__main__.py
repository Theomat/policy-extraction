import importlib

from polext.decision_tree import Leaf

# python -m polext finite/mountain-car/env.py logs/dqn/MountainCar-v0_1/MountainCar-v0.zip --finite all --eval 10


def eval_tree(tree, episodes: int, env) -> float:
    total_reward = 0
    for _ in range(eval_episodes):
        done = False
        state = env.reset()
        while not done:
            state, reward, done, _ = env.step(tree(state))
            total_reward += reward
    return total_reward


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

    # TODO: call method
    if finite_method != "none":
        from polext.finite import build_tree
        from polext.finite.tree_builder import METHODS as FINITE_METHODS

        module = importlib.import_module(
            script_path.replace(".py", "").replace("/", ".")
        )
        states = module.__getattribute__("states")
        predicates = module.__getattribute__("predicates")
        Q_builder = module.__getattribute__("Q_builder")
        Q = Q_builder(model_path)

        tree = Leaf(0)
        if finite_method == "all":
            for method in FINITE_METHODS.keys():
                tree, score = build_tree(states, Q, predicates, max_depth, method)
                print("Method:", method)
                print("Loss:", score)
                if eval_episodes > 0:
                    env = module.__getattribute__("make_env")()
                    total_reward = eval_tree(tree, eval_episodes, env)
                    print(
                        f"Average reward over {eval_episodes} episodes:",
                        total_reward / eval_episodes,
                    )
                print(tree)
        else:
            tree, score = build_tree(states, Q, predicates, max_depth, finite_method)
            print("Loss:", score)
            print(tree)

            if eval_episodes > 0:
                env = module.__getattribute__("make_env")()
                total_reward = eval_tree(tree, eval_episodes, env)
                print(
                    f"Average reward over {eval_episodes} episodes:",
                    total_reward / eval_episodes,
                )
