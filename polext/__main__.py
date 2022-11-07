import importlib


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
    parser.add_argument("--finite", action="store_true", help="finite state space")
    parser.add_argument("--depth", type=int, default=5, help="max decision tree depth")

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path
    finite: bool = parameters.finite
    max_depth: bool = parameters.depth

    # TODO: call method
    if finite:
        from polext.finite import build_tree

        module = importlib.import_module(
            script_path.replace(".py", "").replace("/", ".")
        )
        states = module.__getattribute__("states")
        predicates = module.__getattribute__("predicates")
        Q_builder = module.__getattribute__("Q_builder")
        Q = Q_builder(script_path)

        tree = build_tree(states, Q, predicates, max_depth)
        print(tree)
