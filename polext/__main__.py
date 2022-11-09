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
    parser.add_argument(
        "--finite",
        type=str,
        nargs="?",
        help="finite state space and name of the method to use",
        default="none",
    )
    parser.add_argument("--depth", type=int, default=5, help="max decision tree depth")

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path
    finite_method: str = parameters.finite
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

        if finite_method == "all":
            for method in FINITE_METHODS.keys():
                tree, score = build_tree(states, Q, predicates, max_depth, method)
                print("Method:", method)
                print("Score:", score)
                print(tree)
