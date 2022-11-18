import importlib
from typing import List


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
        "-o",
        "--output",
        type=str,
        default="model.csv",
        help="destination CSV file",
    )

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path
    output: str = parameters.output

    module = importlib.import_module(script_path.replace(".py", "").replace("/", "."))
    states = module.__getattribute__("states")
    predicates = module.__getattribute__("predicates")
    Q_builder = module.__getattribute__("Q_builder")
    Q = Q_builder(model_path)

    with open(output, "w") as fd:
        fd.write("#NON-PERMISSIVE\n")
        N = len(states[0])
        fd.write(f"#BEGIN {N} 1\n")
        lines = []
        for state in states:
            q_values: List[float] = Q(state)
            action = q_values.index(max(q_values))
            line = ",".join(map(str, state)) + "," + str(action) + "\n"
            lines.append(line)
        fd.writelines(lines)
