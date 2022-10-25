if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="polext", description="Extract a Decision Tree from a DQN"
    )
    parser.add_argument(
        type=str,
        dest="script_path",
        action="store",
        help="path to python script for the algorithm",
    )
    parser.add_argument(
        type=str,
        dest="model_path",
        default="",
        help="DQN model file to load",
    )

    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path

    # TODO: call method
