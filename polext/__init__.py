if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(
        description="Extract a Decision Tree from a DQN"
    )
    parser.add_argument(
        type=str,
        dest="script_path",
        action="store",
        help="API file",
    )
    parser.add_argument(
        type=str,
        dset="model_path",
        default="",
        help="model file to load",
    )


    parameters = parser.parse_args()
    script_path: str = parameters.script_path
    model_path: str = parameters.model_path

    #TODO: call method