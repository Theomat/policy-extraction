import json

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import matplotlib.pyplot as plt

import pltpublish as pub

pub.setup()

import numpy as np


aggregate_func = lambda x: np.array(
    [metrics.aggregate_median(x), metrics.aggregate_iqm(x), metrics.aggregate_mean(x)]
)


def easy_plot(scores_dict, filepath="test.png"):
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        scores_dict, aggregate_func, reps=1000
    )
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean"],
        algorithms=list(scores_dict.keys()),
        xlabel="Reward",
    )
    plt.tight_layout()
    plt.savefig(filepath, dpi=500)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="plot_results.py",
        description="Plot the results of an experiment for a given environment",
    )
    parser.add_argument(
        type=str,
        dest="data",
        help="data file",
    )
    parser.add_argument(
        "--iterations", action="store_true", help="show iterations results"
    )

    parameters = parser.parse_args()
    data_file: str = parameters.data
    show_it: bool = parameters.iterations
    import os

    filename = os.path.basename(data_file)
    filename = filename[: filename.rfind(".")]

    # Load data
    with open(data_file) as fd:
        all_data = json.load(fd)

    methods = list(all_data.keys())

    # Remove seeds
    seeds = list(all_data["discrete-dqn"].keys())
    for seed in seeds:
        del all_data[seed]
    methods = list(all_data.keys())
    depths = []
    iterations = []
    # Find base name
    variants = {}
    for method in methods:
        if "-d=" in method and "-it=" in method:
            base_name = method[: method.find("-d=")]
            if base_name not in variants:
                variants[base_name] = []
            variants[base_name].append(method)

            depth = int(method[method.find("-d=") + 3 : method.find("-it=")])
            iteration = int(method[method.find("-it=") + 4 :])

            depths.append(depth)
            iterations.append(iteration)
        if "viper-" in method and "-d=" in method:
            base_name = method[: method.find("-d=")]
            if base_name not in variants:
                variants[base_name] = []
            variants[base_name].append(method)

    iterations = sorted(x for x in iterations if x > 0)
    depths = sorted(depths)

    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.

    # Load all games
    score_dict = {}
    for name, subnames in variants.items():
        for subname in subnames:
            score_dict[subname] = np.array(list(all_data[subname].values())).reshape(
                (-1, 1)
            )
    score_dict["discrete-dqn"] = np.array(
        list(all_data["discrete-dqn"].values())
    ).reshape((-1, 1))
    # score_dict = {key: np.array(list(values.values())).reshape((-1, 1)) for key, values in all_data.items() if key not in seeds}

    # Delete copies of DQN
    saved = False
    for variant in variants["dqn"]:
        if not saved:
            score_dict["dqn"] = score_dict[variant]
            saved = True
        del score_dict[variant]

    if show_it:
        # Plot with respect to iterations
        new_dict = {}
        for method, x in score_dict.items():
            if "-d=" in method and "-it=" in method:
                depth = int(method[method.find("-d=") + 3 : method.find("-it=")])
                if depth == depths[-1]:
                    new_dict[method.replace(f"-d={depth}", "")] = x
            else:
                new_dict[method] = x

        easy_plot(new_dict, f"{filename}_performance_with_iterations.png")

    new_dict = {}
    for method, x in score_dict.items():
        if "-d=" in method and "-it=" in method:
            iteration = int(method[method.find("-it=") + 4 :])
            if iteration == iterations[0]:
                new_dict[method.replace(f"-it={iteration}", "")] = x
        else:
            new_dict[method] = x

    easy_plot(new_dict, f"{filename}_performance_with_depth.png")

    # Load ProcGen scores as a dictionary containing pairs of normalized score
    # matrices for pairs of algorithms we want to compare
    compared_dict = {}
    dqn_perf = score_dict["dqn"]
    compared_dict["discrete-dqn,dqn"] = (score_dict["discrete-dqn"], dqn_perf)
    best_score = {}
    for name in variants:
        if name == "dqn":
            continue
        if "viper-" in name:
            best_score[name] = score_dict[f"{name}-d={depths[-1]}"]
        else:
            if show_it:
                all_scores = [
                    score_dict[f"{name}-d={depths[-1]}-it={it}"] for it in iterations
                ]
                some_list = [
                    (np.mean(s), -np.std(s), i) for i, s in enumerate(all_scores)
                ]
                some_list = sorted(some_list)
                best_score[name] = all_scores[some_list[-1][2]]
            else:
                best_score[name] = score_dict[
                    f"{name}-d={depths[-1]}-it={iterations[0]}"
                ]
        compared_dict[f"{name},dqn"] = (best_score[name], dqn_perf)

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        compared_dict, metrics.probability_of_improvement, reps=1000
    )

    plot_utils.plot_probability_of_improvement(
        average_probabilities, average_prob_cis, figsize=(6, 4)
    )
    plt.tight_layout()
    plt.savefig(f"{filename}_perf_cmp_dqn.png", dpi=500)
    plt.show()

    compared_dict = {}
    discrete_dqn_perf = score_dict["discrete-dqn"]
    compared_dict["dqn,discrete-dqn"] = (dqn_perf, discrete_dqn_perf)

    for name in variants:
        if name == "dqn":
            continue
        compared_dict[f"{name},discrete-dqn"] = (best_score[name], discrete_dqn_perf)

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        compared_dict, metrics.probability_of_improvement, reps=1000
    )
    plot_utils.plot_probability_of_improvement(
        average_probabilities, average_prob_cis, figsize=(8, 4)
    )
    plt.tight_layout()
    plt.savefig(f"{filename}_perf_cmp_discrete_dqn.png", dpi=500)
    plt.show()

    if show_it:
        compared_dict = {}

        for name in variants:
            if name == "dqn":
                continue
            base_score = score_dict[f"{name}-d={depths[-1]}-it=1"]
            all_scores = [
                score_dict[f"{name}-d={depths[-1]}-it={it}"] for it in iterations[1:]
            ]
            some_list = [(np.mean(s), -np.std(s), i) for i, s in enumerate(all_scores)]
            some_list = sorted(some_list)
            best_score = all_scores[some_list[-1][2]]
            compared_dict[f"max({name}-it>1),{name}-it=1"] = (best_score, base_score)

        average_probabilities, average_prob_cis = rly.get_interval_estimates(
            compared_dict, metrics.probability_of_improvement, reps=1000
        )
        plot_utils.plot_probability_of_improvement(
            average_probabilities, average_prob_cis, figsize=(8, 4)
        )
        plt.tight_layout()
        plt.savefig(f"{filename}_perf_impr_iterations.png", dpi=500)
        plt.show()

    compared_dict = {}

    for name in variants:
        if name == "dqn" or "viper-" not in name:
            continue
        base_score = score_dict[f"{name}-d={depths[-1]}"]
        sub_name = name[6:]
        compared_dict[f"viper-{sub_name},{sub_name}"] = (
            base_score,
            best_score[sub_name],
        )
    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        compared_dict, metrics.probability_of_improvement, reps=1000
    )
    plot_utils.plot_probability_of_improvement(
        average_probabilities, average_prob_cis, figsize=(8, 4)
    )
    plt.tight_layout()
    plt.savefig(f"{filename}_perf_impr_viper.png", dpi=500)
    plt.show()
