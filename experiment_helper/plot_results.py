import json

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import matplotlib.pyplot as plt

import pltpublish as pub

pub.setup()

import numpy as np

PTE_ITERATION = 2
QUOTIENT_DQN = "Quotient MDP DQN"
DQN = "DQN"
VIPER = "VIPER"
renaming = [("discrete-dqn", QUOTIENT_DQN)]
method_renaming = [
    ("dqn", DQN),
    ("greedy-q", "Greedy Values"),
    ("optimistic", "Optimistic"),
    ("max-probability", "Max Probability"),
    ("greedy-nactions", "Greedy Optimal Actions"),
    ("entropy", "Adapted Entropy"),
    ("gini", "Gini Index"),
]
prefix_renaming = [("viper", VIPER), ("", "PTE ")]

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

    parameters = parser.parse_args()
    data_file: str = parameters.data
    import os

    filename = os.path.basename(data_file)
    filename = filename[: filename.rfind(".")]

    # Load data
    with open(data_file) as fd:
        all_data = json.load(fd)
        for old_name, new_name in renaming:
            if old_name in all_data:
                all_data[new_name] = all_data[old_name]
                del all_data[old_name]
        # Remove seeds
        seeds = list(all_data[QUOTIENT_DQN].keys())
        for seed in seeds:
            del all_data[seed]
        # Finish renaming
        for key in list(all_data.keys()):
            dst_name: str = key
            for old_name, new_name in method_renaming:
                if f"{old_name}-d=" in dst_name:
                    dst_name = dst_name.replace(f"{old_name}-d=", f"{new_name}-d=")
                    break
            for old_name, new_name in prefix_renaming:
                if dst_name.startswith(old_name) and DQN not in dst_name:
                    dst_name = new_name + dst_name[len(old_name) :]
                    break
            if dst_name != key:
                all_data[dst_name] = all_data[key]
                del all_data[key]

    methods = list(all_data.keys())
    depths = []
    iterations = []
    # Find base name
    variants = {}
    for method in methods:
        if f"{VIPER}-" in method and "-d=" in method:
            base_name = method[: method.find("-d=")]
            if base_name not in variants:
                variants[base_name] = []
            variants[base_name].append(method)
        elif "-d=" in method:
            base_name = method[: method.find("-d=")]
            if base_name not in variants:
                variants[base_name] = []
            variants[base_name].append(method)

            depth = int(method[method.find("-d=") + 3 :])

            if depth not in depths:
                depths.append(depth)


    iterations = sorted(x for x in iterations if x > 0)
    depths = sorted(depths)

    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.

    # Load all games
    score_dict = {}
    for name, subnames in variants.items():
        for subname in subnames:
            score_dict[subname] = np.array([ x for x,y in all_data[subname].values()]).reshape(
                (-1, 1)
            )
    score_dict[QUOTIENT_DQN] = np.array([ x for x,y in all_data[QUOTIENT_DQN].values()]).reshape(
        (-1, 1)
    )

    # Delete copies of DQN
    saved = False
    for variant in variants[DQN]:
        if not saved:
            score_dict[DQN] = score_dict[variant]
            saved = True
        del score_dict[variant]

    # Plot Global performances
    new_dict = {}
    for method, x in score_dict.items():
        if "-d=" in method:
            if "-it=" in method:
                iteration = int(method[method.find("-it=") + 4 :])
                depth = int(method[method.find("-d=") + 3 : method.find("-it=")])
                if iteration != PTE_ITERATION:
                    continue
                method = method.replace(f"-it={iteration}", "")
            else:
                depth = int(method[method.find("-d=") + 3 :])
            if depth == depths[-1]:
                new_dict[method.replace(f"-d={depth}", "").replace("-", " ")] = x

        else:
            new_dict[method.replace("-", " ")] = x

    to_ignore = ["PTE Gini Index"]
    for x in to_ignore:
        if x in new_dict:
            del new_dict[x]

    easy_plot(new_dict, f"{filename}_performance.png")

    # Plot performances w.r.t. depth
    new_dict = {}
    for method, x in score_dict.items():
        if DQN in method:
            continue
        if "-d=" in method and "-it=" in method:
            iteration = int(method[method.find("-it=") + 4 :])
            if iteration == PTE_ITERATION:
                new_dict[method.replace(f"-it={iteration}", "").replace("-", " ")] = x
        else:
            new_dict[method.replace("-", " ")] = x

    easy_plot(new_dict, f"{filename}_performance_with_depth.png")

    # Load ProcGen scores as a dictionary containing pairs of normalized score
    # matrices for pairs of algorithms we want to compare
    compared_dict = {}
    dqn_perf = score_dict[DQN]
    compared_dict[f"{QUOTIENT_DQN},{DQN}"] = (score_dict[QUOTIENT_DQN], dqn_perf)
    best_score = {}
    for name in variants:
        if name == DQN:
            continue
        if f"{VIPER}-" in name:
            best_score[name] = score_dict[f"{name}-d={depths[-1]}"]
        else:
            best_score[name] = score_dict[f"{name}-d={depths[-1]}-it={PTE_ITERATION}"]
        compared_dict[f"{name.replace('-', ' ')},{DQN}"] = (best_score[name], dqn_perf)

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
    discrete_dqn_perf = score_dict[QUOTIENT_DQN]
    compared_dict[f"{DQN},{QUOTIENT_DQN}"] = (dqn_perf, discrete_dqn_perf)

    for name in variants:
        if name == DQN:
            continue
        compared_dict[f"{name.replace('-', ' ')},{QUOTIENT_DQN}"] = (
            best_score[name],
            discrete_dqn_perf,
        )

    average_probabilities, average_prob_cis = rly.get_interval_estimates(
        compared_dict, metrics.probability_of_improvement, reps=1000
    )
    plot_utils.plot_probability_of_improvement(
        average_probabilities, average_prob_cis, figsize=(8, 4)
    )
    plt.tight_layout()
    plt.savefig(f"{filename}_perf_cmp_discrete_dqn.png", dpi=500)
    plt.show()

    if len(iterations) > 1:
        compared_dict = {}

        for name in variants:
            if name == DQN or f"{VIPER}-" in name:
                continue
            base_score = score_dict[f"{name}-d={depths[-1]}-it=1"]
            pte_score = score_dict[f"{name}-d={depths[-1]}-it=2"]
            compared_dict[f"{name},{name.replace('PTE', 'E')}"] = (
                pte_score,
                base_score,
            )

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
        if name == DQN or f"{VIPER}-" not in name:
            continue
        base_score = score_dict[f"{name}-d={depths[-1]}"]
        sub_name = name[6:]
        compared_dict[f"{VIPER} {sub_name},{sub_name}"] = (
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
