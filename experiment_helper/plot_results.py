import json
import matplotlib.pyplot as plt

import numpy as np
import pltpublish as pub

from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D

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

    # Load data
    with open(data_file) as fd:
        all_data = json.load(fd)

    pub.setup()

    methods = list(all_data.keys())

    # Remove seeds
    seeds = list(all_data["discrete-dqn"].keys())
    for seed in seeds:
        del all_data[seed]
    methods = list(all_data.keys())
    # Find base name
    variants = {}
    for method in methods:
        if "-d=" in method and "-it=" in method:
            base_name = method[: method.find("-d=")]
            if base_name not in variants:
                variants[base_name] = []
            variants[base_name].append(method)

    symbols = ["o", "P", "D", "*", "x"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    it2index = []
    depth2index = []

    xlabels = []

    last_x = 3
    for base_name in variants.keys():
        xlabels.append(base_name)
        for variant in variants[base_name]:
            # Parameters of variant
            depth = int(variant[variant.find("-d=") + 3 : variant.find("-it=")])
            iterations = int(variant[variant.find("-it=") + 4 :])
            if iterations not in it2index:
                it2index.append(iterations)
            it_index = it2index.index(iterations)
            if depth not in depth2index:
                depth2index.append(depth)
            depth_index = depth2index.index(depth)
            # Compute points
            x = []
            y = []
            for seed in seeds:
                x.append(last_x)
                y.append(all_data[variant][seed])
            if base_name == "dqn":
                plt.scatter(
                    [1 for _ in y],
                    y,
                    c="black",
                    alpha=0.5,
                    marker=MarkerStyle("+", fillstyle="none"),
                )
                last_x -= 1
                break
            plt.scatter(
                x,
                y,
                marker=MarkerStyle(symbols[it_index], fillstyle="none"),
                c=colors[depth_index],
                alpha=0.5,
            )
            # plt.scatter([x[0]], [np.mean(y)], marker=symbols[it_index], c=colors[depth_index], alpha=1)
        last_x += 1

    xlabels.insert(1, "discrete-dqn")
    y = []
    for seed in seeds:
        y.append(all_data["discrete-dqn"][seed])
    plt.scatter(
        [2 for _ in y],
        y,
        c="black",
        alpha=0.5,
        marker=MarkerStyle("+", fillstyle="none"),
    )

    legend_elements = []
    for depth_index, depth in enumerate(depth2index):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=colors[depth_index],
                lw=0,
                markersize=10,
                marker="o",
                label=f"depth = {depth}",
            )
        )
    for it_index, iteration in enumerate(it2index):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=0,
                label=f"iterations = {iteration}",
                marker=MarkerStyle(symbols[it_index], fillstyle="none"),
            )
        )
    plt.xticks(ticks=list(range(1, len(xlabels) + 1)), labels=xlabels, rotation=45)
    plt.ylabel("Reward")
    plt.xlabel("Method")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.2, -0.1)).set_draggable(True)
    plt.show()
