from typing import Dict, List, Set, Tuple, TypeVar
import random
import warnings


import numpy as np

from polext.decision_tree import DecisionTree, Node, Leaf
from polext.predicate import Predicate

S = TypeVar("S")


def __pick_neighbour__(
    tree: DecisionTree[S],
    leaves: int,
    nodes: int,
    gen: random.Random,
    predicates_table: Dict[Predicate[S], Set[S]],
    nactions: int,
    max_depth: int,
    probability_swap: float,
) -> Tuple[DecisionTree[S], int]:
    preds = list(predicates_table.keys())
    predicate = preds[gen.randint(0, len(preds) - 1)]
    new_action = gen.randint(0, nactions - 1)

    # Either change a node
    if (gen.random() <= probability_swap and nodes > 0) or leaves == 0:
        random_node = gen.randint(1, nodes + leaves)

        def map_tree_node(
            tree: DecisionTree[S],
            n: int,
        ) -> Tuple[DecisionTree[S], int]:
            if isinstance(tree, Node):
                left, n = map_tree_node(tree.left, n)
                right, n = map_tree_node(tree.right, n)
                n -= 1
                if n == 0:
                    return Node(predicate, left, right), n
                else:
                    return Node(tree.predicate, left, right), n
            else:
                n -= 1
                if n == 0:
                    return Leaf(new_action), n
                else:
                    return tree, n

        tree, _ = map_tree_node(tree, random_node)
        return tree, leaves
    else:
        # Or split a leaf
        leaf = gen.randint(1, leaves)

        def map_tree(
            tree: DecisionTree[S],
            n: int,
            depth: int = 0,
        ) -> Tuple[DecisionTree[S], int, int]:
            if isinstance(tree, Node):
                left, n, l1 = map_tree(tree.left, n, depth + 1)
                right, n, l2 = map_tree(tree.right, n, depth + 1)
                return Node(tree.predicate, left, right), n, l1 + l2
            else:
                if depth < max_depth:
                    n -= 1
                    if n == 0:
                        return (
                            Node(predicate, Leaf(new_action), tree),
                            n,
                            2 if depth + 1 < max_depth else 0,
                        )
                    else:
                        return tree, n, 1
                else:
                    return tree, n, 0

        tree, _, leaves = map_tree(tree, leaf)
        return tree, leaves


def __real_to_zero_one__(x: float) -> float:
    return 2 * np.arctan(x) / np.pi + 1


def simulated_annealing_tree_builder(
    loss,
    probability_swap: float,
    beta: float,
    temperature: float,
    alpha: float,
    tries_factor: int,
):
    def f(
        states: Set[S],
        Qtable: Dict[S, List[float]],
        predicates_table: Dict[Predicate[S], Set[S]],
        nactions: int,
        max_depth: int,
    ) -> DecisionTree[S]:
        gen = random.Random(127)

        tries = max_depth * max_depth * tries_factor

        tree = Leaf(0)
        leaves = 1
        nodes = 0
        temp = temperature
        best_loss = __real_to_zero_one__(loss(tree, Qtable, states))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(tries):
                neighbour, new_leaves = __pick_neighbour__(
                    tree,
                    leaves,
                    nodes,
                    gen,
                    predicates_table,
                    nactions,
                    max_depth,
                    probability_swap,
                )
                n_loss = __real_to_zero_one__(loss(neighbour, Qtable, states))
                keep_prob = min(1, np.exp(beta * (n_loss - best_loss) / temp))
                if gen.random() <= keep_prob:
                    tree = neighbour
                    if new_leaves != leaves:
                        nodes += 1
                    leaves = new_leaves
                    best_loss = n_loss
                temp = temp / (1 + alpha)

        return tree

    return f
