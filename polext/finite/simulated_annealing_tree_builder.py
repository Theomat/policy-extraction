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
    random: random.Random,
    predicates_table: Dict[Predicate[S], Set[S]],
    nactions: int,
    max_depth: int,
) -> Tuple[DecisionTree[S], int]:
    leaf = random.randint(1, leaves)

    def map_tree(
        tree: DecisionTree[S],
        n: int,
        pred: Predicate[S],
        new_action: int,
        depth: int = 0,
    ) -> Tuple[DecisionTree[S], int, int]:
        if isinstance(tree, Node):
            left, n, l1 = map_tree(tree.left, n, pred, new_action, depth + 1)
            right, n, l2 = map_tree(tree.right, n, pred, new_action, depth + 1)
            return Node(tree.predicate, left, right), n, l1 + l2
        else:
            if depth < max_depth:
                n -= 1
                if n == 0:
                    return (
                        Node(pred, Leaf(new_action), tree),
                        n,
                        2 if depth + 1 < max_depth else 0,
                    )
                else:
                    return tree, n, 1
            else:
                return tree, n, 0

    preds = list(predicates_table.keys())
    predicate = preds[random.randint(0, len(preds) - 1)]
    new_action = random.randint(0, nactions - 1)
    tree, _, leaves = map_tree(tree, leaf, predicate, new_action)
    return tree, leaves


def __real_to_zero_one__(x: float) -> float:
    return 2 * np.arctan(x) / np.pi + 1


def simulated_annealing_tree_builder(loss):
    def f(
        states: Set[S],
        Qtable: Dict[S, List[float]],
        predicates_table: Dict[Predicate[S], Set[S]],
        nactions: int,
        max_depth: int,
    ) -> DecisionTree[S]:
        gen = random.Random(127)

        tries = max_depth * max_depth * 50

        tree = Leaf(0)
        leaves = 1
        best_loss = __real_to_zero_one__(loss(tree, Qtable, states))
        beta = 200
        temp = 100
        alpha = 0.9
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(tries):
                neighbour, new_leaves = __pick_neighbour__(
                    tree, leaves, gen, predicates_table, nactions, max_depth
                )
                n_loss = __real_to_zero_one__(loss(neighbour, Qtable, states))
                keep_prob = min(1, np.exp(beta * (n_loss - best_loss) / temp))
                if gen.random() <= keep_prob:
                    tree = neighbour
                    leaves = new_leaves
                    best_loss = n_loss
                    if new_leaves == 0:
                        break
                temp = temp / (1 + alpha)

        return tree

    return f
