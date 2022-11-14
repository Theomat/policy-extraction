from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

from polext.decision_tree import DecisionTree
from polext.finite.greedy_builder import (
    greedy_tree_builder,
    greedy_q_selection,
    greedy_opt_action_selection,
)
from polext.finite.max_probability_tree_builder import max_probability_tree_builder
from polext.finite.optimistic_tree_builder import optimistic_tree_builder
from polext.finite.simulated_annealing_tree_builder import (
    simulated_annealing_tree_builder,
)
from polext.predicate import Predicate

S = TypeVar("S")


def tree_loss(
    tree: DecisionTree[S], Qtable: Dict[S, List[float]], states: Iterable[S]
) -> float:
    lost_reward = sum(max(Qtable[s]) - Qtable[s][tree(s)] for s in states)
    return lost_reward


_METHODS_ = {
    "greedy-q": greedy_tree_builder(greedy_q_selection, tree_loss),
    "greedy-nactions": greedy_tree_builder(greedy_opt_action_selection, tree_loss),
    "max-probability": max_probability_tree_builder(tree_loss),
    "optimistic": optimistic_tree_builder(tree_loss),
    "simulated-annealing": simulated_annealing_tree_builder(tree_loss),
}


def build_tree(
    states: List[S],
    Q: Callable[[S], List[float]],
    predicates: List[Predicate[S]],
    max_depth: int,
    method: str,
) -> Tuple[DecisionTree[S], float]:
    Qtable = {s: Q(s) for s in states}
    predicates_table = {
        predicate: {s for s in states if predicate(s)} for predicate in predicates
    }
    nactions = len(Qtable[states[0]])
    tree = _METHODS_[method.lower().strip()](
        set(states), Qtable, predicates_table, nactions, max_depth
    ).simplified()
    return tree, tree_loss(tree, Qtable, states)
