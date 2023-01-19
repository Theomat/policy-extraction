from collections import defaultdict
from typing import Dict, Set, Tuple, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree, Node, Leaf
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.tree_builder import register, TreeBuildingAlgo

S = TypeVar("S")


def gini_tree_builder(
    space: PredicateSpace[S], Qtable: QValuesLearner, max_depth: int, **kwargs
) -> DecisionTree[S]:
    action_set = defaultdict(set)
    for s in space:
        a = np.argmax(Qtable[s])
        action_set[a].add(s)

    return __builder__(space, action_set, max_depth, __gini_score__(action_set))


def __gini_score__(actions: Dict[int, Set[Tuple[bool, ...]]]) -> float:
    score = 0
    len_a = {a: len(s) for a, s in actions.items()}
    total = 0
    for a in len_a:
        score += len_a[a] ** 2
        total += len_a[a]
    return 1 - score / max(1, total * total)


def __leaf__(actions: Dict[int, Set[Tuple[bool, ...]]]) -> Leaf[S]:
    index2action = list(actions.keys())
    to_choose = [len(actions[a]) for a in index2action]
    i = np.argmax(to_choose)
    return Leaf(index2action[i])


def __builder__(
    space: PredicateSpace[S],
    actions: Dict[int, Set[Tuple[bool, ...]]],
    max_depth: int,
    parent_gini: float,
) -> DecisionTree[S]:
    if max_depth == 0 or len(space.seen) <= 1:
        return __leaf__(actions)
    found_split = False
    split = None
    data = None
    for predicate in space.unused_predicates():
        left_a = {
            a: {s for s in subset if space.sat_predicate(s, predicate)}
            for a, subset in actions.items()
        }
        total_len_left = sum(len(s) for s in left_a.values())
        left_gini = __gini_score__(left_a)
        right_a = {
            a: {s for s in subset if not space.sat_predicate(s, predicate)}
            for a, subset in actions.items()
        }
        right_gini = __gini_score__(left_a)
        total_len_right = sum(len(s) for s in right_a.values())
        split_gini = (left_gini * total_len_left + right_gini * total_len_right) / (
            total_len_right + total_len_left
        )
        print("\t splti with", predicate, "obtained Gini:", split_gini, "parent:", parent_gini)
        if split_gini < parent_gini:
            found_split = True
            split = predicate
            data = (left_a, left_gini, right_a, right_gini)

    if not found_split:
        print("no further spit found:", parent_gini)
        return __leaf__(actions)
    (left_a, left_gini, right_a, right_gini) = data  # type: ignore
    left_space, right_space = space.split(split)  # type: ignore
    left = __builder__(left_space, left_a, max_depth - 1, left_gini)
    right = __builder__(right_space, right_a, max_depth - 1, right_gini)
    return Node(split, left, right)  # type: ignore


register(TreeBuildingAlgo("gini", gini_tree_builder), True)
