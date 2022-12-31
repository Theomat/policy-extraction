from typing import Optional, Tuple, TypeVar

import numpy as np

from polext.decision_tree import DecisionTree, Node, Leaf
from polext.predicate import Predicate
from polext.predicate_space import PredicateSpace
from polext.q_values_learner import QValuesLearner
from polext.tree_builder import register, TreeBuildingAlgo, tree_loss

S = TypeVar("S")


def greedy_finisher(
    space: PredicateSpace[S], Qtable: QValuesLearner, **kwargs
) -> Optional[DecisionTree[S]]:
    best_score = 1e99
    answer = None
    for action in range(Qtable.nactions):
        pred, baction, _ = greedy_q_selection(space, Qtable, 2, action, **kwargs)
        tree = Node(pred, Leaf(baction), Leaf(action)) if pred else Leaf(action)
        score = tree_loss(tree, space, Qtable)
        if score < best_score:
            best_score = score
            answer = tree
    return answer


def greedy_tree_builder(fn, build_qmax: bool = False):
    def f(
        space: PredicateSpace[S], qtable: QValuesLearner, max_depth: int, **kwargs
    ) -> DecisionTree[S]:
        best = Leaf(0)
        best_loss = 1e99
        Qmax = {}
        if build_qmax:
            for s in space.seen:
                Qval = qtable[s]
                Qmax[s] = np.max(Qval)
        for action in range(qtable.nactions):
            tree = __rec_tree__(
                fn, space, qtable, max_depth, action, Qmax=Qmax
            ).simplified()
            lost_reward = tree_loss(tree, space, qtable)
            if lost_reward < best_loss:
                best = tree
                best_loss = lost_reward
        return best

    return f


def __split__(
    fn,
    best_predicate: Optional[Predicate[S]],
    best_action: int,
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    depth_left: int,
    previous_action: int,
    **kwargs
) -> DecisionTree[S]:
    if best_predicate is None:
        return Leaf(previous_action)
    else:
        pos, neg = space.split(best_predicate)
        left = __rec_tree__(fn, pos, Qtable, depth_left - 1, best_action, **kwargs)
        right = __rec_tree__(fn, neg, Qtable, depth_left - 1, previous_action, **kwargs)
        return Node(best_predicate, left, right)


def __rec_tree__(
    fn,
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    depth_left: int,
    previous_action: int,
    **kwargs
) -> DecisionTree[S]:
    if depth_left == 1:
        return Leaf(previous_action)
    # Find best split
    best_predicate, best_action, previous_action = fn(
        space, Qtable, depth_left, previous_action, **kwargs
    )

    return __split__(
        fn,
        best_predicate,
        best_action,
        space,
        Qtable,
        depth_left,
        previous_action,
        **kwargs
    )


def greedy_q_selection(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    depth_left: int,
    previous_action: int,
    **kwargs
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_predicate = None
    best_action = previous_action
    best_score = sum(Qtable[s][previous_action] for s in space.states_seen())
    for candidate, sub_states in space.predicates_set.items():
        part = sub_states
        free_score = sum(
            Qtable[s][previous_action] for s in space.seen if s not in part
        )
        for action in range(Qtable.nactions):
            if action == previous_action:
                continue
            score = sum(Qtable[s][action] for s in part) + free_score
            if score > best_score:
                best_predicate = candidate
                best_action = action
                best_score = score
    return best_predicate, best_action, previous_action


def greedy_opt_action_selection(
    space: PredicateSpace[S],
    Qtable: QValuesLearner,
    depth_left: int,
    previous_action: int,
    Qmax: dict,
    **kwargs
) -> Tuple[Optional[Predicate[S]], int, int]:
    best_predicate = None
    best_action = previous_action
    n_before = sum(Qmax[s] <= Qtable[s][previous_action] for s in space.seen)
    best_score = n_before
    for candidate, sub_states in space.predicates_set.items():
        # print("\tcandidate:", candidate)
        part = sub_states
        n_candidate = sum(
            1
            for s in space.seen
            if s not in part and Qmax[s] <= Qtable[s][previous_action]
        )
        for action in range(Qtable.nactions):
            if action == previous_action:
                continue
            n_after = sum(1 for s in part if Qmax[s] <= Qtable[s][action]) + n_candidate
            score = n_after
            # print("\t\taction:", action, "score:", score)
            if score > best_score:
                best_predicate = candidate
                best_action = action
                best_score = score
    return best_predicate, best_action, previous_action


register(TreeBuildingAlgo("greedy-q", greedy_tree_builder(greedy_q_selection)))
register(
    TreeBuildingAlgo(
        "greedy-nactions", greedy_tree_builder(greedy_opt_action_selection, True)
    )
)
