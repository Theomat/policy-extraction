from collections import defaultdict
from typing import Dict, Generic, Iterable, Optional, Tuple, TypeVar, Union

from polext.predicate_space import PredicateSpace
from polext.decision_tree import DecisionTree

S = TypeVar("S")


class Forest(Generic[S]):
    def __init__(self, trees: Iterable[DecisionTree[S]]) -> None:
        self.trees = list(trees)

    def eval(self, state: S) -> int:
        return majority_vote(self.votes(state))

    def eval_pred_space(self, state: Tuple[bool, ...], space: PredicateSpace[S]) -> int:
        return majority_vote(self.votes_pred_space(state, space))

    def __call__(
        self,
        state: Union[S, Tuple[bool, ...]],
        space: Optional[PredicateSpace[S]] = None,
    ) -> int:
        if space is None:
            return self.eval(state)  # type: ignore
        return self.eval_pred_space(state, space)  # type: ignore

    def votes(self, state: S) -> Dict[int, int]:
        votes = defaultdict(int)
        for tree in self.trees:
            votes[tree.eval(state)] += 1
        return votes

    def votes_pred_space(
        self, state: Tuple[bool, ...], space: PredicateSpace[S]
    ) -> Dict[int, int]:
        votes = defaultdict(int)
        for tree in self.trees:
            votes[tree.eval_pred_space(state, space)] += 1
        return votes

    def seed(self, seed: Optional[int]) -> None:
        for i, tree in enumerate(self.trees):
            tree.seed(seed + i)


def majority_vote(votes: Dict[int, int]) -> int:
    max_votes = -1
    maxi = 0
    for action, vote in votes.items():
        if vote > max_votes:
            max_votes = vote
            maxi = action
    return maxi


def vote_uncertainty(votes: Dict[int, int], choice: int) -> float:
    """
    Return the ratio of votes who did not vote for the choice
    """
    total = 0
    disagree = 0
    for action, vote in votes.items():
        total += vote
        if action != choice:
            disagree += vote
    return disagree / total
