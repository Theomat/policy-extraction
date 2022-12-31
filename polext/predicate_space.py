from typing import (
    Generic,
    Iterable,
    Set,
    Tuple,
    TypeVar,
)

from polext.predicate import Predicate

S = TypeVar("S")


class PredicateSpace(Generic[S]):
    def __init__(self, predicates: Iterable[Predicate[S]]) -> None:
        self.predicates = list(predicates)
        self.predicates_set = {p: set() for p in self.predicates}
        self.seen = set()

    def get_representative(self, state: S, save: bool = True) -> Tuple[bool, ...]:
        repres = tuple(p(state) for p in self.predicates)
        if save and repres not in self.seen:
            self.seen.add(repres)
            for i, p in enumerate(self.predicates):
                if repres[i]:
                    self.predicates_set[p].add(repres)
        return repres

    def states_seen(self) -> Set[Tuple[bool, ...]]:
        return self.seen

    def reset(self) -> None:
        self.seen = set()

    def split(
        self, predicate: Predicate
    ) -> "Tuple[PredicateSpace[S], PredicateSpace[S]]":
        index = self.predicates.index(predicate)
        positive = PredicateSpace(self.predicates)
        negative = PredicateSpace(self.predicates)
        for elem in self.seen:
            target = positive if elem[index] else negative
            for i, p in enumerate(self.predicates):
                if elem[i]:
                    target.predicates_set[p].add(elem)
            target.seen.add(elem)
        return positive, negative
