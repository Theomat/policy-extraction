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

    def get_representative(self, state: S) -> Tuple[int, ...]:
        repres = tuple(p(state) for p in self.predicates)
        if repres not in self.seen:
            self.seen.add(repres)
            for i, p in enumerate(self.predicates):
                if repres[i] == 1:
                    self.predicates_set[p].add(repres)
        return repres

    def states_seen(self) -> Set[Tuple[int, ...]]:
        return self.seen

    def reset(self) -> None:
        self.seen = set()

    def split(
        self, predicate: Predicate
    ) -> "Tuple[PredicateSpace[S], PredicateSpace[S]]":
        index = self.predicates.index(predicate)
        new_predicates = self.predicates[:]
        new_predicates.pop(index)
        positive = PredicateSpace(new_predicates)
        negative = PredicateSpace(new_predicates)
        for elem in self.seen:
            new_elem = tuple(x for j, x in enumerate(elem) if j != index)
            target = positive if elem[index] == 1 else negative
            for i, p in enumerate(new_predicates):
                if new_elem[i] == 1:
                    target.predicates_set[p].add(new_elem)
            target.seen.add(new_elem)
        return positive, negative
