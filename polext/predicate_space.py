from typing import (
    Generator,
    Generic,
    Iterable,
    Set,
    Tuple,
    TypeVar,
)
import numpy as np
from polext.predicate import Predicate

S = TypeVar("S")


class PredicateSpace(Generic[S]):
    def __init__(self, predicates: Iterable[Predicate[S]]) -> None:
        self.predicates = list(predicates)
        self.predicate2int = {p: i for i, p in enumerate(self.predicates)}
        self.predicates_set = {p: set() for p in self.predicates}
        self.seen = set()
        self.used = [False for _ in predicates]

    def unused_predicates(self) -> Generator[Predicate[S], None, None]:
        for pred, used in zip(self.predicates, self.used):
            if not used:
                yield pred

    def sat_predicate(self, state: Tuple[bool, ...], predicate: Predicate[S]) -> bool:
        return state[self.predicate2int[predicate]]

    def get_representative(self, state: S, save: bool = True) -> Tuple[bool, ...]:
        repres = tuple(p(state) for p in self.predicates)
        if save:
           self.add_representative(repres)
        return repres

    def add_representative(self, repres: Tuple[bool, ...]):
        if repres not in self.seen:
            self.seen.add(repres)
            for i, p in enumerate(self.predicates):
                if repres[i]:
                    self.predicates_set[p].add(repres)
        return repres

    def states_seen(self) -> Set[Tuple[bool, ...]]:
        return self.seen

    def reset(self) -> None:
        self.seen = set()

    def __iter__(self):
        return self.seen.__iter__()

    def predicate_set_complement(
        self, predicate: Predicate[S]
    ) -> Generator[Tuple[bool, ...], None, None]:
        idx = self.predicate2int[predicate]
        for s in self.seen:
            if not s[idx]:
                yield s

    def split(
        self, predicate: Predicate
    ) -> "Tuple[PredicateSpace[S], PredicateSpace[S]]":
        index = self.predicates.index(predicate)
        positive = PredicateSpace(self.predicates)
        positive.used = self.used[:]
        positive.used[index] = True
        negative = PredicateSpace(self.predicates)
        negative.used = positive.used[:]
        for elem in self.seen:
            target = positive if elem[index] else negative
            for i, p in enumerate(self.predicates):
                if elem[i]:
                    target.predicates_set[p].add(elem)
            target.seen.add(elem)
        return positive, negative

    def random_splits(self, seed: int) -> "Generator[PredicateSpace[S], None, None]":
        states = list(self.seen)
        sample_size = int(np.floor(np.sqrt(len(states))))
        rng = np.random.default_rng(seed)
        while True:
            sub_space = PredicateSpace(self.predicates)
            sub_space.seen = set(
                tuple(x) for x in rng.choice(states, size=sample_size, replace=False)
            )
            # Update predicates_set
            sub_space.predicates_set = {
                p: a.intersection(sub_space.seen)
                for p, a in self.predicates_set.items()
            }
            yield sub_space
