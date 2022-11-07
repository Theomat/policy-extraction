from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

S = TypeVar("S")


@dataclass
class Predicate(Generic[S]):
    name: str
    f: Callable[[S], bool]

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __call__(self, state: S) -> bool:
        return self.f(state)

    def __hash__(self) -> int:
        return self.name.__hash__()
