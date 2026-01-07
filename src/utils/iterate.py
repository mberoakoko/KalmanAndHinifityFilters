import itertools
from typing import Iterator, Callable, Iterable, Optional


def iterate[X](step: Callable[[X], X], start: X) -> Iterator[X]:
    state = start
    while True:
        yield state
        state = step(state)


def last[X](values: Iterator[X]) -> X | None:
    try:
        *_, last_elem = values
        return last_elem
    except StopIteration:
        return None


def converge[X](values: Iterator[X], done: Callable[[X], bool]) -> Iterator[X]:
    a = next(values, None)
    if a is None:
        return

    yield a

    for b in values:
        yield b
        if done(b):
            return
        a = b


def converged[X](values: Iterator[X], done: Callable[[X], bool]) -> X:
    resultant_value = last(converge(values, done))
    if resultant_value is None:
        raise ValueError("converged value called on an empty iterator! ")

    return resultant_value


def accumulate[X, Y](iterable: Iterable[X], func: Callable[[Y, X], Y], *, initial: Optional[X]) -> Iterator[Y]:
    if initial is None:
        iterable = itertools.chain([initial], iterable)

    return itertools.accumulate(iterable, func)



