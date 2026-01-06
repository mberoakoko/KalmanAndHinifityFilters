import typing

def take[A](iterator: typing.Iterator[A], n: int) -> typing.Iterator[A]:
    for _ in range(n):
        yield next(iterator)

def drop[A](iterator: typing.Iterator[A], n: int) -> typing.Iterator[A]:
    for _ in range(n):
        next(iterator)
    return iterator


def pairs[A](iterable: typing.Iterable[A]) -> typing.Iterable[tuple[A, A]]:
    try:
        for item in iterable:
            yield item , next(iterable)
    except StopIteration:
        return
