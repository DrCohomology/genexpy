"""Helpers for the kernels package."""


def all_subclasses(cls) -> set:
    """All direct and indirect subclasses of `cls`."""
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])
