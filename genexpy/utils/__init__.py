"""
Utilities for rankings.

- ``rankings``: rankings as adjacency matrices, samples and multi-samples of rankings, and the conversion of a
  dataframe of experimental results into rankings.
- ``relations``: conversion of scores into rank vectors.
"""
from . import rankings, relations

__all__ = ["rankings", "relations"]
