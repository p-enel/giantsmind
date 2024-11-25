from typing import Any, Callable, List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def get_exist_absent(
    items: List[T], predicate: Callable[[List[T]], List[bool]]
) -> Tuple[List[T], List[int], List[T], List[int]]:
    """
    Split a list into existing and absent items based on a predicate function.

    Args:
        items: Input list of items
        predicate: Function that returns boolean flags for each item

    Returns:
        Tuple of (existing items, existing indices, absent items, absent indices)
    """
    exist_flags = predicate(items)

    # Get existing items and their indices
    exists = [(i, item) for i, (item, flag) in enumerate(zip(items, exist_flags)) if flag]
    exist_indices, exist_items = zip(*exists) if exists else ([], [])

    # Get absent items and their indices in one pass
    absents = [(i, item) for i, (item, flag) in enumerate(zip(items, exist_flags)) if not flag]
    absent_indices, absent_items = zip(*absents) if absents else ([], [])

    return list(exist_items), list(exist_indices), list(absent_items), list(absent_indices)


def reorder_merge_lists(
    docs1: Sequence[T], docs2: Sequence[T], index1: Sequence[int], index2: Sequence[int]
) -> List[T]:
    """
    Merge two sequences into a new list based on provided target indices.

    Args:
        docs1: First sequence of items
        docs2: Second sequence of items
        index1: Target indices for items from docs1
        index2: Target indices for items from docs2

    Returns:
        New list with merged items at specified positions

    Raises:
        ValueError: If indices are invalid or overlapping
    """
    if len(docs1) != len(index1) or len(docs2) != len(index2):
        raise ValueError("Length mismatch between docs and indices")

    if set(index1) & set(index2):
        raise ValueError("Overlapping indices detected")

    result_size = max(max(index1, default=-1), max(index2, default=-1)) + 1
    result = [None] * result_size

    # Use dictionary comprehension for more efficient assignment
    updates = {**{i: doc for i, doc in zip(index1, docs1)}, **{i: doc for i, doc in zip(index2, docs2)}}

    for i, doc in updates.items():
        result[i] = doc

    return result
