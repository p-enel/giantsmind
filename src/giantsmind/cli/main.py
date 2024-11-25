from typing import Callable, Dict

from .collection_operations import get_all_papers_collection
from .context import Context
from .state import State, states
from .state_functions import (
    state_func_AFTER_SEARCH,
    state_func_EXIT,
    state_func_INTERACT_WITH_PAPERS,
    state_func_LOAD_COLLECTION,
    state_func_SEARCH_PAPERS,
    state_func_SELECT_PAPERS,
)


def main():
    state_functions: Dict[State, Callable[[Context], Context]] = {
        states.SEARCH_PAPERS: state_func_SEARCH_PAPERS,
        states.AFTER_SEARCH: state_func_AFTER_SEARCH,
        states.INTERACT_WITH_PAPERS: state_func_INTERACT_WITH_PAPERS,
        states.SELECT_PAPERS: state_func_SELECT_PAPERS,
        states.LOAD_COLLECTION: state_func_LOAD_COLLECTION,
        states.EXIT: state_func_EXIT,
    }

    context = Context(
        current_state=states.SEARCH_PAPERS,
        all_papers_collection=get_all_papers_collection(),
    )

    while True:
        current_state: State = context.current_state
        state_function = state_functions.get(current_state)

        if state_function:
            context = state_function(context)
        else:
            raise ValueError(f"Unknown state: {current_state}")
        if current_state == states.EXIT:
            break


if __name__ == "__main__":
    main()
