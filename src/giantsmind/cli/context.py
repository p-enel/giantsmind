from typing import Optional
from copy import deepcopy
from state import State, states


class CollectionId:
    _value: int

    def __init__(self, value: int):
        self.value = value

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        if not isinstance(new_value, int):
            raise ValueError("CollectionId must be an integer")
        self._value = new_value

    def __repr__(self) -> str:
        return f"CollectionId({self.value})"

    def __int__(self) -> int:
        return self.value


class Context:
    _current_state: State = None
    _papers: Optional[CollectionId] = None
    _previous_state: Optional[State] = None
    _selected_papers: Optional[CollectionId] = None
    _results: Optional[CollectionId] = None
    _search_scope: Optional[CollectionId] = None

    def __init__(
        self,
        current_state: State,
        all_papers_collection: CollectionId,
        previous_state: Optional[State] = None,
    ):
        self.current_state = current_state
        self._all_papers_collection = all_papers_collection
        self._search_scope = all_papers_collection
        self._papers = all_papers_collection

    @property
    def current_state(self) -> State:
        return self._current_state

    @property
    def previous_state(self) -> Optional[State]:
        return self._previous_state

    @property
    def papers(self) -> CollectionId:
        return self._papers

    @property
    def selected_papers(self) -> Optional[CollectionId]:
        return self._selected_papers

    @property
    def results(self) -> Optional[CollectionId]:
        return self._results

    @property
    def search_scope(self) -> Optional[CollectionId]:
        return self._search_scope

    @current_state.setter
    def current_state(self, new_state: State) -> None:
        if not isinstance(new_state, State):
            raise ValueError("State must be an instance of State class")
        if new_state not in states:
            raise ValueError(f"Invalid state: {new_state}")
        if self._current_state:
            self._previous_state = self._current_state
        self._current_state = new_state

    @previous_state.setter
    def previous_state(self, new_state: State) -> None:
        raise ValueError("previous_state is read-only")

    @papers.setter
    def papers(self, new_papers: CollectionId) -> None:
        if not isinstance(new_papers, CollectionId):
            raise ValueError("papers must be an instance of CollectionId")
        self._papers = new_papers

    @selected_papers.setter
    def selected_papers(self, new_papers: CollectionId) -> None:
        if new_papers is not None and not isinstance(new_papers, CollectionId):
            raise ValueError("selected_papers must be an instance of CollectionId")
        self._selected_papers = new_papers

    @results.setter
    def results(self, new_results: CollectionId) -> None:
        if new_results is not None and not isinstance(new_results, CollectionId):
            raise ValueError("results must be an instance of CollectionId")
        self._results = new_results

    @search_scope.setter
    def search_scope(self, new_search_scope: CollectionId) -> None:
        if new_search_scope is not None and not isinstance(new_search_scope, CollectionId):
            raise ValueError("search_scope must be an instance of CollectionId")
        self._search_scope = new_search_scope

    def reset_search_scope(self) -> None:
        self._search_scope = self._all_papers_collection

    def copy(self) -> "Context":
        return deepcopy(self)

    def update(self, to_update) -> "Context":
        new_context = self.copy()
        for key, value in to_update.items():
            setattr(new_context, key, value)
        return new_context

    def __repr__(self) -> str:
        return f"Context(\n    current_state={self.current_state},\n    papers={self.papers},\n    selected_papers={self.selected_papers},\n    results={self.results},\n    search_scope={self.search_scope}\n)"


def context_updater(new_context: Context) -> Context:
    return lambda ctx: ctx.update(new_context)
