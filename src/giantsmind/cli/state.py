from typing import Literal

allowed_states = [
    "SEARCH_PAPERS",
    "AFTER_SEARCH",
    "INTERACT_WITH_PAPERS",
    "SELECT_PAPERS",
    "LOAD_COLLECTION",
    "EXIT",
]


class State:
    _value: Literal[allowed_states]

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return self.value

    def __eq__(self, other: "State") -> bool:
        if not isinstance(other, State):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


class States:
    def __init__(self, allowed_states: list[str]):
        for state in allowed_states:
            setattr(self, state, State(state))
        self._allowed_states = allowed_states

    def __contains__(self, state: State) -> bool:
        return state.value in self._allowed_states


states = States(allowed_states)
