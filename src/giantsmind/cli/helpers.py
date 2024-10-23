from typing import Tuple, Callable, Any, Iterable
from context import Context


class Options:
    _options: Iterable[Tuple[Any, str, Callable]]

    def __init__(self, options: Iterable[Tuple[Any, str, Any]]):
        self._options = tuple(options)
        self._mapping = {key: action for key, _, action in options}

    def __repr__(self) -> str:
        return "\n".join(f"{key}) {txt} : {action}" for key, txt, action in self._options)

    def __str__(self) -> str:
        return "\n".join(f"{key}) {txt}" for key, txt, _ in self._options)

    def __contains__(self, key: str) -> bool:
        return key in self._mapping

    def __getitem__(self, key: str) -> Callable:
        return self._mapping[key]


def list2optargs(options: Tuple[str, Any]) -> Options:
    return [(str(i + 1), option, action) for i, (option, action) in enumerate(options)]


def get_choice(options: Options) -> str:
    if not isinstance(options, Options):
        raise ValueError("options must be an instance of Options class")
    print("Choose from the following options:")
    input_options = input(str(options) + "\n")
    return input_options


def choice_handler(choice: str, options: Options, context) -> Context:
    try:
        action = options[choice]
    except KeyError:
        print("Invalid choice, please try again.")
        return context
    return action(context)


def state_handler(context: Context, options: Options) -> Context:
    choice = get_choice(options)
    new_context = choice_handler(choice, options, context)
    return new_context
