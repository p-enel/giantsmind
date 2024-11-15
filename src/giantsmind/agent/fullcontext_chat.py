from typing import Callable, Dict, Tuple
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from giantsmind.core.data_management import get_context_from_collection
from giantsmind import agent
from giantsmind.utils import utils

SYSTEM_MESSAGE_TEMPLATE = "<system> {sys_txt} </system>\n{paper_context}"
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"
DEFAULT_SYS_MSG_NAME = "full_context.txt"


def generate_sys_message(sys_txt: str, collection_name: str) -> str:
    paper_context = get_context_from_collection(collection_name)
    context = SYSTEM_MESSAGE_TEMPLATE.format(sys_txt=sys_txt, paper_context=paper_context)
    return context


class ChatFullContext:
    _model: BaseChatModel

    def __init__(
        self,
        sys_msg_name: str,
        chat_model: BaseChatModel,
        model: str,
        generate_full_sys_msg: Callable,
        gen_ctx_args: Tuple[Tuple, Dict],
    ) -> None:
        self.init_chat(chat_model, model)
        self._generate_context = generate_full_sys_msg
        self._gen_ctx_args = gen_ctx_args
        sys_msg_path = self.get_sys_msg_path(sys_msg_name)
        self.messages = [self._compile_sys_message(sys_msg_path, gen_ctx_args)]

    def init_chat(self, chat_model: BaseChatModel, model: str) -> None:
        self._model = chat_model(model=model)

    def _load_instructions(self, instruct_path: str) -> str:
        with open(instruct_path, "r") as f:
            sys_txt = f.read()
        return sys_txt

    def _compile_sys_message(self, instruct_path: str, gen_ctx_args: Tuple[Tuple, Dict]) -> SystemMessage:
        sys_instruct = self._load_instructions(instruct_path)
        sys_txt = self._generate_context(
            sys_instruct,
            *gen_ctx_args[0],
            **gen_ctx_args[1],
        )
        return SystemMessage(sys_txt)

    def invoke(self, user_msg: str) -> str:
        human_msg = HumanMessage(user_msg)
        self.messages.append(human_msg)
        ai_msg = self._model.invoke(self.messages)
        self.messages.append(ai_msg)
        return ai_msg.content

    @staticmethod
    def get_sys_msg_path(msg_name: str) -> str:
        return Path(agent.__file__).parent / "messages" / msg_name


if __name__ == "__main__":

    utils.set_env_vars()

    chat = ChatFullContext(
        sys_msg_name=DEFAULT_SYS_MSG_NAME,
        chat_model=ChatAnthropic,
        model=DEFAULT_MODEL,
        generate_full_sys_msg=generate_sys_message,
        gen_ctx_args=(("Testing interactions",), {}),
    )

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        print(chat.invoke(user_input))
