from context import Context
from giantsmind.core import interact_papers


def act_ask_question(context: Context) -> Context:
    interact_papers.one_question_chain(int(context.papers))
    return context
