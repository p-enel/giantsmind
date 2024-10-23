from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser

from giantsmind.utils import utils

utils.set_env_vars()

chat_with_wiki = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    # api_key="" # Optional if not set as an environment variable
)
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=5000,
)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

chat_grok = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    # api_key="" # Optional if not set as an environment variable
)

# print(tool.invoke({"query": "langchain"}))

chat_with_wiki = chat.bind_tools([tool])

prompt_summarize = """
Summarize this introduction of a Wikipedia article. Do not include any
additional information, you only need to summarize the text.

<introduction>
{introduction}
</introduction>

Responses should be properly formatted to be easily read.
"""

prompt = PromptTemplate(template=prompt_summarize, input_variables=["question"])


if __name__ == "__main__":

    messages = []
    query = "What is Wikipedia?"
    messages = [HumanMessage(query)]
    ai_msg = chat_with_wiki.invoke(messages)
    messages.append(ai_msg)
    for tool_call in ai_msg.tool_calls:
        if tool_call["name"] == "wikipedia":
            tool_output = tool.invoke(tool_call["args"]["__arg1"])
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    print(messages)
    print(messages[-1].content)

    chain = (
        (lambda x: [HumanMessage(x)])
        | chat_with_wiki
        | (lambda x: x.tool_calls[0]["args"]["__arg1"])
        | (lambda x: tool.invoke(x))
        | (lambda x: prompt.invoke(x))
        | chat_grok
        | StrOutputParser()
    )

    res = chain.invoke(query)
    print(res)
