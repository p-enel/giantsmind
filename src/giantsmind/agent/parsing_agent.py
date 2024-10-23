from langchain_anthropic import ChatAnthropic
from giantsmind.utils import utils

utils.set_env_vars()


def generate_prompt(user_question: str) -> str:
    prompt = f"""
You are a highly skilled assistant for parsing scientific queries. Your job is to break down the user's request into three parts after you check that it contains a single request for content resources:

0. Single source request check: the user's request should contain only request for content. If multiple request are detected, the user should be informed to split them into individual questions. For example, asking "What are the effects of climate change on coral reefs according to Dr. Goldman, and how have restoration efforts progressed since 2015?" should trigger an error because two distinct content requests are required:
- The effect of climate change on coral reefs in Dr. Goldman papers
- Progession of restoration efforts since 2015
But asking "Can you summarize the key advancements in machine learning for healthcare applications from 2019 onwards, include the main challenges, and mention which institutions are leading these efforts?" is admissible because it contains a single request for content:
- Key advancements in machine learning for healthcare applications from 2019 onwards
Summarizing while including main challenges and mentionning which institutions are leading the efforts can be done from the same content.

1. Metadata plain text search: write a concise sentence containing the scientific articles metadata required for filtering. This includes:
- author(s)
- publication year
- publication name
- title

2. Content search: Provide a short sentence describing what content should be searched within the paper's contents.
3. General knowledge required: Determine if general knowledge is required to support the answer. If so, briefly describe it.

Here are some examples:

<example>
User Question: "What classification methods are used in Albert Smith's papers since 2020?"
Metadata Search: Retrieve papers authored by Albert Smith since 2020
Content Search: classification methods
General Knowledge: None
</example>

<example>
User Question: "List all papers on deep reinforcement learning authored by Jane Doe."
Metadata Search: Retrieve papers authored by Jane Doe
Content Search: deep reinforcement learning
General Knowledge: None
</example>

<example>
Use Question: "What are the recent advancements in quantum computing since 2021? Also, list the authors involved in these papers."
Error: Multiple content requests detected!
- The advancements in quantum computing since 2021
- Authors involved in those papers
Please split the following questions into separate queries.
</example>

<example>
User Question: "Explain the key findings of the paper by John Smith on CRISPR technology."
Metadata Search: Retrieve the paper authored by John Smith
Content Search: key findings on CRISPR technology
General Knowledge: Provide general context on CRISPR technology if needed
</example>

<example>
User Question: "Find papers by either Alice Brown or Bob Green on gene therapy, published between 2018 and 2022."
Metadata Search: Retrieve papers authored by Alice Brown or Bob Green, published between 2018 and 2022
Content Search: gene therapy
General Knowledge: None
</example>

<example>
User Question: "Summarize the methodology of papers on protein folding by Robert Chen published after 2019."
Metadata Search: Retrieve papers authored by Robert Chen published after 2019
Content Search: methodology of protein folding
General Knowledge: Provide context on protein folding if needed
</example>

<example>
User Question: "Who are the main contributors to research on neural networks, and what are the key advancements since 2020?"
Error: Multiple content requests detected!
- The main authors in neural networks research
- Key advancements in neural networks since 2020?
Please split the following questions into separate queries.
</example>

<example>
User Question: "Find studies on renewable energy integration, focusing on papers by Emily Davis published in 2020 or Michael Scott published in 2021."
Metadata Search: Retrieve papers authored by Emily Davis published in 2020 or Michael Scott published in 2021
Content Search: renewable energy integration
General Knowledge: None
</example>

<example>
User Question: "Provide an overview of the role of renewable energy technologies in reducing greenhouse gas emissions according to Dr. Bill Clinton since 2000, including challenges faced and recent technological advancements."
Metadata Search: Retrieve papers from Bill Clinton since 2000 
Content Search: role of renewable energy technologies in reducing greenhouse gas emission
General Knowledge: Provide context a more general context on the role of renewable energy in reducing greenhouse gas emission

<example>
User Question: "Summarize the findings on biodiversity impacts of urbanization from papers authored by Maria Gonzalez or Mark Thompson, published between 2015 and 2020."
Metadata Search: Retrieve papers authored by Maria Gonzalez or Mark Thompson, published between 2015 and 2020
Content Search: biodiversity impacts of urbanization
General Knowledge: Provide context on urbanization and biodiversity if needed
</example>

<example>
User Question: "How has deforestation affected biodiversity in the Amazon rainforest according to Dr. Carter, and what are the socio-economic impacts on the local communities since 2010?"
Error: Multiple content requests detected!
- Effects of deforestation on biodiversity in the Amazon rainforest as explained in Dr. Carter's research.
- Socio-economic impacts on local communities in the Amazon since 2010.
Please split the following questions into separate queries.
</example>

Follow the exact format seen in the examples above. If the user question does not contains multiple requests, your answer should be 3 lines only.

Now it's your turn.
    
User Question: "{user_question}"
    """
    return prompt


def parse_question(user_question):
    """
    Parses the user's question into three parts:
    - Metadata plain text search
    - Content search
    - General knowledge required

    Args:
        user_question (str): The question provided by the user.

    Returns:
        dict: A dictionary containing the parsed outputs: metadata_search, content_search, general_knowledge.
    """

    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    prompt = generate_prompt(user_question)
    response = model.invoke(prompt)

    # Handle multiple questions detected
    if "Error: Multiple questions detected" in response.content:
        return {"error": response.content.strip()}

    parsed_response = response.content.strip().split("\n")

    metadata_search = parsed_response[0].replace("Metadata Search: ", "").strip()
    content_search = parsed_response[1].replace("Content Search: ", "").strip()
    general_knowledge = parsed_response[2].replace("General Knowledge: ", "").strip()

    return {
        "metadata_search": metadata_search,
        "content_search": content_search,
        "general_knowledge": general_knowledge,
    }


# Example usage
if __name__ == "__main__":
    user_question = (
        "Find all papers published by Pierre Enel or Peter Dominey on reservoir computing since 2010."
    )
    parsed_question = parse_question(user_question)
    print(parsed_question)
    # print(parsed_question.content)
from langchain_anthropic import ChatAnthropic
from giantsmind.utils import utils

utils.set_env_vars()


def generate_prompt(user_question: str) -> str:
    prompt = f"""
You are a highly skilled assistant for parsing scientific queries. Your job is to break down the question into three parts:
1. Metadata plain text search: write a concise sentence containing the scientific articles metadata required for filtering. This includes:
- author(s)
- publication year
- publication name
- title

2. Content search: Provide a short sentence describing what content should be searched within the paper's contents.
3. General knowledge required: Determine if general knowledge is required to support the answer. If so, briefly describe it.

Here are some examples:

User Question: "What classification methods are used in Albert Smith's papers since 2020?"
Metadata Search: Retrieve papers authored by Albert Smith since 2020
Content Search: classification methods
General Knowledge: None

User Question: "List all papers on deep reinforcement learning authored by Jane Doe."
Metadata Search: Retrieve papers authored by Jane Doe
Content Search: deep reinforcement learning
General Knowledge: None

User Question: "What are the recent advancements in quantum computing from 2021 onwards?"
Metadata Search: Retrieve papers published since 2021
Content Search: advancements in quantum computing
General Knowledge: None

User Question: "Explain the key findings of the paper by John Smith on CRISPR technology."
Metadata Search: Retrieve the paper authored by John Smith
Content Search: key findings on CRISPR technology
General Knowledge: Provide general context on CRISPR technology if needed

User Question: "Find papers by either Alice Brown or Bob Green on gene therapy, published between 2018 and 2022."
Metadata Search: Retrieve papers authored by Alice Brown or Bob Green, published between 2018 and 2022
Content Search: gene therapy
General Knowledge: None

User Question: "What are the applications of machine learning in climate modeling from recent publications by Sarah Lee and co-authors?"
Metadata Search: Retrieve recent papers authored by Sarah Lee and co-authors
Content Search: applications of machine learning in climate modeling
General Knowledge: None

User Question: "Summarize the methodology of papers on protein folding by Robert Chen published after 2019."
Metadata Search: Retrieve papers authored by Robert Chen published after 2019
Content Search: methodology of protein folding
General Knowledge: None

User Question: "Find studies on renewable energy integration, focusing on papers by Emily Davis published in 2020 or Michael Scott published in 2021."
Metadata Search: Retrieve papers authored by Emily Davis published in 2020 or Michael Scott published in 2021
Content Search: renewable energy integration
General Knowledge: None

User Question: "Summarize the findings on biodiversity impacts of urbanization from papers authored by Maria Gonzalez or Mark Thompson, published between 2015 and 2020."
Metadata Search: Retrieve papers authored by Maria Gonzalez or Mark Thompson, published between 2015 and 2020
Content Search: biodiversity impacts of urbanization
General Knowledge: Provide context on urbanization and biodiversity if needed

User Question: "What techniques are being used for image segmentation in the latest publications by Alex White and co-authors?"
Metadata Search: Retrieve latest papers authored by Alex White and co-authors
Content Search: techniques for image segmentation
General Knowledge: None

User Question: "What are the conclusions in the article titled 'Advances in Neural Networks' by Linda Clark published in 2022?"
Metadata Search: Retrieve the article titled 'Advances in Neural Networks' authored by Linda Clark, published in 2022
Content Search: conclusions
General Knowledge: None

Follow the exact format seen in the examples above, your answer should be 3 lines.

Now it's your turn.
    
User Question: "{user_question}"
    """
    return prompt


def parse_question(user_question):
    """
    Parses the user's question into three parts:
    - Metadata plain text search
    - Content search
    - General knowledge required

    Args:
        user_question (str): The question provided by the user.

    Returns:
        dict: A dictionary containing the parsed outputs: metadata_search, content_search, general_knowledge.
    """

    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
    prompt = generate_prompt(user_question)
    response = model.invoke(prompt)

    # return response
    parsed_response = response.content.strip().split("\n")

    metadata_search = parsed_response[0].replace("Metadata Search: ", "").strip()
    content_search = parsed_response[1].replace("Content Search: ", "").strip()
    general_knowledge = parsed_response[2].replace("General Knowledge: ", "").strip()

    return {
        "metadata_search": metadata_search,
        "content_search": content_search,
        "general_knowledge": general_knowledge,
    }


# Example usage
if __name__ == "__main__":
    user_question = (
        "Find all papers published by Pierre Enel or Peter Dominey on reservoir computing since 2010."
    )
    user_question = "What method is used in Dr. Harlet's paper for brain dissection that is different from Dr. Smith in their papers since 2013?"
    parsed_question = parse_question(user_question)
    print(parsed_question)
    # print(parsed_question.content)
