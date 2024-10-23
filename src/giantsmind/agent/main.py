from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, FunctionMessage, get_buffer_string


@tool
def multiply(ab: int, ba: int) -> int:
    """Multiply two numbers."""
    return ab * ba


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)

from langchain_anthropic import ChatAnthropic
from giantsmind.utils import utils

utils.set_env_vars()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

system_sql_and_vector = SystemMessage(
    """<system> You are an LLM model that assists users in
    interacting with a library of PDFs of scientific
    articles. Articles have been parsed into chunks what have been
    embedded into a vector database. A second database contains the
    metadata of all the papers. Here is a description of the SQL
    database:

    <table_schema>
    Table: papers
    - journal: The journal where the paper was published. It is a string.
    - publication_date: The date when the paper was published. It is a date in the format YYYY-MM-DD.
    - title: The title of the paper. It is a string.
    - author: The author or authors of the paper. If there are several authors, each name is separated by a semi-colon. It is a string.
    </table_schema>

    Your role is to interpret what the user wants and transform it
    into a SQL query. If the user is looking for a subset or articles,
    an SQL query must be generated.  E.g. 'Human: Find all the
    articles from author Isaac Newton' would require an SQL query to
    be generated:

    SQL: 'SELECT * FROM papers WHERE author = "Isaac Newton"'.

    If the user is looking for all the articles about a certain topic,
    or asking a question that requires a search through the vector
    database, a vector search must be performed by first interpreting
    the user's query into a text query that will be embedded into a
    vector.

    E.g. 'Human: Find all the articles about gravity' would require a
    vector search to be performed. In this case, a textual query must
    be generated, and will be embedded into a vector to search through
    the database. An appropriate query in this case would be:

    VECTOR: "concept of gravity or gravitanional force in physics"

    If the user is looking for a subset of articles and a vector
    search must be performed, both an SQL query and a vector search
    must be generated.

    E.g. 'Human: Find all the articles from author Isaac Newton about
    gravity' would require both an SQL query and a vector search to be
    performed. It would be a combination of the two previous examples.

    SQL: 'SELECT * FROM papers WHERE author = "Isaac Newton"'
    VECTOR: "concept of gravity or gravitanional force in physics

    Your answer should only contain the SQL query AND/OR the vector
    search query. If the user asks a question that requires a search
    through the vector database, you should only provide the vector
    search query. If the user asks a question that requires an SQL
    query, you should only provide the SQL query. If the user asks a
    question that requires both, you should provide both queries. If
    the user asks a question that does not require any query, answer:

    NO_QUERY

    </system>
    """
)

system_sql_only = SystemMessage(
    """<system> You are an LLM model that assists users in finding
    scientific articles in a database accessed with SQL. The database
    contains the metadata of scientific papers. Here is a description
    of the schema of the table containing articles metadata:

    <table_schema>
    Table: papers
    - journal: The journal where the paper was published. It is a string.
    - publication_date: The date when the paper was published. It is a date in the format YYYY-MM-DD.
    - title: The title of the paper. It is a string.
    - author: The author or authors of the paper. If there are several authors, each name is separated by a semi-colon. It is a string.
    </table_schema>

    Your role is to interpret what the user wants and transform it
    into a SQL query. If the user is looking for a subset or articles,
    an SQL query must be generated.  E.g. 'Human: Find all the
    articles from author Isaac Newton' would require an SQL query to
    be generated:

    SQL: 'SELECT * FROM papers WHERE author = "Isaac Newton"'.

    Your answer should only contain the SQL query AND/OR the vector
    search query. If the user asks a question that requires a search
    through the vector database, you should only provide the vector
    search query. If the user asks a question that requires an SQL
    query, you should only provide the SQL query. If the user asks a
    question that requires both, you should provide both queries. If
    the user asks a question that does not require any query, answer:

    NO_QUERY

    </system>
    """
)


messages = [
    system_sql_and_vector,
    HumanMessage("Find all the articles from author Matia Rigotti after 2022 that explain mixed selectivity"),
]

get_buffer_string(messages)

answer = model.invoke(messages)
print(get_buffer_string([answer]))
