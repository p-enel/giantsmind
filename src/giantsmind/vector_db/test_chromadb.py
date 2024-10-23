from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the embedding model (OpenAI in this example, you can use others)
MODELS = {"bge-small": {"model": "BAAI/bge-base-en-v1.5", "vector_size": 768}}
embedding = FastEmbedEmbeddings(model_name=MODELS["bge-small"]["model"])

persist_directory = "/home/pierre/.local/share/giantsmind/"

# Initialize Chroma as the vectorstore
# This will store the embeddings locally by default
chroma_db = Chroma(
    embedding_function=embedding, collection_name="my_collection", persist_directory=persist_directory
)

# Add some documents to the ChromaDB for storage
texts = [
    "Langchain is a framework for developing language model applications.",
    "ChromaDB is a fast and efficient vector store that can be used with Langchain.",
    "OpenAI's GPT-3 is a powerful model for generating text and understanding queries.",
    "Retrieval-based question answering systems use vector databases to store and retrieve data.",
]

metadatas = [
    {"author": "Pierre", "year": 2022},
    {"author": "Albert", "year": 2023},
    {"author": "John", "year": 2024},
    {"author": "Jane", "year": 2025},
]

# Add documents (texts) to ChromaDB
chroma_db.add_texts(texts, metadatas)

# Now perform a similarity search using ChromaDB
# This will search for documents similar to the query
query = "What is ChromaDB?"
query = "Proximal Policy Optimization"
vector = chroma_db.embeddings.embed_query(query)
results = chroma_db.similarity_search_with_score(query)
results = chroma_db.similarity_search_by_vector_with_relevance_scores(vector)

# Output the results
print("Results for the query '{}':".format(query))
for i, doc in enumerate(results):
    print(f"Document {i + 1}: {doc.page_content}")


chroma_db.get()
chroma_db.delete_collection()

chroma_client = chroma_db._client

collection = chroma_client.get_collection("my_collection")


# text = "dynamics. By default we use a sparse reward of +1 if the final answer is correct but also experiment with dense rewards matching intermediate steps in a reference solution and rewards synthetically generated using a reward model. We evaluate models with 7B and 13B parameters both starting from supervised fine-tuned (SFT) checkpoints and pre-trained checkpoints. We report four metrics assessing model performance on a task specific test set: 1) maj@1 score computed by greedily sampling once per question, 2) maj@96 score computed by sampling K = 96 times per question and uniformly voting on the final answer, 3) rerank@96 score computed by sampling K = 96 times and choosing the final answer using an Outcome-Based Reward Model (ORM), and 4) pass@96 score computed by sampling the model K = 96 times and taking the best result according to the ground truth answer. We find that overall the simplest method, Expert Iteration (EI) (Anthony et al., 2017), performs best across all metrics for most reward setups and model initializations. Surprisingly, EI is nearly as sample efficient as more sophisticated algorithms like Proximal Policy Optimization (PPO), both requiring only a few thousand samples to converge even when initialized from a pretrained checkpoint. We also observe the gap between pretrained model performance and SFT model performance significantly shrinks (< 10% gap on GSM8K) after RL fine-tuning, with larger models having a smaller gap. Additionally, previous work identified a tradeoff between test time maj@1 performance and pass@96 performance during supervised fine-tuning (Cobbe et al., 2021), with continued training increasing maj@1 score at the expense of pass@96 score. We identify the limited diversity of the dataset as a core reason for this. We show that RL fine-tuning can improve both metrics simultaneously due to the fact that RL generates its own data during training, resulting in a more diverse set of examples to learn from. We then discuss why EI and return conditioned RL are competitive with PPO, suggesting two principal factors. Firstly, the reasoning tasks we consider have entirely deterministic dynamics: a setting in which direct behavior cloning and return conditioned RL is known to do well (Brandfonbrener et al., 2022). In contrast, PPO often succeeds in environments with a high degree of stochasticity (Bhargava et al., 2023). Second, we identify a lack of sophisticated exploration carried out by models during RL fine-tuning. This limitation significantly impacts any performance or sample complexity advantages PPO may have when fine-tuning the pretrained model. We come to this conclusion from a number of observations, noting in particular quickly saturating pass@96 scores early in RL training. We conclude with a discussion of the impacts of our observations on RLHF and the future of LLM fine-tuning via RL. In summary we make the following contributions:\n• A comprehensive study of PPO fine-tuning of LLMs on reasoning tasks using different types of rewards, model sizes and initializations.\n• A comparison to expert iteration and return-conditioned RL from which we find expert iteration reliably attains the best performance and competitive sample complexity across the board.\n• A discussion of the implications of our findings for RLHF and the future of RL fine-tuning for LLMs, identifying exploration as a major limiting factor.\nRelated Work LLM Reasoning: State-of-the-art large language models (OpenAI, 2023; Touvron et al., 2023; Bai et al., 2022; Chowdhery et al., 2022) demonstrate increasingly impressive abilties on hard reasoning tasks as studied by a wide range of math, science, and code benchmarks (Cobbe et al., 2021; Hendrycks et al., 2021b; Sawada et al., 2023; Liang et al., 2022; Srivastava et al., 2022; Rein et al., 2023; Mialon et al., 2023; Chollet, 2019; Mishra et al., 2022; Hendrycks et al., 2021a; Austin et al., 2021; Patel et al., 2021; Gao et al., 2021). Chain of thought (CoT) (Wei et al., 2022) and related techniques (Chen et al., 2022; Yao et al., 2023; Besta et al., 2023) have emerged as dominant methods siginficantly boosting LLM performance on these types of tasks. CoT methods allow LLMs to defer giving their final answer by first generating a ”chain of thought” involving intermediate computations needed to correctly solve the problem. Another line of work combines base LLM reasoning capabilities with planning and search algorithms to further boost performance on a wide range of tasks (Yao et al., 2023; Besta et al., 2023; Ye et al., 2022; Yao et al., 2022; Dohan et al., 2022). Tree of thought (Yao et al., 2023) for example combines LLMs with a breadth first search algorithm, relying on the LLM to both propose actions and evaluate state. Other works combine LLMs with tools (Schick et al., 2023; Qin et al., 2023; Zhou et al., 2023a) further boosting reasoning capability. Combining GPT-4 with a python code interpreter for generation and self-verification achieves an impressive 84% on the hard MATH benchmark (Hendrycks et al., 2021a; Zhou et al., 2023a). Other works focus on LLMs for mathematical reasoning in natural language (Cobbe et al., 2021; Lewkowycz et al., 2022; Azerbayev et al., 2023; Lightman et al., 2023; Patel et al., 2021; Zhu et al., 2023; Rafailov et al., 2023). Particularly relevant to our study is Cobbe et al. (2021) which fine-tunes GPT-3 on supervised math word problem (MWP) reasoning traces. In addition they train solution verifiers called Outcome Based Reward Models (ORMs) which predict the probability of correctly solving a question Q giving a prefix of intermediate steps Pi = (S1, ..., Si) i.e. p(is correct(A)|Q, Pi) where A is a solution with prefix Pi. Process based reward models (PRMs) (Uesato et al., 2022; Lightman et al., 2023) can also be trained to instead look at the step-level accuracy of solutions. More recent work (Luo et al., 2023) utlizies a PRM distilled from GPT-4 feedback as a reward signal during PPO. RL for LLM fine-tuning: Reinforcement Learning from Human Feedback (RLHF) is perhaps the most well-known application of RL techniques for fine-tuning LLMs. RLHF (Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022; Glaese et al., 2022; Peng et al., 2021; Ramamurthy et al., 2022) most often works by training a reward model to capture human preferences over a task τ . The reward model is then used to score LLM responses to prompts from the task after which policy improvement is performed. PPO is most often used (Ouyang et al., 2022; Bai et al., 2022) but several recent works including ReST (Gulcehre et al., 2023), Reward-Ranked Fine-tuning (Dong et al., 2023), and AlpacaFarm (Dubois et al., 2023) all demonstrate simply fine-tuning on high return responses with the standard cross-entropy loss can attain comparable performance. We broadly refer to this class of algorithms as Expert Iteration.\nA large body of work studying RL for LLM fine-tuning also exists outside of the RLHF sphere. Work on text games (Yao et al., 2020; Ammanabrolu and Riedl, 2019) and other interactive textual environments (Zhou et al., 2023b; Carta et al., 2023) seek to ground LLMs via interaction and RL. RL has also been applied to improving model performance on controllable generation and question answering tasks (Lu et al., 2022; Liu et al., 2022). Various forms of expert iteration have also been applied to improve LLM reasoning capabilities (Huang et al., 2022; Yuan et al., 2023; Zelikman et al., 2022; Uesato et al., 2022). For example “Scaling Relationship on Learning Mathematical Reasoning with Large Language Models” (Yuan et al., 2023) applies a single round of expert iteration across multiple model sizes on GSM8K. They observe sizeable gains in all metrics for smaller models, with gains diminishing for larger models. A related body of work studies RL for code generation (Le et al., 2022; Shen et al., 2023; Rozi`ere et al., 2023). Shen et al. (2023) in particular reports a huge increase in StarCoder’s (Li et al., 2023) maj@1 performance after a single round of expert iteration, jumping from ∼30% to ∼60%. Despite all the above work, it remains unclear exactly what factors account for the biggest impact during RL fine-tuning due to wide variance in tasks, pretraining data, supervised fine-tuning data, RL algorithm used, and the reward source. Our work conducts a thorough analysis of all these factors to understand exactly how different algorithms compare when applied to improving LLM reasoning capability. As a result we are able to identify key bottlenecks to further LLM improvement via RL and provide a discussion on promising future directions.3 Methods Reasoning as an RL problem."

# long_doc = Document(page_content=text, metadata={"source": "Langchain Documentation"})

# # Split the document into smaller chunks using a text splitter
# # RecursiveCharacterTextSplitter handles splitting intelligently by breaking on sentences, paragraphs, etc.
# splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
# chunks = splitter.split_text(long_doc.page_content)
# chunked_docs = [Document(page_content=chunk, metadata=long_doc.metadata) for chunk in chunks]


# doc = Document(page_content=doc)
# chroma_db.add_documents([chunked_docs])
