import dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_neo4j import Neo4jGraph
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from src.configs import settings
from src.generators import AnswerGenerator, Text2Cypher
from src.retrievers import KnowledgeRetriever
from src.schemas import GenerationFlowState

dotenv.load_dotenv(override=True)


# init models
llm = GoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0.0,
    max_tokens=8096,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=settings.EMBEDDING_DEPLOYMENT_NAME,
    model=settings.EMBEDDING_MODEL_NAME,
    azure_endpoint=settings.EMBEDDING_AZURE_ENDPOINT,
    api_version=settings.EMBEDDING_API_VERSION,
    api_key=settings.EMBEDDING_AZURE_OPENAI_API_KEY,
)

neo4j = Neo4jGraph(
    url=settings.NEO4J_URL,
    username=settings.NEO4J_USER,
    password=settings.NEO4J_PWD,
    refresh_schema=True,
    enhanced_schema=True,
)

milvus = Milvus(
    embedding_function=embeddings,
    enable_dynamic_field=True,
    auto_id=True,
    connection_args={"uri": settings.MILVUS_URI, "token": settings.MILVUS_TOKEN},
    collection_name=settings.MILVUS_COLLECTION_NAME,
)

# init tasks
knowledge_retriever = KnowledgeRetriever(graph_db=neo4j)
answer_generator = AnswerGenerator(llm=llm)
text2cypher = Text2Cypher(llm=llm, graph_db=neo4j, vector_db=milvus)


# define workflow
graph = StateGraph(GenerationFlowState)
graph.add_node("text2cypher", text2cypher.arun)
graph.add_node("knowledge_retriever", knowledge_retriever.arun)
graph.add_node("answer_generator", answer_generator.arun)

graph.add_edge(START, "text2cypher")
graph.add_edge("text2cypher", "knowledge_retriever")
graph.add_edge("knowledge_retriever", "answer_generator")
graph.add_edge("answer_generator", END)

compiled_graph = graph.compile()
