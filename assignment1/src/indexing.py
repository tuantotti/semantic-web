import dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_neo4j import Neo4jGraph
from src.configs import settings
from langchain_core.documents import Document

dotenv.load_dotenv(override=True)

cypher = """MATCH (n)
WHERE n:Route OR n:Stop
RETURN DISTINCT n.name AS name"""

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=settings.EMBEDDING_DEPLOYMENT_NAME,
    model=settings.EMBEDDING_MODEL_NAME,
    azure_endpoint=settings.AZURE_ENDPOINT,
    api_version=settings.API_VERSION,
    api_key=settings.AZURE_OPENAI_API_KEY,
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

result = neo4j.query(cypher)
node_names = [record["name"] for record in result if record["name"]]
# node_names = [tokenize(name) for name in node_names]
milvus.add_documents([Document(page_content=name) for name in node_names])