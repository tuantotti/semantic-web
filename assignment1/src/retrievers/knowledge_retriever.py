from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_store import GraphStore

from src.loggers import logger
from src.schemas import BaseStep, GenerationFlowState


class KnowledgeRetriever(BaseStep):
    def __init__(self, graph_db: GraphStore, **kwargs: dict):
        super().__init__(**kwargs)
        self.graph_db = graph_db

    async def arun(self, state: GenerationFlowState) -> GenerationFlowState:
        logger.info("KnowledgeRetriever")
        _errors = state.get("errors", [])

        if len(_errors) > 0:
            return state

        contexts = state.get("contexts", [])
        if not contexts:
            logger.info(f"No context to run cypher query")
            return state

        retrieved_contexts = []
        try:
            for doc in contexts:
                _cypher_query = doc.metadata.get("cypher", "")
                results = self.graph_db.query(query=_cypher_query)

                graph_context = Document(
                    page_content="",
                    metadata={
                        "cypher": _cypher_query,
                        "graph_data": results,
                    },
                )

                retrieved_contexts.append(graph_context)

            state["contexts"] = retrieved_contexts
        except Exception as err:
            logger.exception(f"Can not retrieve graph data because of {err}")
            _errors.append(err)
            state["errors"] = _errors
        return state
