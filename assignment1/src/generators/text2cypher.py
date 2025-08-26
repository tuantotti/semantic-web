from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from langchain_neo4j.graphs.graph_store import GraphStore
from neo4j_graphrag.retrievers.text2cypher import extract_cypher
from neo4j_graphrag.schema import format_schema

from src.loggers import logger
from src.prompts.text2cypher import TEXT2CYPHER_PROMPT
from src.schemas import BaseStep, GenerationFlowState


class Text2Cypher(BaseStep):

    def __init__(
        self,
        llm: LLM,
        graph_db: GraphStore,
        prompt: str = TEXT2CYPHER_PROMPT,
        **kwargs: dict,
    ):
        """
        Initializes the Text2Cypher retriever with the given language model and keyword arguments.

        Args:
            llm (LLM): The language model used for generating cypher queries.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.graph_db = graph_db
        self.prompt = prompt

        _prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
        )
        self._chain = _prompt | self.llm | StrOutputParser()

        corrector_schema = [
            Schema(el["start"], el["type"], el["end"])
            for el in self.graph_db.get_structured_schema.get("relationships", [])
        ]
        self.cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    async def arun(self, state: GenerationFlowState) -> GenerationFlowState:
        errors = state.get("errors", [])
        question = state.get("question", None)
        if len(errors) > 0:
            return state

        cypher_query = ""
        try:
            cypher_query = await self._generate_cypher_query(question)
        except Exception as e:
            logger.exception(f"Can not generate cypher because of {e}")
            errors.append(e)
            state["errors"] = errors

        if not cypher_query or errors:
            return state

        try:

            context = [
                Document(
                    page_content="",
                    metadata={
                        "cypher": cypher_query,
                    },
                )
            ]
            state["contexts"] = context

        except Exception as err:
            logger.exception(
                f"Can not find similar entity in cypher code because of {err}"
            )
            errors.append(f"Cannot link entities because of {err}")
            state["errors"] = errors

        return state

    async def _generate_cypher_query(self, question: str) -> str:
        logger.info("Text2Cypher")
        graph_schema = construct_schema(
            self.graph_db.get_structured_schema,
            [],
            [],
            self.graph_db._enhanced_schema,
        )
        cypher_query = await self._chain.ainvoke(
            {"question": question, "schema": graph_schema}
        )
        cypher_query = extract_cypher(cypher_query)
        cypher_query = self.cypher_query_corrector.correct_query(cypher_query)
        logger.info(f"Cypher code: {cypher_query}")

        return cypher_query


def construct_schema(
    structured_schema: Dict[str, Any],
    include_types: List[str] = [],
    exclude_types: List[str] = [],
    is_enhanced: bool = True,
) -> str:
    """Filter the schema based on included or excluded types"""

    def filter_func(x: str) -> bool:
        return x in include_types if include_types else x not in exclude_types

    filtered_schema: Dict[str, Any] = {
        "node_props": {
            k: v
            for k, v in structured_schema.get("node_props", {}).items()
            if filter_func(k)
        },
        "rel_props": {
            k: v
            for k, v in structured_schema.get("rel_props", {}).items()
            if filter_func(k)
        },
        "relationships": [
            r
            for r in structured_schema.get("relationships", [])
            if all(filter_func(r[t]) for t in ["start", "end", "type"])
        ],
    }
    return format_schema(filtered_schema, is_enhanced)
