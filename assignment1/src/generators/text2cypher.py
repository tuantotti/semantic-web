import re
from itertools import product
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_neo4j.chains.graph_qa.cypher_utils import (CypherQueryCorrector,
                                                          Schema)
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
        vector_db: VectorStore,
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
        self.vector_db = vector_db

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
            cypher_queries = await self._map_entities(cypher_query)
            contexts = [
                Document(
                    page_content="",
                    metadata={
                        "cypher": cypher_augmented_query["cypher"],
                        "score": cypher_augmented_query["score"],
                    },
                )
                for cypher_augmented_query in cypher_queries
                if cypher_augmented_query
            ]
            state["contexts"] = contexts

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

    async def _map_entities(self, cypher_query: str):
        # Execute the Cypher query using the database
        entity_map = []
        entities = self._extract_entity_from_cypher(cypher_query)

        if not entities:
            logger.info("No entities to link")
            return input

        for entity in entities:
            results = self._map_entity(entity=entity)
            entity_map.append(
                {
                    "question": entity,
                    "database": [
                        {
                            "content": r.page_content,
                            "score": (
                                r.metadata.get("score")
                                if r.metadata.get("score")
                                else 1.0
                            ),
                        }
                        for r in results
                    ],
                }
            )

        cypher_augmented_queries = self._synthetic_cypher_query(
            cypher_query, entity_map
        )

        return cypher_augmented_queries

    def _map_entity(self, entity: str) -> List[Document]:
        # entity = tokenize(entity)
        return self.vector_db.search(query=entity, search_type="similarity")

    def _extract_entity_from_cypher(self, cypher: str) -> List[str]:
        pattern = r"{\w+:\s*\"([^\"]+)\"}"
        cypher = cypher.replace("'", "\"")
        matches = re.findall(pattern, cypher)
        matches = list(set(matches))
        return matches

    def _synthetic_cypher_query(self, raw_cypher: str, entity_map: list) -> list:
        cypher_variations = []

        entities_in_question = [entity["question"] for entity in entity_map]
        entities_in_database = [entity["database"] for entity in entity_map]

        combinations = []

        # Use itertools.product to generate all possible combinations
        for combination in product(*entities_in_database):
            entities = [item["content"] for item in combination]
            score = self._calculate_score(combination, type="mean")
            combinations.append({"entities": entities, "score": score})

        # Sort combinations by mean score in descending order
        combinations = sorted(combinations, key=lambda x: x["score"], reverse=True)

        cypher_variations = []
        for combination in combinations:
            entities = combination["entities"]
            score = combination["score"]

            temp_cypher = raw_cypher
            for entity_in_question, entity_in_database in zip(
                entities_in_question, entities
            ):
                temp_cypher = temp_cypher.replace(
                    entity_in_question, entity_in_database
                )

            cypher_variations.append({"cypher": temp_cypher, "score": score})

        return cypher_variations

    def _calculate_score(self, results: List, type: str) -> float:
        score = 0.0
        if type == "mean":
            score = sum(item["score"] for item in results) / len(results)
        return score


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
