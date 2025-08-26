from typing import List, Tuple

from langchain_neo4j.graphs.graph_document import Relationship
from pydantic import BaseModel


class GraphSchema(BaseModel):
    """
    Graph schema with node and edge types which connect together in format
    (source_type, edge_type, target_type)
    """

    schema: List[Tuple[str, str, str]] = []

    def node_types(self) -> List[str]:
        """
        Returns:
            List[str]: list of node type
        """
        types = []
        for source, _, target in self.schema:
            types.append(source)
            types.append(target)
        return list(set(types))

    def edge_types(self) -> List[str]:
        """
        Returns:
            List[str]: List of edge type
        """
        types = []
        for _, edge, _ in self.schema:
            types.append(edge)
        return list(set(types))

    def validate(self, relation: Relationship) -> bool:
        """
        Args:
            relation (Relationship)

        Returns:
            bool: validation value
        """
        source_type, edge_type, target_type = (
            relation.source.type,
            relation.type,
            relation.target.type,
        )
        for source, edge, target in self.schema:
            if source == source_type and edge == edge_type and target == target_type:
                return True

        return False
