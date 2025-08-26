from abc import ABC, abstractmethod
from typing import Any, AnyStr, Dict, List, TypedDict

from langchain_core.documents import Document


class BaseState(TypedDict):
    logs: Dict[str, Any]
    errors: List[Dict[str, Any]]


class GenerationFlowState(BaseState):
    """Generation state"""

    question: AnyStr
    answer: AnyStr
    contexts: List[Document]
    metadata: Dict


class BaseStep(ABC):
    def __init__(self, **kwargs: dict) -> None:
        _configs: Dict[str, Any] = dict()

    @abstractmethod
    async def arun(self, input: BaseState) -> BaseState:
        raise NotImplementedError
