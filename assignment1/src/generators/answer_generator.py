from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.language_models import LLM
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate

from src.loggers import logger
from src.prompts.answer_generator import ANSWER_GENERATOR_PROMPT
from src.schemas import BaseStep, GenerationFlowState


class AnswerGenerator(BaseStep):
    def __init__(self, llm: LLM, prompt: str = ANSWER_GENERATOR_PROMPT):
        self.llm = llm
        self.chain = self._create_single_chain(
            prompt_template=prompt, parser=StrOutputParser()
        )

    def _create_single_chain(
        self, prompt_template: str, parser: StrOutputParser
    ) -> BaseCombineDocumentsChain:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"],
        )
        return prompt | self.llm | parser

    async def arun(self, state: GenerationFlowState) -> GenerationFlowState:
        """Generate answer

        Args:
            state (GenerationFlowState): input state

        Returns:
            GenerationFlowState: output state with generated answer
        """
        logger.info("AnswerGenerator")
        contexts = state.get("contexts", [])
        question = state.get("question")
        errors = state.get("errors", [])

        if errors:
            return state

        if not question:
            logger.error("No question to generate answer")
            return state

        answer = ""
        try:
            context_strs = [
                f"{
                    context.metadata.get("cypher")}\n{
                    context.metadata.get("graph_data")}"
                for context in contexts
            ]
            context_str = "\n".join(context_strs)
            answer = await self.chain.ainvoke(
                {"question": question, "context": context_str}
            )

        except Exception as e:
            error = f"Can not generate answer because of {e}"
            logger.error(error)
            errors.append(error)
            state["errors"] = errors

        state["answer"] = answer

        return state
