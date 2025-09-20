from typing import List

import gradio as gr
from gradio import ChatMessage

from src.configs import settings
from src.workflow import compiled_graph


async def generate_answer(message: str, history: List) -> str:
    """Generate answer for user's message

    Args:
        message (str): user's message
        history (List): user's conversational history

    Returns:
        str: generated answer
    """

    state = await compiled_graph.ainvoke({"question": message})
    return state["answer"]


async def response(message: str, history: List[ChatMessage]):
    """Create and display response for user interface

    Args:
        message (str): user's message
        history (List[ChatMessage]): conversational history

    Yields:
        List: all conversational history
    """
    final_answer = await generate_answer(message, history)
    history.append(ChatMessage(role="assistant", content=final_answer))
    yield history


demo = gr.ChatInterface(
    response,
    title="FEBMS Chatbot",
    type="messages",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=settings.APPLICATION_API_PORT)
