from typing import Any, cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from agent_context import AgentContext
from agent_tools import build_tools
from models import OpenAIModel
from prompt import SYSTEM_PROMPT


class Agent:
    def __init__(self, llm: OpenAIModel):
        tools = build_tools(llm)

        self._checkpointer = InMemorySaver()
        self._agent = create_agent(
            model=llm.model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            context_schema=AgentContext,
            checkpointer=self._checkpointer,
        )

    def run(self, user_input: str, image_path: str, output_path: str, thread_id: str):
        agent_input = {
            "messages": [
                HumanMessage(content=user_input),
            ]
        }

        context = AgentContext(
            image_path=image_path,
            output_path=output_path,
        )

        config = {
            "configurable":{
                "thread_id": thread_id,
            }
        }
        final_answer = ""

        for event in self._agent.stream(
            cast(Any, agent_input),
            context=context,
            config=config,
            stream_mode=["messages", "updates"],
            version="v2"
        ):
            if event["type"] == "messages":
                chunk, metadata = event["data"]
                if metadata.get("langgraph_node") != "model":
                    continue

                if isinstance(chunk, AIMessageChunk) and chunk.text:
                    print(chunk.text, end="", flush=True)
            elif event["type"] == "updates":
                model_update = event["data"].get("model")

                if model_update is None:
                    continue
                message = model_update["messages"][-1]

                if (
                        isinstance(message, AIMessage)
                        and not message.tool_calls
                        and message.text
                ):
                    final_answer = message.text

        return final_answer
