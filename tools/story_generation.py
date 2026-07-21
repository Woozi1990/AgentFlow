from langchain_core.messages import HumanMessage

from models import OpenAIModel
from prompt import STORY_SYSTEM_PROMPT


class StoryGenerationTool:
    def __init__(self, llm: OpenAIModel):
        self.llm = llm

    def execute(self, prompt: str, language: str):
        # print(f"[Tool] Generating story from: {prompt}")

        generate_prompt = [
            HumanMessage(
                content=(
                    f"Target language: {language}\n\n"
                    f"Story prompt: \n{prompt}"
                )
            ),
        ]

        response = self.llm.call_llm(
            system_prompt=STORY_SYSTEM_PROMPT,
            messages=generate_prompt,
        )
        return {
            "success": True,
            "story": response.content,
            "language": language,
        }
