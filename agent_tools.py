from langchain.tools import tool, ToolRuntime
from langchain_core.tools import BaseTool

from agent_context import AgentContext
from models import OpenAIModel
from tools.image_description import ImageDescriptionTool
from tools.speech_synthesize import SpeechSynthesisTool
from tools.story_generation import StoryGenerationTool


def build_tools(llm: OpenAIModel) -> list[BaseTool]:
    image_description_service = ImageDescriptionTool(llm)
    story_generation_service = StoryGenerationTool(llm)
    speech_synthesis_service = SpeechSynthesisTool()

    @tool
    def describe_image(runtime: ToolRuntime[AgentContext]) -> dict:
        """Describe the main content of the image supplied by the user."""
        image_path = runtime.context.image_path
        return image_description_service.execute(image_path)

    @tool
    def generate_story(prompt: str, language: str) -> dict:
        """Generate a story based on the supplied description or prompt.

        Args:
            prompt: The content, description, setting, or instructions on which
                the story should be based.
            language: The language in which the complete story must be written,
                such as 'English' or 'Simplified Chinese'.
        """
        return story_generation_service.execute(
            prompt=prompt,
            language=language,
        )

    @tool
    def synthesize_speech(text: str, lang: str, runtime: ToolRuntime[AgentContext]) -> dict:
        """Convert text into speech.

        Args:
            text: The complete text to convert into speech.
            lang: A gTTS-supported language code, such as
                'en' for English or 'zh-CN' for Simplified Chinese.
        """
        output_path = runtime.context.output_path
        return speech_synthesis_service.execute(text, lang, output_path)

    return [
        describe_image,
        generate_story,
        synthesize_speech
    ]
