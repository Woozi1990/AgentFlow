import base64
import mimetypes

from langchain_core.messages import HumanMessage

from models import OpenAIModel
from prompt import IMAGE_DESCRIPTION_SYSTEM_PROMPT


class ImageDescriptionTool:
    def __init__(self, llm: OpenAIModel):
        self.llm = llm

    def execute(self, image_path: str):
        # print(f"[Tool] Reading image : {image_path}")
        base64_image = ""
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        image_url = f"data:{mime_type};base64,{base64_image}"
        describe_prompt = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Describe the main content of this image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            )
        ]
        response = self.llm.call_llm(
            system_prompt=IMAGE_DESCRIPTION_SYSTEM_PROMPT,
            messages=describe_prompt,
        )

        return {
            "success": True,
            "image_path": image_path,
            "description": response.content
        }
