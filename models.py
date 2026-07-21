from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_BASE_URL, AZURE_OPENAI_MODEL


class OpenAIModel:
    def __init__(self, ):
        self._model = ChatOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=AZURE_OPENAI_BASE_URL,
            model=AZURE_OPENAI_MODEL,
            temperature=1.2
        )

    @property
    def model(self) -> BaseChatModel:
        return self._model

    def call_llm(self,
                 system_prompt: str,
                 messages: list[BaseMessage],
                 model=None
                 ):
        request_messages = [
            SystemMessage(content=system_prompt),
            *messages,
        ]
        selected_model = model or self._model
        return selected_model.invoke(input=request_messages)
