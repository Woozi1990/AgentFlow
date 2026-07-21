from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    image_path: str
    output_path: str
