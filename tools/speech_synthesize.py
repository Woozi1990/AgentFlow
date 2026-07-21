from typing import Any

from gtts import gTTS


class SpeechSynthesisTool:
    def __init__(self):
        pass


    def execute(self, text: str, lang:str, output_path:str) -> dict[str, Any]:
        # print(f"[Tool] Synthesizing speech : {text[:40]}...")

        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)

        return {
            "success": True,
            "audio_path": output_path
        }