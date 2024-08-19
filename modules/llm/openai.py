from openai import OpenAI
from typing import List, Dict, Generator, Union, Optional

class OpenAILLM:
    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
        ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def invoke(
            self, 
            messages: List[Dict[str,str]], 
            *,
            max_tokens: int = 2048,
            temperature: float = 0.5,
            top_p: float = 0.5,
            stream: bool = False,
        ) -> Dict | Generator:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream
        )
        return response