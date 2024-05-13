from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from openai import OpenAI


class OpenAILikeLLM(LLM):
    """
    OpenAI-like LLM wrapper. Use OpenAI-Python SDK to call the API.

    Attention:
        llm_config(Dict) can only passed in when use "invoke" method.
        Like:
            llm.invoke(prompt,**llm_config)
    """

    # @classmethod
    # def set_config(cls, 
    #              api_key: str, 
    #              base_url: str,
    #              model: str,
    # ) -> None:
    #     """Initialize the OpenAILikeLLM instance."""
    #     # super().__init__()
    #     return cls(api_key=api_key, base_url=base_url, model=model)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the OpenAI API to get a completion for the prompt.
        
        Args:
            prompt (str): The prompt to pass into the API.
            stop (Optional[List[str]]): The stop sequence to use.
            run_manager (Optional[CallbackManagerForLLMRun]): A callback manager
                instance that can be used to track events.
            **kwargs: Additional keyword arguments to pass into the API.
                such as api_key, base_url, temperature, top_p, max_tokens, etc.
        """
        client = OpenAI(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
        )

        model_params = kwargs["params"]
        response = client.chat.completions.create(
            **model_params,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=kwargs.get("model")
        )
        return response.choices[0].message.content
    
    @property
    def _llm_type(self) -> str:
        return "openai-like"