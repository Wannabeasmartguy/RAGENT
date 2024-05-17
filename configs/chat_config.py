from typing import Literal, List, Dict
from abc import ABC, abstractmethod

from api.dependency import APIRequestHandler,SUPPORTED_SOURCES


class ChatProcessStrategy(ABC):
    @abstractmethod
    def create_completion(self, messages: List[Dict[str, str]]) -> Dict:
        pass


class ChatProcessor(ChatProcessStrategy):
    """
    处理聊天消息的策略模式实现类
    """
    def __init__(
            self, 
            requesthandler: APIRequestHandler,
            model_type: str,
            llm_config: Dict
        ) -> None:
        self.requesthandler = requesthandler
        self.model_type = model_type
        self.llm_config = llm_config

    def create_completion(
            self, 
            messages: List[Dict[str, str]],
            
        ) -> Dict:
        """
        通过 requesthandler ，根据 model_type ，调用相应的 API 接口
        """
        if self.model_type.lower() in SUPPORTED_SOURCES["sources"]:

            # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中的对应值为 "sdk" ，则走 OpenAI 的 SDK
            if SUPPORTED_SOURCES["sources"][self.model_type.lower()] == "sdk":
                path = "/chat/openai-like-chat/openai"

            # 否则，走 request 或另行定义的 SDK （如 Groq）
            else:
                # path = "/chat/openai-like-chat/xxxx"
                pass

        response = self.requesthandler.post(
            path,
            data={
                "llm_config": self.llm_config,
                "llm_params": self.llm_config.get(
                    "params",
                    {
                        "temperature": 0.5,
                        "top_p": 0.1,
                        "max_tokens": 4096
                    }
                ),
                "messages": messages
            }
        )
        
        return response