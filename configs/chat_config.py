from typing import Literal, List, Dict, Any
from abc import ABC, abstractmethod

from api.dependency import APIRequestHandler,SUPPORTED_SOURCES


class ChatProcessStrategy(ABC):
    @abstractmethod
    def create_completion(self, messages: List[Dict[str, str]]) -> Dict:
        pass


class AgentChatProcessoStrategy(ABC):
    @abstractmethod
    def create_reflection_agent_response(self) -> Dict:
        pass

    @abstractmethod
    def create_rag_agent_response(self) -> Dict:
        pass

    @abstractmethod
    def create_base_chat_agent_response(self) -> Dict:
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


class AgentChatProcessor(AgentChatProcessoStrategy):
    """
    处理 Agent Chat 消息的策略模式实现类
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
        
    def create_rag_agent_response(
        self,
        name: str,
        messages: List[Dict[str, str]],
        is_rerank: bool,
        is_hybrid_retrieve: bool,
        hybrid_retriever_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        通过 requesthandler ，根据 model_type ，创建一个agentchat的lc-rag响应

        Args:
            name (str): 使用到的 Chroma collection 的名称
            messages (List[Dict[str, str]]): 对话消息列表
            is_rerank (bool): 是否进行重排序
            is_hybrid_retrieve (bool): 是否进行混合检索
            hybrid_retriever_weight (float): 混合检索的权重
        """
        if self.model_type.lower() == "llamafile":
            response = self.requesthandler.post(
                endpoint="/agentchat/lc-rag/create-rag-response",
                data={
                    "messages": messages,
                    "llm_config": self.llm_config,
                    "llm_params": self.llm_config.get(
                        "params",
                        {
                            "temperature": 0.5,
                            "top_p": 0.1,
                            "max_tokens": 4096
                        }
                    )
                },
                params={
                    "name": name,
                    "is_rerank": is_rerank,
                    "is_hybrid_retrieve": is_hybrid_retrieve,
                    "hybrid_retriever_weight": hybrid_retriever_weight,
                }
            )
            return response
    
    def create_base_chat_agent_response(self) -> Dict:
        pass

    def create_reflection_agent_response(self) -> Dict:
        pass