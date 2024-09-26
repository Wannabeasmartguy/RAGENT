from abc import ABC, abstractmethod
from typing import Literal, List, Dict, Any, Generator

class OpenAILikeModelConfigProcessStrategy(ABC):
    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def update_config(self):
        pass

    @abstractmethod
    def delete_model_config(self):
        pass

    @abstractmethod
    def get_model_config(self):
        pass


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


class CozeChatProcessStrategy(ABC):
    @abstractmethod
    def create_coze_agent_response(self) -> Dict:
        pass

    @abstractmethod
    def get_bot_config(self) -> Dict:
        pass