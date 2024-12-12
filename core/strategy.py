from abc import ABC, abstractmethod
from typing import Literal, List, Dict, Any, Generator


class DialogProcessStrategy(ABC):
    @abstractmethod
    def update_dialog_name(self):
        pass

    @abstractmethod
    def update_dialog_config(self):
        pass

    @abstractmethod
    def update_chat_history(self):
        pass

    @abstractmethod
    def create_dialog(self):
        pass

    @abstractmethod
    def delete_dialog(self):
        pass

    @abstractmethod
    def get_dialog(self):
        pass

    @abstractmethod
    def get_all_dialogs(self):
        pass


class EncryptorStrategy(ABC):
    @abstractmethod
    def encrypt(self, data: str) -> str:
        pass

    @abstractmethod
    def decrypt(self, data: str) -> str:
        pass


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


class RAGChatProcessStrategy(ABC):
    @abstractmethod
    def create_custom_rag_response(self):
        pass


class CozeChatProcessStrategy(ABC):
    @abstractmethod
    def create_coze_agent_response(self) -> Dict:
        pass

    @abstractmethod
    def get_bot_config(self) -> Dict:
        pass
