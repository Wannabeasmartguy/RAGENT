from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from autogen_agentchat.teams import BaseGroupChat
from autogen_ext.models import OpenAIChatCompletionClient

T = TypeVar('T', bound=BaseGroupChat)

class BaseTeamBuilder(Generic[T], ABC):
    def __init__(self):
        self.model_client = None
    
    @abstractmethod
    def set_model_client(self, source: str, config_list: list) -> "BaseTeamBuilder[T]":
        """设置模型客户端"""
        pass
    
    @abstractmethod
    def build(self) -> T:
        """构建并返回一个团队"""
        pass 