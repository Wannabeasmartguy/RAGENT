from abc import ABC, abstractmethod
from autogen_agentchat.teams import BaseTeam
from autogen_ext.models import OpenAIChatCompletionClient

class BaseTeamBuilder(ABC):
    def __init__(self):
        self.model_client = None
    
    @abstractmethod
    def set_model_client(self, source: str, config_list: list) -> "BaseTeamBuilder":
        """设置模型客户端"""
        pass
    
    @abstractmethod
    def build(self) -> BaseTeam:
        """构建并返回一个团队"""
        pass 