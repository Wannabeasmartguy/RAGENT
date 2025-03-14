from typing import List, Dict, Any, Literal, TypeVar, Union
from uuid import uuid4
from enum import Enum
from abc import abstractmethod
from core.models.llm import (
    OpenAIConfig,
    AzureOpenAIConfig,
    OllamaConfig,
    OpenAILikeConfig,
    GroqConfig,
)

from pydantic import BaseModel, Field


class BaseAgentTemplate(BaseModel):
    """用于创建agent的template"""
    user_id: str = Field(..., description="User who the template belongs to."),
    id: str = Field(
        default=str(uuid4()),
        description="Template id. It is used to identify the template for user, not used in generation.",
    )
    name: str = Field(
        default="Default Template",
        description="Template name. It is used to identify the template for user, not used in generation.",
    )
    description: str = Field(
        default="No description",
        description="Template description. It is used to describe the template for user, not used in generation.",
    )

    @abstractmethod
    def to_dict(self) -> dict:
        """将实例转换为字典，并在llm中添加config_type"""
        pass


AgentTemplate = TypeVar("AgentTemplate", bound=BaseAgentTemplate)


class ReflectionAgentTeamTemplate(BaseAgentTemplate):
    """用于创建reflection agent的team template"""

    llm: Union[AzureOpenAIConfig, OpenAIConfig, OpenAILikeConfig, OllamaConfig, GroqConfig] = Field(..., description="LLM config")
    primary_agent_system_message: str = Field(
        ..., description="Primary agent system message"
    )
    critic_agent_system_message: str = Field(
        ..., description="Critic agent system message"
    )
    max_messages: int = Field(..., description="Max messages")
    termination_text: str = Field(..., description="Termination text")

    team_type: Literal["reflection"] = "reflection"

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
    
    def to_dict(self) -> dict:
        """将实例转换为字典，并在llm中添加config_type"""
        raw_dict = self.model_dump()
        raw_dict["llm"]["config_type"] = self.llm.config_type()
        return raw_dict


class AgentTemplateType(Enum):
    """Agent团队类型的枚举"""
    REFLECTION = "reflection"
    # 未来可以添加更多类型
    # DEBATE = "debate"
    # QA = "qa"
    # 等等...
