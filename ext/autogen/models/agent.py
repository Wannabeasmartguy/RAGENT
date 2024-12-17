from typing import List, Dict, Any
from uuid import uuid4

from core.model.llm import LLMBaseConfig, LLMConfigType

from pydantic import BaseModel, Field

class ReflectionAgentTeamTemplate(BaseModel):
    """用于创建reflection agent的team template"""
    id: str = Field(default=str(uuid4()), description="Template id. It is used to identify the template for user, not used in generation.")
    name: str = Field(default="Default Template", description="Template name. It is used to identify the template for user, not used in generation.")
    description: str = Field(default="No description", description="Template description. It is used to describe the template for user, not used in generation.")

    llm: LLMConfigType = Field(..., description="LLM config")
    primary_agent_system_message: str = Field(..., description="Primary agent system message")
    critic_agent_system_message: str = Field(..., description="Critic agent system message")
    max_messages: int = Field(..., description="Max messages")
    termination_text: str = Field(..., description="Termination text")

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return cls(**config)
