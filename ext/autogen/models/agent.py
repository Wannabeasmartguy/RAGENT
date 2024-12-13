from core.model.llm import LLMBaseConfig

from pydantic import BaseModel, Field

class ReflectionAgentTeamTemplate(BaseModel):
    """用于创建reflection agent的team template"""
    llm: LLMBaseConfig = Field(..., description="LLM config")
    primary_agent_system_message: str = Field(..., description="Primary agent system message")
    critic_agent_system_message: str = Field(..., description="Critic agent system message")
    max_messages: int = Field(..., description="Max messages")
    termination_text: str = Field(..., description="Termination text")
