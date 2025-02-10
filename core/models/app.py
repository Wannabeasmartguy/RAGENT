from typing import Optional, Any, Dict, List, Union, Literal
from pydantic import BaseModel, Field


class BaseChatState(BaseModel):
    """Base class for chat state"""

    current_run_id: str
    user_id: Optional[str] = Field(default=None)
    run_name: Optional[str] = Field(default=None)
    user_data: Optional[Dict[str, Any]] = Field(default=None)

    current_run_index: Optional[int] = Field(default=None)


class BaseMessage(BaseModel):
    """Base message model for all types of messages"""
    role: Literal["user", "assistant", "system"]
    content: str
    reasoning_content: Optional[str] = None

class UserMessage(BaseMessage):
    """User message model"""
    role: Literal["user"] = "user"
    images: Optional[Any] = None

class AssistantMessage(BaseMessage):
    """Assistant message model"""
    role: Literal["assistant"] = "assistant"
    function_call: Optional[Dict[str, Any]] = None

class SystemMessage(BaseMessage):
    """System message model"""
    role: Literal["system"] = "system"

MessageType = Union[UserMessage, AssistantMessage, SystemMessage]


class ClassicChatState(BaseChatState):
    """Classic chat state used in streamlit chat page"""

    config_list: Optional[List[Dict[str, Any]]] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    llm_model_type: Optional[str] = Field(default=None)
    chat_history: Optional[List[MessageType]] = Field(default_factory=list)


class KnowledgebaseConfigInRAGChatState(BaseModel):
    """Knowledge base config in RAG chat state"""
    collection_name: Optional[str]
    query_mode: Literal["file", "collection"] = Field(default="collection", description="The query mode of the knowledge base")
    selected_file: Optional[str] = Field(default=None, description="The file name of the selected file in the knowledge base, only used when query_mode is 'file'")
    is_rerank: bool = Field(default=False, description="Whether to use rerank document chunks retrieved from the knowledge base")
    is_hybrid_retrieve: bool = Field(default=False, description="Whether to use hybrid retrieve document chunks in the knowledge base")
    hybrid_retrieve_weight: float = Field(default=0.5, description="The weight of the hybrid retrieve document chunks in the knowledge base")


class RAGChatState(BaseChatState):
    """RAG chat state used in streamlit RAG chat page"""

    config_list: Optional[List[Dict[str, Any]]] = Field(default=None)
    llm_model_type: Optional[str] = Field(default=None)
    knowledge_base_config: Optional[KnowledgebaseConfigInRAGChatState] = Field(default=None)
    source_documents: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(default=None)
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=None)


class AgentChatState(BaseChatState):
    """Agent chat state used in streamlit Agent chat page"""

    template: Optional[Dict[str, Any]] = Field(default=None, description="Agent team template created by user")
    agent_state: Optional[Dict[str, Any]] = Field(default=None, description="Agent state, generated during the conversation between user and agent")
    team_state: Optional[Dict[str, Any]] = Field(default=None, description="Team state, generated during the conversation between user and agent team")
