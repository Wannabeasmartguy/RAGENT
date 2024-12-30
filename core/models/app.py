from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, Field


class BaseChatState(BaseModel):
    """Base class for chat state"""

    current_run_id: str
    user_id: Optional[str] = Field(default=None)
    run_name: Optional[str] = Field(default=None)
    user_data: Optional[Dict[str, Any]] = Field(default=None)

    current_run_index: Optional[int] = Field(default=None)


class ClassicChatState(BaseChatState):
    """Classic chat state used in streamlit chat page"""

    config_list: Optional[List[Dict[str, Any]]] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    llm_model_type: Optional[str] = Field(default=None)
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=None)


class RAGChatState(BaseChatState):
    """RAG chat state used in streamlit RAG chat page"""

    config_list: Optional[List[Dict[str, Any]]] = Field(default=None)
    llm_model_type: Optional[str] = Field(default=None)
    source_documents: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(default=None)
    chat_history: Optional[List[Dict[str, Any]]] = Field(default=None)


class AgentChatState(BaseChatState):
    """Agent chat state used in streamlit Agent chat page"""

    pass