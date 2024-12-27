from datetime import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, ConfigDict

class BaseChatState(BaseModel):
    """Base class for chat state"""
    current_run_id: str
    user_id: Optional[str] = None
    run_name: Optional[str] = None
    user_data: Optional[Dict[str, Any]] = None
    
    current_run_index: Optional[int] = None


class ClassicChatState(BaseChatState):
    """Classic chat state used in streamlit chat page"""
    config_list: Optional[List[Dict[str, Any]]] = None
    system_prompt: Optional[str] = None
    llm_model_type: Optional[str] = None

class RAGChatState(BaseChatState):
    """RAG chat state used in streamlit RAG chat page"""
    config_list: Optional[List[Dict[str, Any]]] = None
    llm_model_type: Optional[str] = None


class AgentChatState(BaseChatState):
    """Agent chat state used in streamlit Agent chat page"""
    pass

