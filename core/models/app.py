from typing import Optional, Any, Dict, List, Union, Literal, Set
from pydantic import BaseModel, Field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum


class SerializationMode(Enum):
    """消息序列化模式，决定序列化时包含哪些字段"""
    FULL = "full"  # 完整序列化，包含所有字段
    STORAGE = "storage"  # 存储模式
    MODEL = "model"  # 模型交互模式，不包含时间戳和推理内容
    EXPORT = "export"  # 导出模式


class BaseChatState(BaseModel):
    """Base class for chat state"""

    current_run_id: str
    user_id: Optional[str] = Field(default=None)
    run_name: Optional[str] = Field(default=None)
    user_data: Optional[Dict[str, Any]] = Field(default=None)

    current_run_index: Optional[int] = Field(default=None)


class BaseMessage(BaseModel, ABC):
    """Base message model for all types of messages"""
    role: Literal["user", "assistant", "system"]
    content: str
    reasoning_content: Optional[str] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @property
    def sensitive_fields(self) -> Set[str]:
        """需要特殊处理的敏感字段"""
        return {'created_at', 'updated_at', 'reasoning_content'}
    
    def to_dict(self, mode: SerializationMode = SerializationMode.FULL) -> Dict[str, Any]:
        """转换消息为字典，根据不同模式选择性地排除字段
        
        Args:
            mode: 序列化模式，决定包含哪些字段
            
        Returns:
            Dict[str, Any]: 消息的字典表示
        """
        exclude = set()
        
        if mode == SerializationMode.MODEL:
            exclude.update({'created_at', 'updated_at', 'reasoning_content'})
        elif mode == SerializationMode.STORAGE:
            pass
        elif mode == SerializationMode.EXPORT:
            pass
            
        return self.model_dump(exclude=exclude)
    
    @abstractmethod
    def validate_content(self) -> bool:
        """验证消息内容的有效性"""
        pass


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: str


class UserMessage(BaseMessage):
    """User message model"""
    role: Literal["user"] = "user"
    content: List[Union[TextContent, ImageContent]]
    
    def validate_content(self) -> bool:
        """验证用户消息内容"""
        return len(self.content) > 0 and all(
            isinstance(c, (TextContent, ImageContent)) for c in self.content
        )


class AssistantMessage(BaseMessage):
    """Assistant message model"""
    role: Literal["assistant"] = "assistant"
    function_call: Optional[Dict[str, Any]] = None
    
    def validate_content(self) -> bool:
        """验证助手消息内容"""
        return bool(self.content)


class SystemMessage(BaseMessage):
    """System message model"""
    role: Literal["system"] = "system"
    
    def validate_content(self) -> bool:
        """验证系统消息内容"""
        return bool(self.content)


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
