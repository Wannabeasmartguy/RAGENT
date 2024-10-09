from datetime import datetime
from typing import Optional, Any, Dict, Literal, List
from pydantic import BaseModel, ConfigDict, Field

class EmbeddingModelConfiguration(BaseModel):
    """嵌入模型配置"""
    id: str
    name: str
    embedding_type: str
    embedding_model_name_or_path: str
    device: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_type: Optional[Literal['azure', 'openai']] = None
    api_version: Optional[str] = None
    max_seq_length: Optional[int] = None
    additional_model_data: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class KnowledgeBaseConfiguration(BaseModel):
    """知识库配置"""
    id: str
    name: str
    embedding_model_id: str

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class GlobalSettings(BaseModel):
    """全局设置"""
    default_model: str

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class EmbeddingConfiguration(BaseModel):
    """完整的嵌入配置"""
    global_settings: GlobalSettings
    models: List[EmbeddingModelConfiguration]
    knowledge_bases: List[KnowledgeBaseConfiguration]
    user_id: Optional[str] = None
    created_at: Optional[datetime] = datetime.now()
    updated_at: Optional[datetime] = datetime.now()

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

class CollectionEmbeddingConfiguration(EmbeddingConfiguration):
    """知识库集合嵌入配置"""
    collection_name: str
    '''The name of the collection to use for the embedding'''
    collection_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())