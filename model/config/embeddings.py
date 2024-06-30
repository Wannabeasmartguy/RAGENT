from datetime import datetime
from typing import Optional, Any, Dict, Literal
from pydantic import BaseModel, ConfigDict


class EmbeddingConfiguration(BaseModel):
    """embedding config that is stored in the database"""
    user_id: str
    '''The user id of the user who created this embedding'''
    model_id: str
    '''The model id of the model that this embedding is for'''
    embedding_type: str
    '''The type of embedding to use'''
    embedding_model_name_or_path: str
    '''The model name or path to use for the embedding'''
    device: Optional[str] = None
    '''The device to use for the embedding'''
    api_key: Optional[str] = None
    '''The API key to use for the embedding'''
    base_url: Optional[str] = None
    '''The API base to use for the embedding'''
    api_type: Literal['azure', 'openai'] = None
    '''The API type to use for the embedding, only used in (azure) openai embeddings'''
    api_version: Optional[str] = None
    '''The API version to use for the embedding, only used in azure openai embeddings'''
    max_seq_length: Optional[int] = None
    '''The maximum sequence length for the embedding'''
    additional_model_data: Optional[str] = None
    '''Other model data to use for the embedding'''
    created_at: Optional[datetime] = None
    '''The timestamp of when this run was created'''
    updated_at: Optional[datetime] = None
    '''The timestamp of when this run was last updated'''

    model_config = ConfigDict(from_attributes=True,protected_namespaces=())

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

class CollectionEmbeddingConfiguration(EmbeddingConfiguration):
    """knowledgebase collection embedding config that is stored in the database"""
    collection_name: str
    '''The name of the collection to use for the embedding'''
    collection_id: Optional[str] = None
    '''The id of the collection to use for the embedding'''