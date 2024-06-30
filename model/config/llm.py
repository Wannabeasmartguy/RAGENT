from datetime import datetime
from typing import Optional, Any, Dict, Literal
from pydantic import BaseModel, ConfigDict


class LLMConfiguration(BaseModel):
    """Base class for llm config that is stored in the database"""
    model_id: Optional[str] = None
    '''The model id to distinguish between different configurations of the same llm model'''
    model_name: Optional[str] = None
    '''The model name to use for the llm'''
    user_id: Optional[str] = None
    '''The user id to use for the llm'''
    api_key: Optional[str] = None
    '''The API key to use for the llm'''
    additional_model_data: Optional[Dict[str, Any]] = None
    '''Other model data to use for the llm'''
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


class OpenAIConfiguration(LLMConfiguration):
    """openai llm config that is stored in the database"""
    pass

class AzureOpenAILLMConfiguration(LLMConfiguration):
    """openai like llm config that is stored in the database"""
    base_url: Optional[str] = None
    '''The API base to use for the llm'''
    api_version: Optional[str] = None
    '''The API version to use for the llm'''
    api_type: Optional[Literal["azure"]] = None
    '''The API type to use for the llm'''


class OpenAILikeLLMConfiguration(LLMConfiguration):
    """openai like llm config that is stored in the database"""
    base_url: Optional[str] = None
    '''The API base to use for the llm'''