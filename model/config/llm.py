from datetime import datetime
from typing import Optional, Any, Dict, Literal
from pydantic import BaseModel, ConfigDict


class OpenAIConfiguration(BaseModel):
    """openai llm config that is stored in the database"""
    model_name: Optional[str] = None
    '''The model name to use for the llm'''
    api_key: Optional[str] = None
    '''The API key to use for the llm'''
    model_data: Optional[Optional[Dict[str, Any]]] = None
    '''Other model data to use for the llm'''
    created_at: Optional[datetime] = None
    '''The timestamp of when this run was created'''
    updated_at: Optional[datetime] = None
    '''The timestamp of when this run was last updated'''

    model_config = ConfigDict(from_attributes=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict


class AzureOpenAILLMConfiguration(BaseModel):
    """openai like llm config that is stored in the database"""
    model_name: Optional[str] = None
    '''The model name to use for the llm'''
    api_key: Optional[str] = None
    '''The API key to use for the llm'''
    base_url: Optional[str] = None
    '''The API base to use for the llm'''
    api_version: Optional[str] = None
    '''The API version to use for the llm'''
    api_type: Optional[Literal["azure"]] = None
    '''The API type to use for the llm'''
    model_data: Optional[Optional[Dict[str, Any]]] = None
    '''Other model data to use for the llm'''
    created_at: Optional[datetime] = None
    '''The timestamp of when this run was created'''
    updated_at: Optional[datetime] = None
    '''The timestamp of when this run was last updated'''

    model_config = ConfigDict(from_attributes=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict


class OpenAILikeLLMConfiguration(BaseModel):
    """openai like llm config that is stored in the database"""
    model_name: Optional[str] = None
    '''The model name to use for the llm'''
    api_key: Optional[str] = None
    '''The API key to use for the llm'''
    base_url: Optional[str] = None
    '''The API base to use for the llm'''
    model_data: Optional[Optional[Dict[str, Any]]] = None
    '''Other model data to use for the llm'''
    created_at: Optional[datetime] = None
    '''The timestamp of when this run was created'''
    updated_at: Optional[datetime] = None
    '''The timestamp of when this run was last updated'''

    model_config = ConfigDict(from_attributes=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict