from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, ConfigDict


class AssistantRun(BaseModel):
    """Assistant Run that is stored in the database"""

    name: Optional[str] = None
    """Assistant name"""
    run_id: str
    """Run UUID"""
    run_name: Optional[str] = None
    """Run name"""
    user_id: Optional[str] = None
    """ID of the user participating in this run"""
    llm: Optional[Dict[str, Any]] = None
    """LLM data (name, model, etc.)"""
    # {
    #     'chat_history': [
    #         {
    #             'role': 'user', 
    #             'content': '谁是我的哥哥', 
    #             'metrics': {}
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': '要确定谁是你的哥哥，需要了解你的家庭情况和家庭成员。哥哥通常指的是与你同父同母或同父异母、同母异父的比你年长的男性亲属。如果你有多个哥哥，那么他们都是你的哥哥。',
    #             'metrics': {}
    #         }
    #     ],
    #     'llm_messages': [
    #         {
    #             'role': 'user', 
    #             'content': '谁是我的哥哥', 
    #             'metrics': {}
    #         },
    #         {
    #             'role': 'assistant',
    #             'content': '要确定谁是你的哥哥，需要了解你的家庭情况和家庭成员。哥哥通常指的是与你同父同母或同父异母、同母异父的比你年长的男性亲属。如果你有多个哥哥，那么他们都是你的哥哥。',
    #             'metrics': {
    #                 'time': 9.861949400001322,
    #                 'prompt_tokens': 11,
    #                 'completion_tokens': 107,
    #                 'total_tokens': 118
    #             }
    #         }
    #     ],
    #     'references': [],
    #     'user_id': 'user',
    #     'retrieval': 'last_n'
    # }
    memory: Optional[Dict[str, Any]] = None
    """Assistant Memory"""
    assistant_data: Optional[Dict[str, Any]] = None
    """Metadata associated with this assistant"""
    run_data: Optional[Dict[str, Any]] = None
    """Metadata associated with this run"""
    user_data: Optional[Dict[str, Any]] = None
    """Metadata associated the user participating in this run"""
    task_data: Optional[Dict[str, Any]] = None
    """Metadata associated with the assistant tasks"""
    created_at: Optional[datetime] = None
    """The timestamp of when this run was created"""
    updated_at: Optional[datetime] = None
    """The timestamp of when this run was last updated"""

    model_config = ConfigDict(from_attributes=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

    def assistant_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at", "task_data"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict