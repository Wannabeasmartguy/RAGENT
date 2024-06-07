from pydantic import BaseModel
from typing import List


class Prompt(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#e63969ab"""
    prompt: str


class Onboarding(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#68108101"""
    prologue: str
    suggested_questions: List[str]


class API(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#d5d2762e"""
    api_id: str
    name: str
    description: str


class Plugin(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#0babec64"""
    plugin_id: str
    name: str
    description: str
    icon_url: str
    api_info_list: List[API]


class Model(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#24646564"""
    model_id: str
    model_name: str


class Bot_Single_Agent(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#fb1eead3"""
    bot_id: str
    name: str
    description: str
    icon_url: str
    create_time: int
    update_time: int
    version: str
    prompt_info: Prompt
    onboarding_info: Onboarding
    bot_mode: int
    plugin_info_list: List[Plugin]
    model_info: Model


class Bot_Multi_Agent(BaseModel):
    """Check https://www.coze.cn/docs/developer_guides/get_metadata#fb1eead3"""
    bot_id: str
    name: str
    description: str
    icon_url: str
    create_time: int
    update_time: int
    version: str
    prompt_info: Prompt
    onboarding_info: Onboarding
    bot_mode: int
    plugin_info_list: List[Plugin]
    model_info: List[Model]