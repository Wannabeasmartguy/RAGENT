from pydantic import BaseModel

from typing import Dict, Literal, List


class card(BaseModel):
    card_type: int
    template_url: str
    template_id: int
    template_h5: Dict
    response_for_model: Dict
    response_type: str

class Message(BaseModel):
    role: str
    type: Literal['function_call', 'tool_response', 'answer', 'verbose', 'follow_up']
    content: str | card
    content_type: Literal['text','card']

class CozeBotResponse(BaseModel):
    messages: List[Message]
    conversation_id: str
    code: int
    msg: str