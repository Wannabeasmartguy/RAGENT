from pydantic import BaseModel, Field, ValidationError, field_validator
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import Stream
from typing import Dict, Any, Literal, Union

class BaseRAGResponse(BaseModel):
    """Response model for RAG responses"""
    response_id: str
    """The id of the response"""
    response_type: Literal['RAGResponse'] = 'RAGResponse'
    """The type of the response"""
    answer: Union[ChatCompletion, Stream[ChatCompletionChunk], Dict]
    """The answer to the question, in the form of a dictionary"""

    source_documents: Dict[str, Any]
    """The source documents used to generate the answer, include 3 fields: `result`, `page_content`, `metadatas`"""

    @field_validator('source_documents')
    def validate_source_documents(cls, value):
        if isinstance(value, dict):
            if 'result' not in value or 'page_content' not in value or 'metadatas' not in value:
                raise ValueError("The dict must contain 'result', 'page_content' and 'metadatas' keys")
        return value

    @field_validator('answer')
    def validate_answer_dict(cls, value):
        if isinstance(value, dict):
            if 'role' not in value or 'content' not in value:
                raise ValueError("The dict must contain 'role' and 'content' keys")
        return value
    
    class Config:
        arbitrary_types_allowed = True


if __name__ == '__main__':
    try:
        response = BaseRAGResponse(
            answer={
                "role": "assistant",
                "content": "This is a test answer."
            },
            source_documents={
                "result": "This is a test vecstore query result.",
                "page_content": "This is a test page content.",
                "metadatas": [{
                    "title": "This is a test title.",
                    "url": "https://example.com"
                }]
            }
        )
        print(response)
        print(type(response.answer))
    except ValidationError as e:
        print(e.json())