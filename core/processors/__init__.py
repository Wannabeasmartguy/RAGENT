from typing import TypeVar

from .chat.classic import ChatProcessor
from .chat.rag import RAGChatProcessor
from .config.llm import OAILikeConfigProcessor
from .dialog.dialog_processors import (
    BaseDialogProcessor,
    ClassicChatDialogProcessor, 
    RAGChatDialogProcessor
)
from .vector.chroma.kb_processors import (
    ChromaVectorStoreProcessorWithNoApi,
    ChromaCollectionProcessorWithNoApi,
)

ALLDIAGLOGPROCESSOR = TypeVar('ALLDIAGLOGPROCESSOR', bound=BaseDialogProcessor)

__all__ = [
    'ChatProcessor',
    'RAGChatProcessor',
    'OAILikeConfigProcessor',
    'ClassicChatDialogProcessor',
    'RAGChatDialogProcessor',
    'ChromaVectorStoreProcessorWithNoApi',
    'ChromaCollectionProcessorWithNoApi',
]
