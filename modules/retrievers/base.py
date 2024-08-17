import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Base class for all retrievers"""
    def __init__(self, docs: List[str]):
        self.docs = [{'page_content': doc} for doc in docs]

    @abstractmethod
    def invoke(self, query: str) -> List[Dict[str, Any]]:
        return self.docs

    @abstractmethod
    async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
        return self.docs