from abc import ABC, abstractmethod
from typing import Optional, Sequence

from pydantic import BaseModel

from modules.types.document import Document

class BaseDocumentCompressor(BaseModel, ABC):
    """Base class for document compressors."""

    @abstractmethod
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return await self.compress_documents(documents, query)