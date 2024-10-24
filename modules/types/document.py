from pydantic import BaseModel, Field
from typing import Any, Literal, Optional


class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadatas: Optional[dict] = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    relevant_score: Optional[float] = None
    """Relevance score of the document to a given query."""
    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg."""
        super().__init__(page_content=page_content, **kwargs)