from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class Node(BaseModel):
    id: str
    type: str
    properties: Dict[str, Any] = {}

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str
    properties: Dict[str, Any] = {}

class GraphDocument(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]
    source: Any = None

class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]

class UnstructuredRelation(BaseModel):
    head: str = Field(
        description="extracted head entity like Microsoft, Apple, John. "
        "Must use human-readable unique identifier."
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description="extracted tail entity like Microsoft, Apple, John. "
        "Must use human-readable unique identifier."
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )