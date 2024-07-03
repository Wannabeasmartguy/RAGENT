from abc import ABC, abstractmethod
from typing import List, Optional
try:
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.engine import create_engine, Engine, Row
    from sqlalchemy.inspection import inspect
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.schema import MetaData, Table, Column
    from sqlalchemy.sql.expression import text, select, delete
    from sqlalchemy.types import DateTime, String, Integer
except ImportError:
    raise ImportError("`sqlalchemy` not installed")

class Sqlstorage(ABC):
    @abstractmethod
    def get_table(self) -> Table:
        pass

    @abstractmethod
    def table_exists(self) -> bool:
        pass

    @abstractmethod
    def create(self) -> None:
        pass
    
    @abstractmethod
    def read(self, id: str) -> Row:
        pass

    @abstractmethod
    def upsert(self, row: Row) -> Optional[Row]:
        pass
    
    @abstractmethod
    def delete_table(self) -> None:
        pass