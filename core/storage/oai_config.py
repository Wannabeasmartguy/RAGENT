import os
from typing import List, Optional, Dict
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.models.llm import OpenAILikeConfigInStorage
from core.encryption import FernetEncryptor
from config.constants import (
    OPENAI_LIKE_CONFIGS_BASE_DIR, 
    OPENAI_LIKE_CONFIGS_DB_FILE, 
    OPENAI_LIKE_CONFIGS_DB_TABLE
)
from utils.log.logger_config import setup_logger
from loguru import logger

Base = declarative_base()

class OpenAIConfigDB(Base):
    __tablename__ = OPENAI_LIKE_CONFIGS_DB_TABLE
    
    config_id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    base_url = Column(Text, nullable=False)
    api_key = Column(Text, nullable=False)
    custom_name = Column(Text)
    description = Column(Text)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class OpenAIConfigSQLiteStorage:
    def __init__(self, db_path: str = OPENAI_LIKE_CONFIGS_DB_FILE):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.encryptor = FernetEncryptor()

    def _encrypt_config(self, config: OpenAILikeConfigInStorage) -> Dict:
        encrypted = config.model_dump()
        encrypted["base_url"] = self.encryptor.encrypt(config.base_url)
        encrypted["api_key"] = self.encryptor.encrypt(config.api_key)
        return encrypted

    def _decrypt_config(self, db_config: Dict) -> OpenAILikeConfigInStorage:
        db_config["base_url"] = self.encryptor.decrypt(db_config["base_url"])
        db_config["api_key"] = self.encryptor.decrypt(db_config["api_key"])
        return OpenAILikeConfigInStorage(**db_config)

    def save_config(self, config: OpenAILikeConfigInStorage) -> str:
        encrypted = self._encrypt_config(config)
        with self.Session() as session:
            db_config = OpenAIConfigDB(**encrypted)
            session.add(db_config)
            session.commit()
            return db_config.config_id

    def get_config(self, user_id: str, config_id: str) -> Optional[OpenAILikeConfigInStorage]:
        with self.Session() as session:
            result = session.query(OpenAIConfigDB).filter(
                OpenAIConfigDB.config_id == config_id,
                OpenAIConfigDB.user_id == user_id
            ).first()
            if result:
                return self._decrypt_config(result.__dict__)
            return None

    def list_configs(self, user_id: str) -> List[OpenAILikeConfigInStorage]:
        with self.Session() as session:
            results = session.query(OpenAIConfigDB).filter(
                OpenAIConfigDB.user_id == user_id
            ).all()
            return [self._decrypt_config(r.__dict__) for r in results]

    def delete_config(self, user_id: str, config_id: str) -> bool:
        with self.Session() as session:
            deleted = session.query(OpenAIConfigDB).filter(
                OpenAIConfigDB.config_id == config_id,
                OpenAIConfigDB.user_id == user_id
            ).delete()
            session.commit()
            return deleted > 0 