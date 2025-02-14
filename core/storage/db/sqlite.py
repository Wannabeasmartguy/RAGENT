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

from sqlite3 import OperationalError
from datetime import datetime
from typing import Optional, List, Literal, Any, Dict
from loguru import logger
from sqlalchemy.schema import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from core.models.memory import AssistantRun
from core.storage.db.base import Sqlstorage
from utils.log.logger_config import setup_logger
from core.encryption import FernetEncryptor
from core.strategy import EncryptorStrategy
import json
import time


class SqlAssistantStorage(Sqlstorage):
    def __init__(
        self,
        table_name: str,
        db_url: Optional[str] = None,
        db_file: Optional[str] = None,
        db_engine: Optional[Engine] = None,
        encryptor: Optional[EncryptorStrategy] = None,
    ):
        """
        This class provides assistant storage using a sqlite database.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url
            3. Use the db_file
            4. Create a new in-memory database

        :param table_name: The name of the table to store assistant runs.
        :param db_url: The database URL to connect to.
        :param db_file: The database file to connect to.
        :param db_engine: The database engine to use.
        :param encryptor: The encryptor to use. If not provided, a new FernetEncryptor will be created.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            _engine = create_engine(f"sqlite:///{db_file}")
        else:
            _engine = create_engine("sqlite://")

        if _engine is None:
            raise ValueError("Must provide either db_url, db_file or db_engine")

        # Database attributes
        self.table_name: str = table_name
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()

        # Database session
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.db_engine)

        # Database table for storage
        self.table: Table = self.get_table()

        # Initialize encryptor
        self.encryptor: EncryptorStrategy = encryptor or FernetEncryptor()

    def _sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """清理和预处理输入数据"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = value.replace(';', '').replace('--', '')
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_input(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_input(item) if isinstance(item, dict) else item 
                    for item in value
                ]
            elif isinstance(value, datetime):
                sanitized[key] = value.isoformat()
            else:
                sanitized[key] = value
        return sanitized

    def _encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """加密敏感数据"""
        encrypted_data = data.copy()
        sensitive_fields = ['llm', 'memory']

        try:
            for field in sensitive_fields:
                if field in encrypted_data and encrypted_data[field]:
                    # 确保 datetime 对象被正确序列化
                    if field == 'memory' and 'chat_history' in encrypted_data[field]:
                        chat_history = encrypted_data[field]['chat_history']
                        # logger.debug(f"Processing chat history for encryption, length: {len(chat_history)}")
                        for msg in chat_history:
                            if isinstance(msg.get('created_at'), datetime):
                                msg['created_at'] = msg['created_at'].isoformat()
                            if isinstance(msg.get('updated_at'), datetime):
                                msg['updated_at'] = msg['updated_at'].isoformat()
                    
                    # logger.debug(f"Encrypting field '{field}' with type: {type(encrypted_data[field])}")
                    encrypted_data[field] = self.encryptor.encrypt(json.dumps(encrypted_data[field]))
        except Exception as e:
            logger.error(f"Error encrypting '{field}': {e}")
            logger.error(f"Data that caused error: {encrypted_data[field]}")
        return encrypted_data

    def _decrypt_sensitive_data(self, row: Row[Any]) -> Dict[str, Any]:
        """解密敏感数据"""
        decrypted_data = dict(row._mapping)
        sensitive_fields = ['llm', 'memory']

        try:
            for field in sensitive_fields:
                if field in decrypted_data and decrypted_data[field]:
                    # logger.debug(f"Decrypting field '{field}'")
                    decrypted = json.loads(self.encryptor.decrypt(decrypted_data[field]))
                    
                    # 将时间戳字符串转换回 datetime 对象
                    if field == 'memory' and 'chat_history' in decrypted:
                        chat_history = decrypted['chat_history']
                        # logger.debug(f"Processing chat history for decryption, length: {len(chat_history)}")
                        for msg in chat_history:
                            if 'created_at' in msg:
                                msg['created_at'] = datetime.fromisoformat(msg['created_at'])
                            if 'updated_at' in msg:
                                msg['updated_at'] = datetime.fromisoformat(msg['updated_at'])
                    decrypted_data[field] = decrypted
        except Exception as e:
            logger.error(f"Error decrypting '{field}': {e}")
            logger.error(f"Data that caused error: {decrypted_data[field]}")
        return decrypted_data
    
    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            # Database ID/Primary key for this run
            Column("run_id", String, primary_key=True),
            # Assistant name
            Column("name", String),
            # Run name
            Column("run_name", String),
            # ID of the user participating in this run
            Column("user_id", String),
            # -*- LLM data (name, model, etc.)
            Column("llm", sqlite.JSON),
            # -*- Assistant memory
            Column("memory", sqlite.JSON),
            # Metadata associated with this assistant
            Column("assistant_data", sqlite.JSON),
            # Metadata associated with this run
            Column("run_data", sqlite.JSON),
            # Metadata associated the user participating in this run
            Column("user_data", sqlite.JSON),
            # Metadata associated with the assistant tasks
            Column("task_data", sqlite.JSON),
            # The timestamp of when this run was created.
            Column("created_at", sqlite.DATETIME, default=datetime.now()),
            # The timestamp of when this run was last updated.
            Column("updated_at", sqlite.DATETIME, onupdate=datetime.now()),
            extend_existing=True,
            sqlite_autoincrement=True,
        )

    def table_exists(self) -> bool:
        # logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name)
        except Exception as e:
            logger.error(e)
            return False

    def create(self) -> None:
        if not self.table_exists():
            logger.info(f"Creating table: {self.table.name}")
            self.table.create(self.db_engine)

    def _read(self, session: Session, run_id: str) -> Optional[Row[Any]]:
        stmt = select(self.table).where(self.table.c.run_id == run_id)
        try:
            return session.execute(stmt).first()
        except OperationalError:
            # Create table if it does not exist
            self.create()
        except Exception as e:
            logger.warning(e)
        return None

    def read(self, run_id: str) -> Optional[AssistantRun]:
        with self.Session() as sess:
            existing_row: Optional[Row[Any]] = self._read(session=sess, run_id=run_id)
            if existing_row is not None:
                decrypted_row = self._decrypt_sensitive_data(existing_row)
                return AssistantRun.model_validate(decrypted_row)
            return None

    def get_all_run_ids(
            self, 
            user_id: Optional[str] = None, 
            filter: Optional[Literal["created_at", "updated_at"]] = "created_at"
        ) -> List[str]:
        run_ids: List[str] = []
        try:
            with self.Session() as sess:
                # get all run_ids for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if filter == "created_at":
                    # order by created_at desc by default
                    stmt = stmt.order_by(self.table.c.created_at.desc())
                elif filter == "updated_at":
                    # order by updated_at desc
                    stmt = stmt.order_by(self.table.c.updated_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None and row.run_id is not None:
                        decrypted_row = self._decrypt_sensitive_data(row)
                        run_ids.append(decrypted_row['run_id'])
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return run_ids

    def get_all_runs(
            self, 
            user_id: Optional[str] = None, 
            filter: Optional[Literal["created_at", "updated_at"]] = "created_at",
            debug_mode: bool = False
        ) -> List[AssistantRun]:
        conversations: List[AssistantRun] = []
        try:
            # 在调试模式下，允许返回所有对话
            if debug_mode:
                logger.debug("Debug mode: returning all dialogs")
            # 在正常模式下，如果没有提供user_id，直接返回空列表
            elif user_id is None:
                logger.debug("No user_id provided, returning empty list")
                return []
            
            with self.Session() as sess:
                # get all runs for this user
                stmt = select(self.table)
                if not debug_mode and user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if filter == "created_at":
                    stmt = stmt.order_by(self.table.c.created_at.desc())
                elif filter == "updated_at":
                    stmt = stmt.order_by(self.table.c.updated_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row.run_id is not None:
                        decrypted_row = self._decrypt_sensitive_data(row)
                        conversations.append(AssistantRun.model_validate(decrypted_row))
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return conversations
    
    def get_specific_run(self, run_id: str, user_id: Optional[str] = None) -> Optional[AssistantRun]:
        try:
            # 如果没有提供user_id，直接返回None
            if user_id is None:
                logger.debug("No user_id provided, returning None")
                return None
            
            with self.Session() as sess:
                stmt = select(self.table).where(
                    self.table.c.run_id == run_id,
                    self.table.c.user_id == user_id
                )
                row = sess.execute(stmt).first()
                if row is not None:
                    decrypted_row = self._decrypt_sensitive_data(row)
                    return AssistantRun.model_validate(decrypted_row)
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        """
        Create a new assistant run if it does not exist, otherwise update the existing conversation.
        """
        start_time = time.time()
        with self.Session() as sess:
            try:
                logger.debug(f"Starting upsert for run_id: {row.run_id}")
                # Before upserting, encrypt sensitive data
                encrypted_data = self._encrypt_sensitive_data(
                    self._sanitize_input(row.model_dump())
                )
                
                if not encrypted_data:
                    logger.error("Failed to encrypt data")
                    return None
                
                # Create an insert statement
                stmt = sqlite.insert(self.table).values(
                    run_id=encrypted_data['run_id'],
                    name=encrypted_data['name'],
                    run_name=encrypted_data['run_name'],
                    user_id=encrypted_data['user_id'],
                    llm=encrypted_data['llm'],
                    memory=encrypted_data['memory'],
                    assistant_data=encrypted_data['assistant_data'],
                    run_data=encrypted_data['run_data'],
                    user_data=encrypted_data['user_data'],
                    task_data=encrypted_data['task_data'],
                    updated_at=datetime.now(),
                )

                # Define the upsert if the run_id already exists
                update_dict = {
                    'name': encrypted_data['name'],
                    'run_name': encrypted_data['run_name'],
                    'user_id': encrypted_data['user_id'],
                    'llm': encrypted_data['llm'],
                    'memory': encrypted_data['memory'],
                    'assistant_data': encrypted_data['assistant_data'],
                    'run_data': encrypted_data['run_data'],
                    'user_data': encrypted_data['user_data'],
                    'task_data': encrypted_data['task_data'],
                    'updated_at': datetime.now(),
                }

                # Filter out None values if necessary
                update_dict = {k: v for k, v in update_dict.items() if v is not None}

                stmt = stmt.on_conflict_do_update(
                    index_elements=["run_id"],
                    set_=update_dict,  # The updated value for each column
                )

                sess.execute(stmt)
                sess.commit()  # Make sure to commit the changes to the database
                logger.info(f"Successfully upserted run_id: {row.run_id} in {time.time() - start_time:.2f}s")
                return self.read(run_id=encrypted_data['run_id'])
            except Exception as e:
                logger.error(f"Error during upsert: {e}")
                sess.rollback()
                return None

    def delete_table(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)
    
    def delete_run(self, run_id: str, user_id: Optional[str] = None) -> None:
        if user_id is None:
            logger.debug("No user_id provided, skipping deletion")
            return
        with self.Session() as sess, sess.begin():
            stmt = delete(self.table).where(
                self.table.c.run_id == run_id,
                self.table.c.user_id == user_id
            )
            sess.execute(stmt)
            logger.info(f"Deleted assistant run: run_id = {run_id}")
