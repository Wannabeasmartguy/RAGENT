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
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                # 移除危险字符
                sanitized[key] = value.replace(';', '').replace('--', '')
            else:
                sanitized[key] = value
        return sanitized

    def _encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        encrypted_data = data.copy()
        sensitive_fields = ['llm', 'memory']
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encryptor.encrypt(json.dumps(encrypted_data[field]))
        return encrypted_data

    def _decrypt_sensitive_data(self, row: Row[Any]) -> Dict[str, Any]:
        decrypted_data = dict(row._mapping)
        sensitive_fields = ['llm', 'memory']
        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = json.loads(self.encryptor.decrypt(decrypted_data[field]))
                except Exception as e:
                    logger.warning(f"Error decrypting {field}: {e}")
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
            filter: Optional[Literal["created_at", "updated_at"]] = "created_at"
        ) -> List[AssistantRun]:
        conversations: List[AssistantRun] = []
        try:
            with self.Session() as sess:
                # get all runs for this user
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
                    if row.run_id is not None:
                        decrypted_row = self._decrypt_sensitive_data(row)
                        conversations.append(AssistantRun.model_validate(decrypted_row))
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return conversations
    
    def get_specific_run(self, run_id: str, user_id: Optional[str] = None) -> AssistantRun:
        try:
            with self.Session() as sess:
                # get specific run for this user
                stmt = select(self.table).where(self.table.c.run_id == run_id)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                    
                # execute query
                row = sess.execute(stmt).first()
                if row is not None:
                    decrypted_row = self._decrypt_sensitive_data(row)
                    return AssistantRun.model_validate(decrypted_row)
            
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        """
        Create a new assistant run if it does not exist, otherwise update the existing conversation.
        """
        start_time = time.time()
        with self.Session() as sess:
            # Before upserting, encrypt sensitive data
            encrypted_data = self._encrypt_sensitive_data(
                self._sanitize_input(row.model_dump())
            )

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

            try:
                sess.execute(stmt)
                sess.commit()  # Make sure to commit the changes to the database
                logger.info(f"Upserted assistant run: run_id = {encrypted_data['run_id']}, name = {encrypted_data['run_name']} in {time.time() - start_time:.2f} seconds")
                return self.read(run_id=encrypted_data['run_id'])
            except OperationalError as oe:
                logger.debug(f"OperationalError occurred: {oe}")
                self.create()  # This will only create the table if it doesn't exist
                try:
                    sess.execute(stmt)
                    sess.commit()
                    return self.read(run_id=encrypted_data['run_id'])
                except Exception as e:
                    logger.warning(f"Error during upsert: {e}")
                    sess.rollback()  # Rollback the session in case of any error
                    return None
            except Exception as e:
                logger.warning(f"Error during upsert: {e}")
                sess.rollback()
                return None

    def delete_table(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)
    
    def delete_run(self, run_id: str) -> None:
        with self.Session() as sess, sess.begin():
            stmt = delete(self.table).where(self.table.c.run_id == run_id)
            sess.execute(stmt)
            logger.info(f"Deleted assistant run: run_id = {run_id}")
