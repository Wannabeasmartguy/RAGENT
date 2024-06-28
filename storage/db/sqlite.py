try:
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.engine import create_engine, Engine, Row
    from sqlalchemy.inspection import inspect
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.schema import MetaData, Table, Column
    from sqlalchemy.sql.expression import text, select, delete
    from sqlalchemy.types import DateTime, String
except ImportError:
    raise ImportError("`sqlalchemy` not installed")

from sqlite3 import OperationalError
from typing import Optional, List, Literal, Any
from loguru import logger

from model.memory.base import MemoryRow
from model.chat.assistant import AssistantRun
from utils.basic_utils import current_datetime


class SqliteMemoryDb:
    def __init__(
        self,
        table_name: str,
        schema: Optional[str] = None,
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
    ):
        """
        This class provides a memory store backed by a SQLite table.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url to create the engine

        Args:
            table_name (str): The name of the table to store memory rows.
            schema (Optional[str]): The schema to store the table in. Defaults to None.
            db_url (Optional[str]): The database URL to connect to. Defaults to None.
            db_engine (Optional[Engine]): The database engine to use. Defaults to None.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)

        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")

        self.table_name: str = table_name
        self.schema: Optional[str] = schema
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.db_engine)
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            Column("id", String, primary_key=True),
            Column("user_id", String),
            Column("memory", sqlite.JSON),
            Column("created_at", DateTime(timezone=True), server_default=text("CURRENT_TIMESTAMP")),
            Column("updated_at", DateTime(timezone=True), onupdate=text("CURRENT_TIMESTAMP")),
            extend_existing=True,
        )

    def create_table(self) -> None:
        if not self.table_exists():
            logger.debug(f"Creating table: {self.table_name}")
            self.table.create(self.db_engine)

    def memory_exists(self, memory: MemoryRow) -> bool:
        columns = [self.table.c.id]
        with self.Session() as sess:
            with sess.begin():
                stmt = select(*columns).where(self.table.c.id == memory.id)
                result = sess.execute(stmt).first()
                return result is not None

    def read_memories(
        self, user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None
    ) -> List[MemoryRow]:
        memories: List[MemoryRow] = []
        with self.Session() as sess, sess.begin():
            try:
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                if limit is not None:
                    stmt = stmt.limit(limit)

                if sort == "asc":
                    stmt = stmt.order_by(self.table.c.created_at.asc())
                else:
                    stmt = stmt.order_by(self.table.c.created_at.desc())

                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None:
                        memories.append(MemoryRow.model_validate(row))
            except Exception:
                # Create table if it does not exist
                self.create_table()
        return memories

    def upsert_memory(self, memory: MemoryRow) -> None:
        """Create a new memory if it does not exist, otherwise update the existing memory"""

        with self.Session() as sess, sess.begin():
            # Create an insert statement
            stmt = sqlite.insert(self.table).values(
                id=memory.id,
                user_id=memory.user_id,
                memory=memory.memory,
            )

            # Define the upsert if the memory already exists
            # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_=dict(
                    user_id=stmt.excluded.user_id,
                    memory=stmt.excluded.memory,
                ),
            )

            try:
                sess.execute(stmt)
            except Exception:
                # Create table and try again
                self.create_table()
                sess.execute(stmt)

    def delete_memory(self, id: str) -> None:
        with self.Session() as sess, sess.begin():
            stmt = delete(self.table).where(self.table.c.id == id)
            sess.execute(stmt)

    def delete_table(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name, schema=self.schema)
        except Exception as e:
            logger.error(e)
            return False

    def clear_table(self) -> bool:
        with self.Session() as sess:
            with sess.begin():
                stmt = delete(self.table)
                sess.execute(stmt)
                return True




class SqlAssistantStorage:
    def __init__(
        self,
        table_name: str,
        db_url: Optional[str] = None,
        db_file: Optional[str] = None,
        db_engine: Optional[Engine] = None,
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
            Column("created_at", sqlite.DATETIME, default=current_datetime()),
            # The timestamp of when this run was last updated.
            Column("updated_at", sqlite.DATETIME, onupdate=current_datetime()),
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
            return AssistantRun.model_validate(existing_row) if existing_row is not None else None

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
                        run_ids.append(row.run_id)
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
                        conversations.append(AssistantRun.model_validate(row))
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
                return AssistantRun.model_validate(row)
            
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return None

    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        """
        Create a new assistant run if it does not exist, otherwise update the existing conversation.
        """
        with self.Session() as sess:
            # Create an insert statement
            stmt = sqlite.insert(self.table).values(
                run_id=row.run_id,
                name=row.name,
                run_name=row.run_name,
                user_id=row.user_id,
                llm=row.llm,
                memory=row.memory,
                assistant_data=row.assistant_data,
                run_data=row.run_data,
                user_data=row.user_data,
                task_data=row.task_data,
                updated_at=current_datetime(),
            )

            # Define the upsert if the run_id already exists
            update_dict = {
                'name': row.name,
                'run_name': row.run_name,
                'user_id': row.user_id,
                'llm': row.llm,
                'memory': row.memory,
                'assistant_data': row.assistant_data,
                'run_data': row.run_data,
                'user_data': row.user_data,
                'task_data': row.task_data,
                'updated_at': current_datetime(),
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
                logger.info(f"Upserted assistant run: run_id = {row.run_id}, name = {row.run_name}")
                return self.read(run_id=row.run_id)
            except OperationalError as oe:
                logger.debug(f"OperationalError occurred: {oe}")
                self.create()  # This will only create the table if it doesn't exist
                try:
                    sess.execute(stmt)
                    sess.commit()
                    return self.read(run_id=row.run_id)
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