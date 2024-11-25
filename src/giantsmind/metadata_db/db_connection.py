import sqlite3
from contextlib import contextmanager
from typing import Generator, List, Optional, Type

from giantsmind.metadata_db.database_functions import DatabaseFunction
from giantsmind.metadata_db.models import DatabaseConfig, DatabaseConnection
from giantsmind.utils.logging import logger


class SQLiteConnection:
    """Basic SQLite database connection with context manager support."""

    def __init__(self, database: str | None = None, db_functions: List[DatabaseFunction] | None = None):
        self._conn: sqlite3.Connection | None = None
        self._database = database
        self._db_functions = db_functions

    def connect(self) -> sqlite3.Connection:
        self._conn = sqlite3.connect(self._database)
        self._setup_database_functions(self._db_functions)
        return self._conn

    def close(self) -> None:
        try:
            if self._conn:
                self._conn.close()
        except Exception as e:
            logger.error(f"Failed to close connection: {e}")
            raise
        finally:
            self._conn = None

    def _setup_database_functions(self, db_functions: List[DatabaseFunction]) -> None:
        """Setup database connection with required functions."""
        for func in db_functions:
            self._conn.create_function(func.name, func.num_params, func.func)


class DatabaseManager:
    """Singleton manager for database connections."""

    _instance: Optional["DatabaseManager"] = None

    def __init__(self, connection_cls: Type[DatabaseConnection], config: DatabaseConfig) -> None:
        self.connection_cls: Type[DatabaseConnection] = connection_cls
        self.config: DatabaseConfig = config
        self._connection: Optional[DatabaseConnection] = None
        self._reference_count: int = 0

    @classmethod
    def get_instance(
        cls, connection_cls: Type[DatabaseConnection], config: DatabaseConfig
    ) -> "DatabaseManager":
        if not cls._instance:
            cls._instance = cls(connection_cls, config)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance (primarily for testing)."""
        if cls._instance and cls._instance._connection:
            try:
                cls._instance._connection.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing connection during reset: {e}")
            finally:
                cls._instance._connection = None
                cls._instance._reference_count = 0
        cls._instance = None

    @contextmanager
    def get_connection(self) -> Generator[DatabaseConnection, None, None]:
        try:
            if self._connection:
                yield self._connection
            self._connection = self.connection_cls(self.config.path, self.config.db_functions)
            conn = self._connection.connect()
            self._reference_count += 1
            yield conn
        finally:
            self._reference_count -= 1
            if self._reference_count == 0:
                self._connection.close()
                self._connection = None
