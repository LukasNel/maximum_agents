from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import duckdb
import os
import glob
from pathlib import Path
from .types import ParcelT, AccessControlT, SettingsT


class Backend(ABC):
    def __init__(self, settings: SettingsT, api_key: Optional[str] = None):
        self.settings = settings
        self.api_key = api_key
    
    @abstractmethod
    def load_parcel(self, database_id: str, parcel: ParcelT, overwrite: bool = False) -> None:
        pass
    
    @abstractmethod
    def execute_sql(self, database_id: str, sql_query: str, params: Optional[Dict[str, Any]] = None, access_control: Optional[AccessControlT] = None) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def database_exists(self, database_id: str) -> bool:
        pass
    
    @abstractmethod
    def create_database(self, database_id: str) -> None:
        pass
    
    @abstractmethod
    def list_databases(self) -> List[str]:
        pass
    
    @abstractmethod
    def list_tables(self, database_id: str) -> List[str]:
        pass
    
    @abstractmethod
    def table_exists(self, database_id: str, table_name: str) -> bool:
        pass
    
    @abstractmethod
    def get_table_schema(self, database_id: str, table_name: str) -> List[Dict[str, str]]:
        pass
    
    @abstractmethod
    def add_row(self, database_id: str, table_name: str, row_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> None:
        pass
    
    @abstractmethod
    def update_row_by_id(self, database_id: str, table_name: str, row_id: str, update_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> bool:
        pass
    
    @abstractmethod
    def overwrite_table(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        pass
    
    @abstractmethod
    def append_data(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        pass

    @abstractmethod
    def load_csv_with_schema_detection(self, database_id: str, csv_file_path: str, table_name: str, overwrite: bool = False) -> List[Dict[str, str]]:
        """Load CSV file and return schema information"""
        pass

    @abstractmethod 
    def load_dataframe_with_schema_detection(self, database_id: str, dataframe, table_name: str, overwrite: bool = False) -> List[Dict[str, str]]:
        """Load DataFrame and return schema information"""
        pass


class LocalBackend(Backend):
    def __init__(self, settings: SettingsT, api_key: Optional[str] = None):
        super().__init__(settings, api_key)
        self._connections: Dict[str, duckdb.DuckDBPyConnection] = {}
    
    def _get_db_path(self, database_id: str) -> str:
        if not self.api_key:
            raise ValueError("API key is required")
        
        base_path = self.settings.database_path or "databases"
        api_key_dir = os.path.join(base_path, self.api_key)
        os.makedirs(api_key_dir, exist_ok=True)
        return os.path.join(api_key_dir, f"{database_id}.duckdb")
    
    def _get_connection(self, database_id: str) -> duckdb.DuckDBPyConnection:
        if database_id not in self._connections:
            db_path = self._get_db_path(database_id)
            self._connections[database_id] = duckdb.connect(db_path)
        return self._connections[database_id]
    
    def database_exists(self, database_id: str) -> bool:
        try:
            db_path = self._get_db_path(database_id)
            return os.path.exists(db_path)
        except Exception:
            return False
    
    def create_database(self, database_id: str) -> None:
        self._get_connection(database_id)
    
    def table_exists(self, database_id: str, table_name: str) -> bool:
        try:
            conn = self._get_connection(database_id)
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name]
            ).fetchone()
            return result is not None and result[0] > 0
        except Exception:
            return False
    
    def get_table_schema(self, database_id: str, table_name: str) -> List[Dict[str, str]]:
        if not self.table_exists(database_id, table_name):
            return []
        
        conn = self._get_connection(database_id)
        result = conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
            [table_name]
        )
        return [{"column_name": row[0], "data_type": row[1]} for row in result.fetchall()]
    
    def add_row(self, database_id: str, table_name: str, row_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> None:
        if access_control and access_control.read_only:
            raise ValueError("Write operations not allowed with read-only access control")
        
        if not self.table_exists(database_id, table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        conn = self._get_connection(database_id)
        columns = list(row_data.keys())
        values = list(row_data.values())
        placeholders = ', '.join(['?' for _ in columns])
        columns_str = ', '.join(columns)
        
        insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
        conn.execute(insert_sql, values)
        conn.commit()
    
    def update_row_by_id(self, database_id: str, table_name: str, row_id: str, update_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> bool:
        if access_control and access_control.read_only:
            raise ValueError("Write operations not allowed with read-only access control")
        
        if not self.table_exists(database_id, table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        conn = self._get_connection(database_id)
        
        # Check if row exists
        check_result = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE id = ?",
            [row_id]
        ).fetchone()
        
        if not check_result or check_result[0] == 0:
            return False
        
        # Build UPDATE statement
        set_clauses = [f"{col} = ?" for col in update_data.keys()]
        set_clause = ', '.join(set_clauses)
        values = list(update_data.values()) + [row_id]
        
        update_query = f"UPDATE {table_name} SET {set_clause} WHERE id = ?"
        conn.execute(update_query, values)
        conn.commit()
        return True
    
    def overwrite_table(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        if access_control and access_control.read_only:
            raise ValueError("Write operations not allowed with read-only access control")
        
        if not data:
            raise ValueError("Cannot overwrite table with empty data")
        
        conn = self._get_connection(database_id)
        df = pd.DataFrame(data)
        
        # Drop existing table if it exists
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create new table from DataFrame
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        conn.commit()
    
    def append_data(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        if access_control and access_control.read_only:
            raise ValueError("Write operations not allowed with read-only access control")
        
        if not data:
            return  # Nothing to append
        
        if not self.table_exists(database_id, table_name):
            raise ValueError(f"Table {table_name} does not exist")
        
        conn = self._get_connection(database_id)
        df = pd.DataFrame(data)
        
        # Append data using DuckDB's efficient method
        conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        conn.commit()
    
    def load_csv_with_schema_detection(self, database_id: str, csv_file_path: str, table_name: str, overwrite: bool = False) -> List[Dict[str, str]]:
        """Load CSV file and return schema information"""
        conn = self._get_connection(database_id)
        
        # Check if table exists
        table_exists = self.table_exists(database_id, table_name)
        
        if table_exists and not overwrite:
            raise ValueError(f"Table {table_name} already exists. Use overwrite=True to replace it.")
        
        if table_exists and overwrite:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Use DuckDB's CSV auto-detection to create table
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{csv_file_path}')")
        conn.commit()
        
        # Return the schema
        return self.get_table_schema(database_id, table_name)
    
    def load_dataframe_with_schema_detection(self, database_id, dataframe, table_name: str, overwrite: bool = False) -> List[Dict[str, str]]:
        """Load DataFrame and return schema information"""
        conn = self._get_connection(database_id)
        
        # Check if table exists
        table_exists = self.table_exists(database_id, table_name)
        
        if table_exists and not overwrite:
            raise ValueError(f"Table {table_name} already exists. Use overwrite=True to replace it.")
        
        if table_exists and overwrite:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Create table from DataFrame using DuckDB's efficient method
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM dataframe")
        conn.commit()
        
        # Return the schema
        return self.get_table_schema(database_id, table_name)
    
    def load_parcel(self, database_id: str, parcel: ParcelT, overwrite: bool = False) -> None:
        conn = self._get_connection(database_id)
        
        # Check if table exists
        table_exists = self.table_exists(database_id, parcel.table_name)
        
        if table_exists and not overwrite:
            raise ValueError(f"Table {parcel.table_name} already exists. Use overwrite=True to replace it.")
        
        if table_exists and overwrite:
            conn.execute(f"DROP TABLE IF EXISTS {parcel.table_name}")
        
        # Create table schema
        schema_parts = []
        for col_name, col_meta in parcel.parcel_schema.items():
            col_type = col_meta.type.value if col_meta.type else "VARCHAR"
            schema_parts.append(f"{col_name} {col_type}")
        
        schema_sql = f"CREATE TABLE {parcel.table_name} ({', '.join(schema_parts)})"
        conn.execute(schema_sql)
        
        # Insert data
        if parcel.rows:
            columns = list(parcel.parcel_schema.keys())
            placeholders = ', '.join(['?' for _ in columns])
            insert_sql = f"INSERT INTO {parcel.table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            for row in parcel.rows:
                values = [row.get(col) for col in columns]
                conn.execute(insert_sql, values)
        
        conn.commit()
    
    def execute_sql(self, database_id: str, sql_query: str, params: Optional[Dict[str, Any]] = None, access_control: Optional[AccessControlT] = None) -> pd.DataFrame:
        conn = self._get_connection(database_id)
        
        # Apply access control if provided
        if access_control:
            if access_control.read_only and any(keyword in sql_query.upper() for keyword in ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER']):
                raise ValueError("Write operations not allowed with read-only access control")
            
            if access_control.denied_tables:
                for table in access_control.denied_tables:
                    if table in sql_query:
                        raise ValueError(f"Access denied to table: {table}")
        
        # Execute query
        if params:
            result = conn.execute(sql_query, list(params.values()))
        else:
            result = conn.execute(sql_query)
        
        df = result.df()
        
        # Apply row limit if specified
        if access_control and access_control.row_limit:
            df = df.head(access_control.row_limit)
        
        return df
    
    def list_databases(self) -> List[str]:
        if not self.api_key:
            raise ValueError("API key is required")
        
        base_path = self.settings.database_path or "databases"
        api_key_dir = os.path.join(base_path, self.api_key)
        
        if not os.path.exists(api_key_dir):
            return []
        
        db_files = glob.glob(os.path.join(api_key_dir, "*.duckdb"))
        return [Path(f).stem for f in db_files]
    
    def list_tables(self, database_id: str) -> List[str]:
        if not self.database_exists(database_id):
            return []
        
        conn = self._get_connection(database_id)
        result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'")
        return [row[0] for row in result.fetchall()]


class ModalBackend(Backend):
    def __init__(self, settings: SettingsT, api_key: Optional[str] = None):
        super().__init__(settings, api_key)
        if not self.settings.modal_endpoint:
            raise ValueError("Modal endpoint required for ModalBackend")
        if not self.api_key:
            raise ValueError("API key required for ModalBackend")
    
    def database_exists(self, database_id: str) -> bool:
        # TODO: Implement modal backend database existence check
        raise NotImplementedError("Modal backend not yet implemented")
    
    def create_database(self, database_id: str) -> None:
        # TODO: Implement modal backend database creation
        raise NotImplementedError("Modal backend not yet implemented")
    
    def table_exists(self, database_id: str, table_name: str) -> bool:
        # TODO: Implement modal backend table existence check
        raise NotImplementedError("Modal backend not yet implemented")
    
    def get_table_schema(self, database_id: str, table_name: str) -> List[Dict[str, str]]:
        # TODO: Implement modal backend table schema retrieval
        raise NotImplementedError("Modal backend not yet implemented")
    
    def add_row(self, database_id: str, table_name: str, row_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> None:
        # TODO: Implement modal backend row addition
        raise NotImplementedError("Modal backend not yet implemented")
    
    def update_row_by_id(self, database_id: str, table_name: str, row_id: str, update_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> bool:
        # TODO: Implement modal backend row update
        raise NotImplementedError("Modal backend not yet implemented")
    
    def overwrite_table(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        # TODO: Implement modal backend table overwrite
        raise NotImplementedError("Modal backend not yet implemented")
    
    def append_data(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        # TODO: Implement modal backend data append
        raise NotImplementedError("Modal backend not yet implemented")
    
    def load_csv_with_schema_detection(self, database_id: str, csv_file_path: str, table_name: str, overwrite: bool = False) -> List[Dict[str, str]]:
        # TODO: Implement modal backend CSV loading
        raise NotImplementedError("Modal backend not yet implemented")

    def load_dataframe_with_schema_detection(self, database_id, dataframe, table_name: str, overwrite: bool = False) -> List[Dict[str, str]]:
        # TODO: Implement modal backend DataFrame loading
        raise NotImplementedError("Modal backend not yet implemented")
    
    def load_parcel(self, database_id: str, parcel: ParcelT, overwrite: bool = False) -> None:
        # TODO: Implement modal backend parcel loading
        raise NotImplementedError("Modal backend not yet implemented")
    
    def execute_sql(self, database_id: str, sql_query: str, params: Optional[Dict[str, Any]] = None, access_control: Optional[AccessControlT] = None) -> pd.DataFrame:
        # TODO: Implement modal backend SQL execution
        raise NotImplementedError("Modal backend not yet implemented")
    
    def list_databases(self) -> List[str]:
        # TODO: Implement modal backend database listing
        raise NotImplementedError("Modal backend not yet implemented")
    
    def list_tables(self, database_id: str) -> List[str]:
        # TODO: Implement modal backend table listing
        raise NotImplementedError("Modal backend not yet implemented")


def get_backend(settings: SettingsT, api_key: Optional[str] = None) -> Backend:
    if settings.backend == "local":
        return LocalBackend(settings, api_key)
    elif settings.backend == "modal":
        return ModalBackend(settings, api_key)
    else:
        raise ValueError(f"Unknown backend: {settings.backend}")