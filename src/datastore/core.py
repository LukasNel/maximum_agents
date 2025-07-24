from typing import Optional, Dict, Any, Union, List
import pandas as pd
import os
from pathlib import Path
from .types import SettingsT, ParcelT, AccessControlT, ColumnMetadataT, DuckDBTypes
from .backends import get_backend, Backend


class MaximumDataStore:
    def __init__(self, settings: SettingsT, api_key: Optional[str] = None):
        self.settings = settings
        self.api_key = api_key
        self.backend: Backend = get_backend(settings, api_key)
    
    def load_into_database(self, database_id: str, parcel: ParcelT, overwrite: bool = False) -> None:
        """
        Load a parcel of data into the specified database.
        
        Args:
            database_id: Unique identifier for the database
            parcel: The data parcel to load
            overwrite: Whether to overwrite existing table if it exists
        
        Raises:
            ValueError: If table exists and overwrite is False
        """
        if not self.backend.database_exists(database_id):
            self.backend.create_database(database_id)
        
        self.backend.load_parcel(database_id, parcel, overwrite)
    
    def sql_engine(
        self, 
        database_id: str, 
        sql_query: str, 
        optional_params: Optional[Dict[str, Any]] = None,
        access_control: Optional[AccessControlT] = None
    ) -> pd.DataFrame:
        """
        Execute SQL query on the specified database.
        
        Args:
            database_id: Unique identifier for the database
            sql_query: SQL query to execute
            optional_params: Optional parameters for the query
            access_control: Access control settings
        
        Returns:
            pandas.DataFrame: Query results
        
        Raises:
            ValueError: If database doesn't exist or access control violations
        """
        if not self.backend.database_exists(database_id):
            raise ValueError(f"Database {database_id} does not exist")
        
        return self.backend.execute_sql(database_id, sql_query, optional_params, access_control)
    
    def table_exists(self, database_id: str, table_name: str) -> bool:
        """
        Check if a table exists in the specified database.
        
        Args:
            database_id: Unique identifier for the database
            table_name: Name of the table to check
        
        Returns:
            bool: True if table exists, False otherwise
        """
        if not self.backend.database_exists(database_id):
            return False
        return self.backend.table_exists(database_id, table_name)
    
    def get_table_schema(self, database_id: str, table_name: str) -> List[Dict[str, str]]:
        """
        Get the schema of a table.
        
        Args:
            database_id: Unique identifier for the database
            table_name: Name of the table
        
        Returns:
            List[Dict[str, str]]: List of column information with keys 'column_name' and 'data_type'
        
        Raises:
            ValueError: If database doesn't exist
        """
        if not self.backend.database_exists(database_id):
            raise ValueError(f"Database {database_id} does not exist")
        
        return self.backend.get_table_schema(database_id, table_name)
    
    def add_row(self, database_id: str, table_name: str, row_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> None:
        """
        Add a single row to a table.
        
        Args:
            database_id: Unique identifier for the database
            table_name: Name of the table
            row_data: Dictionary of column names to values
            access_control: Access control settings
        
        Raises:
            ValueError: If database doesn't exist, table doesn't exist, or access control violations
        """
        if not self.backend.database_exists(database_id):
            raise ValueError(f"Database {database_id} does not exist")
        
        self.backend.add_row(database_id, table_name, row_data, access_control)
    
    def update_row_by_id(self, database_id: str, table_name: str, row_id: str, update_data: Dict[str, Any], access_control: Optional[AccessControlT] = None) -> bool:
        """
        Update a row in a table by its ID.
        
        Args:
            database_id: Unique identifier for the database
            table_name: Name of the table
            row_id: ID of the row to update
            update_data: Dictionary of column names to new values
            access_control: Access control settings
        
        Returns:
            bool: True if row was updated, False if row was not found
        
        Raises:
            ValueError: If database doesn't exist, table doesn't exist, or access control violations
        """
        if not self.backend.database_exists(database_id):
            raise ValueError(f"Database {database_id} does not exist")
        
        return self.backend.update_row_by_id(database_id, table_name, row_id, update_data, access_control)
    
    def overwrite_table(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        """
        Completely replace a table's contents with new data.
        
        Args:
            database_id: Unique identifier for the database
            table_name: Name of the table
            data: List of dictionaries representing rows
            access_control: Access control settings
        
        Raises:
            ValueError: If database doesn't exist, empty data, or access control violations
        """
        if not self.backend.database_exists(database_id):
            self.backend.create_database(database_id)
        
        self.backend.overwrite_table(database_id, table_name, data, access_control)
    
    def append_data(self, database_id: str, table_name: str, data: List[Dict[str, Any]], access_control: Optional[AccessControlT] = None) -> None:
        """
        Append data to an existing table.
        
        Args:
            database_id: Unique identifier for the database
            table_name: Name of the table
            data: List of dictionaries representing rows to append
            access_control: Access control settings
        
        Raises:
            ValueError: If database doesn't exist, table doesn't exist, or access control violations
        """
        if not self.backend.database_exists(database_id):
            raise ValueError(f"Database {database_id} does not exist")
        
        self.backend.append_data(database_id, table_name, data, access_control)

    def load_csv_into_database(
        self, 
        database_id: str, 
        csv_file_path: str, 
        table_name: str,
        overwrite: bool = False,
        hint: Optional[str] = None
    ) -> ParcelT:
        """
        Load a CSV file into the database with automatic schema detection.
        
        Args:
            database_id: Unique identifier for the database
            csv_file_path: Path to the CSV file
            table_name: Name for the database table
            overwrite: Whether to overwrite existing table if it exists
            hint: Optional hint about the table contents
            
        Returns:
            ParcelT: The created parcel with detected schema
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If table exists and overwrite is False
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Create database if it doesn't exist
        if not self.backend.database_exists(database_id):
            self.backend.create_database(database_id)
        
        # Load CSV using backend method
        schema_result = self.backend.load_csv_with_schema_detection(database_id, csv_file_path, table_name, overwrite)
        
        # Convert DuckDB types to our enum types
        type_mapping = {
            'INTEGER': DuckDBTypes.INTEGER,
            'BIGINT': DuckDBTypes.BIGINT,
            'DOUBLE': DuckDBTypes.DOUBLE,
            'VARCHAR': DuckDBTypes.VARCHAR,
            'BOOLEAN': DuckDBTypes.BOOLEAN,
            'DATE': DuckDBTypes.DATE,
            'TIMESTAMP': DuckDBTypes.TIMESTAMP,
            'FLOAT': DuckDBTypes.FLOAT,
            'DECIMAL': DuckDBTypes.DECIMAL,
        }
        
        # Build schema metadata
        parcel_schema = {}
        for schema_info in schema_result:
            col_name = schema_info["column_name"]
            col_type = schema_info["data_type"]
            mapped_type = type_mapping.get(col_type, DuckDBTypes.VARCHAR)
            parcel_schema[col_name] = ColumnMetadataT(
                type=mapped_type,
                description=f"Auto-detected {col_type} column from CSV"
            )
        
        # Get sample data for the parcel
        sample_df = self.sql_engine(database_id, f"SELECT * FROM {table_name} LIMIT 100")
        sample_rows = sample_df.to_dict('records')
        
        # Create and return the parcel
        parcel = ParcelT(
            table_name=table_name,
            hint=hint or f"Auto-loaded from CSV file: {Path(csv_file_path).name}",
            parcel_schema=parcel_schema,
            rows=sample_rows,
            readonly=False  # CSV uploads are typically meant to be editable
        )
        
        return parcel

    def load_dataframe_into_database(
        self,
        database_id: str,
        dataframe: pd.DataFrame,
        table_name: str,
        overwrite: bool = False,
        hint: Optional[str] = None
    ) -> ParcelT:
        """
        Load a pandas DataFrame into the database with automatic schema detection.
        
        Args:
            database_id: Unique identifier for the database
            dataframe: The pandas DataFrame to load
            table_name: Name for the database table
            overwrite: Whether to overwrite existing table if it exists
            hint: Optional hint about the table contents
            
        Returns:
            ParcelT: The created parcel with detected schema
        """
        if dataframe.empty:
            raise ValueError("Cannot load empty DataFrame")
        
        # Create database if it doesn't exist
        if not self.backend.database_exists(database_id):
            self.backend.create_database(database_id)
        
        # Load DataFrame using backend method
        schema_result = self.backend.load_dataframe_with_schema_detection(database_id, dataframe, table_name, overwrite)
        
        # Convert DuckDB types to our enum types
        type_mapping = {
            'INTEGER': DuckDBTypes.INTEGER,
            'BIGINT': DuckDBTypes.BIGINT,
            'DOUBLE': DuckDBTypes.DOUBLE,
            'VARCHAR': DuckDBTypes.VARCHAR,
            'BOOLEAN': DuckDBTypes.BOOLEAN,
            'DATE': DuckDBTypes.DATE,
            'TIMESTAMP': DuckDBTypes.TIMESTAMP,
            'FLOAT': DuckDBTypes.FLOAT,
            'DECIMAL': DuckDBTypes.DECIMAL,
        }
        
        # Build schema metadata
        parcel_schema = {}
        for schema_info in schema_result:
            col_name = schema_info["column_name"]
            col_type = schema_info["data_type"]
            mapped_type = type_mapping.get(col_type, DuckDBTypes.VARCHAR)
            parcel_schema[col_name] = ColumnMetadataT(
                type=mapped_type,
                description=f"Auto-detected {col_type} column from DataFrame"
            )
        
        # Get sample data for the parcel
        sample_rows = dataframe.head(100).to_dict('records')
        
        # Create and return the parcel
        parcel = ParcelT(
            table_name=table_name,
            hint=hint or f"Auto-loaded DataFrame with {len(dataframe)} rows",
            parcel_schema=parcel_schema,
            rows=sample_rows,
            readonly=False
        )
        
        return parcel