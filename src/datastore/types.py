import re
from enum import Enum
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, AfterValidator
from typing_extensions import Annotated

JsonValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]

class DuckDBTypes(str, Enum):
    # Integer types
    TINYINT = "TINYINT"
    SMALLINT = "SMALLINT"
    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    HUGEINT = "HUGEINT"
    UTINYINT = "UTINYINT"
    USMALLINT = "USMALLINT"
    UINTEGER = "UINTEGER"
    UBIGINT = "UBIGINT"

    # Floating point types
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"

    # Decimal type
    DECIMAL = "DECIMAL"

    # String type
    VARCHAR = "VARCHAR"  # DuckDB treats CHAR and TEXT as VARCHAR

    # Boolean type
    BOOLEAN = "BOOLEAN"

    # Date/Time types
    DATE = "DATE"
    TIME = "TIME"
    TIMESTAMP = "TIMESTAMP"
    TIMESTAMPTZ = "TIMESTAMPTZ"

    # Other types
    UUID = "UUID"
    JSON = "JSON"

    # Special types
    OPENAI_EMBEDDING = "FLOAT[1536]"


column_name_regex = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _check_name(name: str) -> str:
    if not column_name_regex.match(name):
        raise ValueError(f"Invalid column name: {name}")
    return name


TableNameT = Annotated[str, AfterValidator(_check_name)]
ColumnNameT = Annotated[str, AfterValidator(_check_name)]

ParcelDataT = List[Dict[ColumnNameT, JsonValue]]


class ForeignKeyReferenceT(BaseModel):
    table_name: TableNameT
    column_name: ColumnNameT


class ColumnMetadataT(BaseModel):
    type: Optional[DuckDBTypes] = None
    description: Optional[str] = None
    foreign_key_references: Optional[ForeignKeyReferenceT] = None


class ParcelT(BaseModel):
    table_name: str
    hint: Optional[str] = None
    parcel_schema: Dict[ColumnNameT, ColumnMetadataT]
    rows: ParcelDataT
    readonly: bool = True


class SettingsT(BaseModel):
    backend: str = "local"  # "local" or "modal"
    database_path: Optional[str] = None
    modal_endpoint: Optional[str] = None
    timeout: int = 30


class AccessControlT(BaseModel):
    allowed_tables: Optional[List[TableNameT]] = None
    denied_tables: Optional[List[TableNameT]] = None
    row_limit: Optional[int] = None
    read_only: bool = True