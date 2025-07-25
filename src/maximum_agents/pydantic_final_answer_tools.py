from typing import Dict, Any, Union
import typing
from smolagents import FinalAnswerTool
from typing import Type, get_type_hints, get_origin, get_args
import inspect
from pydantic import BaseModel
from typing import Literal
from typing_extensions import NotRequired, TypedDict, Required


class JSONSchema(TypedDict, total=False):
    type: Required[str]
    description: NotRequired[str]
    nullable: NotRequired[bool]
    enum: NotRequired[list[Any]]
    items: NotRequired["JSONSchema"]
    additionalProperties: NotRequired["JSONSchema"]
    anyOf: NotRequired[list[str]]
    properties: NotRequired[Dict[str, "JSONSchema"]]
    required: NotRequired[list[str]]


def pydantic_to_schema(
    model_class: Type[BaseModel], description: str | None = None
) -> JSONSchema:
    """
    Convert a Pydantic model into a JSON schema format compatible with the Tool class.

    Args:
        model_class: A Pydantic model class (subclass of BaseModel)
        description: Optional description for the entire schema

    Returns:
        A dictionary representing the JSON schema
    """
    # if not inspect.isclass(model_class) or not issubclass(model_class, BaseModel):
    #     raise TypeError("Input must be a Pydantic model class (subclass of BaseModel)")

    # Get model schema from Pydantic
    schema = model_class.model_json_schema()

    # Get field descriptions from docstrings if available
    field_descriptions = {}
    for field_name, field in model_class.model_fields.items():
        if field.description:
            field_descriptions[field_name] = field.description

    # Create the base schema
    result_schema: JSONSchema = {"type": "object", "properties": {}}

    if description:
        result_schema["description"] = description
    elif "description" in schema:
        result_schema["description"] = schema["description"]
    else:
        result_schema["description"] = f"Schema for {model_class.__name__}"

    # Process each field
    properties = {}
    for field_name, field in model_class.model_fields.items():
        field_schema = process_field(field_name, field, model_class, field_descriptions)
        if field_schema:
            properties[field_name] = field_schema

    result_schema["properties"] = properties

    # Include required fields if any
    required_fields = [
        name for name, field in model_class.model_fields.items() if field.is_required()
    ]
    if required_fields:
        result_schema["required"] = required_fields
    return result_schema


def process_field(
    field_name: str,
    field,
    model_class: Type[BaseModel],
    field_descriptions: Dict[str, str],
) -> JSONSchema:
    """
    Process a single field and convert it to the appropriate schema format.

    Args:
        field_name: Name of the field
        field: Pydantic field object
        model_class: The parent model class
        field_descriptions: Dictionary of field descriptions

    Returns:
        A dictionary representing the field schema
    """
    # Get type hints to handle complex types
    type_hints = get_type_hints(model_class)
    field_type = type_hints.get(field_name)

    # Start with basic schema
    field_schema: JSONSchema = {"type": "string"}  # Default type

    # Add description if available
    if field_name in field_descriptions:
        field_schema["description"] = field_descriptions[field_name]
    elif field.description:
        field_schema["description"] = field.description
    else:
        field_schema["description"] = f"The {field_name} field"

    # Handle different types
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Union types (Optional is Union[T, None])
    if origin is Union:
        if type(None) in args:
            field_schema["nullable"] = True
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            # This is an Optional[T] field
            field_schema = process_type(non_none_types[0], field_schema)
        else:
            # Complex Union type - use anyOf
            field_schema["anyOf"] = [
                process_type(arg, {"type": "string"})["type"] for arg in non_none_types
            ]

    # Handle List types
    elif origin is list:
        field_schema["type"] = "array"
        if args:
            item_type = args[0]
            if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                field_schema["items"] = pydantic_to_schema(item_type)
            else:
                field_schema["items"] = process_type(item_type, {"type": "string"})

    # Handle Dict types
    elif origin is dict:
        field_schema["type"] = "object"
        if len(args) >= 2:
            # For Dict[str, Something], we can provide additionalProperties
            value_type = args[1]
            if inspect.isclass(value_type) and issubclass(value_type, BaseModel):
                field_schema["additionalProperties"] = pydantic_to_schema(value_type)
            else:
                field_schema["additionalProperties"] = process_type(
                    value_type, {"type": "string"}
                )

    # Handle nested models
    elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        nested_schema = pydantic_to_schema(field_type)
        field_schema.update(nested_schema)
    # Handle basic types
    elif isinstance(field_type, type):
        field_schema = process_type(field_type, field_schema)

    # Handle Literal types
    elif origin is Literal:
        args = get_args(field_type)
        field_schema["enum"] = list(args)
        # Determine the type from the first value
        if args:
            first_arg_type = type(args[0])
            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
            }
            if first_arg_type in type_mapping:
                field_schema["type"] = type_mapping[first_arg_type]
    else:
        raise ValueError(f"Unsupported field type: {field_type} for field {field_name}")

    return field_schema


def process_type(python_type: Type[Any], schema: JSONSchema) -> JSONSchema:
    """
    Convert Python type to JSON schema type.

    Args:
        python_type: The Python type
        schema: Existing schema to update

    Returns:
        Updated schema dictionary
    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        None: "null",
    }

    # Handle Literal types
    origin = get_origin(python_type)
    if origin is Literal:
        args = get_args(python_type)
        schema["enum"] = list(args)
        # Determine the type from the first value
        if args:
            first_arg_type = type(args[0])
            if first_arg_type in type_mapping:
                schema["type"] = type_mapping[first_arg_type]
        return schema
    # Map the type if it's in our mapping
    if python_type in type_mapping:
        schema["type"] = type_mapping[python_type]
        # convert typing.List[<random_type>] to array of <random_type>
    elif typing.get_origin(python_type) is list:
        schema["type"] = "array"
        args = typing.get_args(python_type)
        schema["items"] = process_type(args[0], {"type": "string"})
    else:
        # Default to "string" for unknown types
        schema["type"] = "string"

    return schema


class PydanticFinalAnswerTool(FinalAnswerTool):
    def __init__(
        self,
        model: Type[BaseModel],
        description: str = "A user object",
        context: dict[str, Any] = {},
        *args,
        **kwargs,
    ):
        self.inputs: Dict[str, Any] = {"answer": pydantic_to_schema(model, description)}
        self.model_pydantic: Type[BaseModel] = model
        self.context = context
        super().__init__(*args, **kwargs)

    def forward(self, answer: dict[str, Any]) -> dict[str, Any]:
        data = self.model_pydantic.model_validate(answer, context=self.context)

        return data.model_dump()
