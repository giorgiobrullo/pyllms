"""Utilities for converting Pydantic models to JSON schemas for structured outputs."""

from typing import Type, Dict, Any, get_origin, get_args, Union
from enum import Enum

def pydantic_to_json_schema(model: Type) -> Dict[str, Any]:
    """
    Convert a Pydantic model to a JSON schema suitable for OpenAI's structured outputs.

    Args:
        model: A Pydantic model class

    Returns:
        A JSON schema dictionary compatible with OpenAI's response_format
    """
    try:
        from pydantic import BaseModel

        if not issubclass(model, BaseModel):
            raise ValueError("Model must be a Pydantic BaseModel")

        # Get the JSON schema from Pydantic
        schema = model.model_json_schema()

        # OpenAI's format requires these fields at the top level
        formatted_schema = {
            "type": "json_schema",
            "name": model.__name__.lower(),
            "strict": True,
            "schema": schema
        }

        # Ensure additionalProperties is False for all objects in the schema
        _ensure_additional_properties_false(formatted_schema["schema"])

        return formatted_schema

    except ImportError:
        raise ImportError("Pydantic is required for model-based schemas. Install it with: pip install pydantic")

def _ensure_additional_properties_false(schema: Dict[str, Any]) -> None:
    """
    Recursively ensure all objects in the schema have additionalProperties: false.
    This is required for OpenAI's structured outputs.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        # Recursively process nested schemas
        for key, value in schema.items():
            if isinstance(value, dict):
                _ensure_additional_properties_false(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _ensure_additional_properties_false(item)

        # Process definitions
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                _ensure_additional_properties_false(def_schema)

def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic model class."""
    try:
        from pydantic import BaseModel
        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except ImportError:
        return False