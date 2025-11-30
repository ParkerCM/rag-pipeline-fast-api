from typing import Any

from pydantic import BaseModel


class QueryResponse(BaseModel):
    response: str
    documents: list[dict[str, Any]]