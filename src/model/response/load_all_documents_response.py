from pydantic import BaseModel


class LoadAllDocumentsResponse(BaseModel):
    documents_added: int