from pydantic import BaseModel


class DeleteAllDocumentsResponse(BaseModel):
    documents_deleted: int