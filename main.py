import uvicorn
from fastapi import FastAPI

from src.model.response.delete_all_documents_response import DeleteAllDocumentsResponse
from src.model.response.load_all_documents_response import LoadAllDocumentsResponse
from src.model.response.query_response import QueryResponse
from src.service.rag_service import RAGService

app = FastAPI()
service = RAGService()


@app.get("/query")
def query(q: str) -> QueryResponse:
    return service.process_query(q)

@app.get("/reload-documents")
def reload_documents(force: bool = False) -> LoadAllDocumentsResponse:
    return service.load_all_documents(force)

@app.delete("/delete-all-documents")
def delete_all_documents() -> DeleteAllDocumentsResponse:
   return service.delete_all_documents()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)