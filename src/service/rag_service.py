from src.db.chroma_vector_store import ChromaVectorStore
from src.document.document_chunker import DocumentChunker
from src.document.document_loader import DocumentLoader
from src.embedding.embedding_manager import EmbeddingManager
from src.llm.groq_llm import GroqLLM
from src.model.response.delete_all_documents_response import DeleteAllDocumentsResponse
from src.model.response.load_all_documents_response import LoadAllDocumentsResponse
from src.model.response.query_response import QueryResponse


class RAGService:

    def __init__(self):
        self.vector_store = ChromaVectorStore()
        self.document_loader = DocumentLoader()
        self.document_chunker = DocumentChunker()
        self.embedding_manager = EmbeddingManager()
        self.llm = GroqLLM()

    def load_all_documents(self) -> LoadAllDocumentsResponse:
        documents = self.document_loader.load_documents("data", self.vector_store.get_existing_file_names())
        chunked_documents = self.document_chunker.chunk_documents(documents)
        embeddings = self.embedding_manager.generate_embeddings([doc.page_content for doc in chunked_documents])
        self.vector_store.add_documents(chunked_documents, embeddings)

        return LoadAllDocumentsResponse(documents_added=len(chunked_documents))

    def delete_all_documents(self) -> DeleteAllDocumentsResponse:
        deleted_document_count = self.vector_store.delete_all_documents()
        return DeleteAllDocumentsResponse(documents_deleted=deleted_document_count)

    def process_query(self, query: str) -> QueryResponse:
        embedded_query = self.embedding_manager.generate_embeddings([query])
        results = self.vector_store.search_documents(embedded_query)

        context = "\n\n".join([result["document"] for result in results])
        query_response = self.llm.generate_response(query, context)

        return QueryResponse(response=query_response, documents=results)
