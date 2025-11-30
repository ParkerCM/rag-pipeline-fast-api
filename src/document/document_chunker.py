from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentChunker:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents) -> list[Document]:
        """Chunk documents into smaller chunks."""

        print(f"Chunking {len(documents)} documents into smaller chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        split_documents = text_splitter.split_documents(documents)

        print(f"Chunked {len(documents)} documents into {len(split_documents)} chunks")
        return split_documents
