import uuid
from typing import Any

import chromadb
from langchain_core.documents import Document
from numpy import ndarray


class ChromaVectorStore:

    def __init__(self, collection_name: str = "rag_documents", persistence_directory_path: str = "vector", delete_existing_db: bool = False):
        self.collection_name = collection_name
        self.persistence_directory_path = persistence_directory_path
        self.delete_existing_db = delete_existing_db
        self.client = None
        self.collection = None
        self._initialize_chroma_store()

    def _initialize_chroma_store(self):
        try:
            self.client = chromadb.PersistentClient(path=self.persistence_directory_path)

            self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description": "RAG documents"})
            print(f"Initialized Chroma vector store in directory {self.persistence_directory_path} with collection {self.collection_name}")
            print(f"There are currently {self.collection.count()} documents in the collection")
        except Exception as e:
            print(f"Error initializing Chroma vector store: {e}")
            raise

    def delete_all_documents(self) -> int:
        before_document_count = self.collection.count()

        if before_document_count == 0:
            print("No documents to delete from the vector store")
            return 0

        results = self.collection.get()
        ids = results.get('ids', [])
        self.collection.delete(ids=ids)
        after_document_count = self.collection.count()

        if after_document_count > 0:
            raise ValueError("Failed to delete all documents from the vector store")

        return before_document_count

    def add_documents(self, documents: list[Document], embeddings: ndarray):
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must be equal")
        elif len(documents) == 0:
            print("No documents to add to the vector store")
            return

        print(f"Adding {len(documents)} documents to the vector store")

        ids = []
        metadatas = []
        document_texts = []
        embeddings_list = []

        for i, (document, embedding) in enumerate(zip(documents, embeddings)):
            id = f"doc_{uuid.uuid4()}_{i}"
            ids.append(id)

            metadata = dict(document.metadata)
            metadata["doc_index"] = str(i)
            metadata["content_length"] = str(len(document.page_content))
            metadatas.append(metadata)

            document_texts.append(document.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                documents=document_texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            print(f"Added {len(documents)} documents to the vector store")
            print(f"There are currently {self.collection.count()} documents in the collection")
        except Exception as e:
            print(f"Error adding documents to the vector store: {e}")
            raise

    def search_documents(self, query: ndarray, top_k: int = 5, score_threshold: float = 0.0) -> list[dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=query.tolist(),
                n_results=top_k
            )

            retrieved_documents = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                print(f"Retrieved {len(documents)} initial documents from search")

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance

                    print(f"Similarity score for document {doc_id}: {similarity_score}")

                    if similarity_score >= score_threshold:
                        retrieved_documents.append({
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_documents)} documents with similarity score >= {score_threshold}")
            else:
                print("No documents retrieved")

            return retrieved_documents
        except Exception as e:
            print(f"Error searching documents in the vector store: {e}")
            return []

    def get_existing_file_names(self) -> set[str]:
        try:
            results = self.collection.get(include=['metadatas'])
            metadatas = results['metadatas']
            file_names = set()

            for metadata in metadatas:
                file_names.add(metadata['file_name'])

            print(f"Found {len(file_names)} existing files in the vector store")

            return file_names
        except Exception as e:
            print(f"Error getting existing file names from the vector store: {e}")
            return set()