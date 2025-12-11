import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, CSVLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

from src.model.document_type import DocumentType


class DocumentLoader:

    def load_documents(self, directory: str, existing_file_names: set[str]) -> list[Document]:
        """Load documents from a list of directories.

        Supported file types: PDF, TXT, CSV."""

        print(f"Recursively loading documents from {directory}")

        found_documents = self._get_documents_recursively(directory)

        print(f"Documents found: {found_documents}")

        documents = self._load_all_documents(found_documents, existing_file_names)

        print(f"Loaded {len(documents)} documents from {directory}")
        return documents

    def _load_all_documents(self, documents: list[Path], existing_file_names: set[str]) -> list[Document]:
        all_loaded_documents = []

        for document in documents:
            if document.name in existing_file_names:
                print(f"Skipping existing file {document.name}")
                continue
            elif document.name.endswith(".pdf"):
                doc_type = DocumentType.PDF
                loader = PyMuPDFLoader(str(document))
            elif document.name.endswith(".txt"):
                doc_type = DocumentType.TEXT
                loader = TextLoader(str(document))
            elif document.name.endswith(".csv"):
                doc_type = DocumentType.CSV
                loader = CSVLoader(str(document))
            elif document.name.endswith(".docx"):
                doc_type = DocumentType.WORD
                loader = Docx2txtLoader(str(document))
            else:
                continue

            loaded_documents = loader.load()

            for loaded_document in loaded_documents:
                loaded_document.metadata["file_name"] = document.name
                loaded_document.metadata["file_type"] = doc_type.value

            all_loaded_documents.extend(loaded_documents)

        if all_loaded_documents:
            print(f"Successfully loaded {len(all_loaded_documents)} documents")
            return all_loaded_documents

        print("No documents loaded")
        return []

    def _get_documents_recursively(self, directory: str) -> list[Path]:
        path = Path(directory)
        files: list[Path] = []

        for entry in path.iterdir():
            if entry.is_dir():
                files.extend(self._get_documents_recursively(str(entry)))
            elif entry.is_file() and entry.suffix.lower() in DocumentType.get_file_extensions():
                files.append(entry)

        return files
