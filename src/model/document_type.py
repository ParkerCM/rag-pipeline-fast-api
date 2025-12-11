from enum import Enum

class DocumentType(Enum):
    PDF = "pdf"
    TEXT = "txt"
    CSV = "csv"
    WORD = "docx"

    @staticmethod
    def get_file_extensions() -> list[str]:
        return [f'.{doc_type.value}' for doc_type in DocumentType]