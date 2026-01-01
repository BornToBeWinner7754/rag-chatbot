# app/rag/loader.py
from pypdf import PdfReader
from langchain_core.documents import Document


def load_docs(path: str):
    reader = PdfReader(path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": page_num + 1}
                )
            )

    return documents
