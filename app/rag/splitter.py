# app/rag/splitter.py

def split_docs(docs, chunk_size=500, overlap=50):
    chunks = []

    for doc in docs:
        text = doc.page_content
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(
                type(doc)(
                    page_content=chunk_text,
                    metadata=doc.metadata
                )
            )

            start = end - overlap

    return chunks
