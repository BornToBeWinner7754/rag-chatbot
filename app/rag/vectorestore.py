# app/rag/vectorestore.py
import faiss, numpy as np
from typing import List
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import os

FAISS_PATH = "faiss_store/index.faiss"

class FAISSVectorStore:
    def __init__(self, index, documents: List[Document]):
        self.index = index
        self.documents = documents

    def similarity_search(self, query_embedding, k=4):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return [self.documents[i] for i in indices[0]]

class HybridVectorStore(FAISSVectorStore):
    def __init__(self, index, documents: List[Document]):
        super().__init__(index, documents)
        # Build BM25 on the documents
        self.bm25 = BM25Okapi([doc.page_content.split() for doc in documents])

    def hybrid_search(self, query: str, k: int = 4, embeddings=None):
        # BM25 retrieval
        tokens = query.split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_idx = np.argsort(bm25_scores)[-k:].tolist()

        # FAISS retrieval
        faiss_idx = []
        if embeddings is not None:
            qv = embeddings.embed_query(query)
            qv = np.array([qv]).astype("float32")
            _, idxs = self.index.search(qv, k)
            faiss_idx = idxs[0].tolist()

        # Combine unique indices
        all_idx = list(set(top_bm25_idx + faiss_idx))
        return [self.documents[i] for i in all_idx]

def create_vectorstore(chunks: List[Document], embeddings):
    # Build FAISS index from scratch
    vectors = embeddings.embed_documents([doc.page_content for doc in chunks])
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))
    return FAISSVectorStore(index, chunks)

def save_index(index):
    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, FAISS_PATH)

def load_index(dimension):
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)
    return faiss.IndexFlatL2(dimension)