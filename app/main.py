# app/main.py
import os, numpy as np
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

from app.rag.loader import load_docs, Document
from app.rag.splitter import split_docs
from app.rag.embeddings import get_embeddings
from app.rag.vectorestore import create_vectorstore, HybridVectorStore
from app.rag.chain import create_rag_chain
from app.rag.llm import get_llm
from sentence_transformers import CrossEncoder

app = FastAPI(title="RAG Chatbot API (Production-ready)")

# --- Startup: Load and index documents ---
docs = load_docs("data/data_rag.pdf")
chunks = split_docs(docs)
embeddings = get_embeddings()

# Build FAISS index on embeddings
vectorstore = create_vectorstore(chunks, embeddings)

# Wrap in HybridVectorStore for BM25+FAISS search
hybrid_store = HybridVectorStore(vectorstore.index, chunks)

# Prepare LLM and reranker
llm = get_llm()  # Groq LLM
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Create RAG pipeline with all components
rag_chain = create_rag_chain(hybrid_store, llm, embeddings, cross_encoder)

# --- API Models ---
class ChatRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    documents: List[str]

# Utility: Generator to stream answer in chunks
async def chunked_response(answer: str, chunk_size: int = 100):
    for i in range(0, len(answer), chunk_size):
        yield answer[i:i+chunk_size]

# --- /chat endpoint (streaming) ---
@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.query
    # Generate answer (synchronous call for now)
    answer = rag_chain(query)
    # Stream back the answer in chunks
    return StreamingResponse(chunked_response(answer), media_type="text/event-stream")

# --- /ingest endpoint: multi-doc ingestion ---
@app.post("/ingest")
async def ingest(request: IngestRequest):
    # Receive new documents as raw text
    new_docs = []
    for text in request.documents:
        text = text.strip()
        if text:
            new_docs.append(Document(page_content=text, metadata={"source": "api"}))
    if not new_docs:
        raise HTTPException(status_code=400, detail="No documents provided")
    # Split new docs into chunks
    new_chunks = []
    for doc in new_docs:
        new_chunks.extend(split_docs([doc]))
    # Embed new chunks and add to FAISS index
    vectors = embeddings.embed_documents([c.page_content for c in new_chunks])
    vectorstore.index.add(np.array(vectors).astype("float32"))
    # Update BM25 index (rebuild for simplicity)
    hybrid_store.documents.extend(new_chunks)
    hybrid_store.bm25 = HybridVectorStore(vectorstore.index, hybrid_store.documents).bm25
    return {"status": "success", "ingested_documents": len(new_chunks)}
