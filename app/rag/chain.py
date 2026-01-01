import time
from app.utils.logger import logger

def create_rag_chain(vectorstore, llm, embeddings, cross_encoder):

    def run(query: str):
        logger.info("query_received", query=query)

        # -----------------------
        # Query rewriting
        # -----------------------
        start = time.time()
        rewrite_prompt = f"Rewrite the question clearly: {query}"
        rewritten = llm.invoke(rewrite_prompt).content
        logger.info(
            "query_rewritten",
            rewritten_query=rewritten,
            time_taken=time.time() - start
        )

        # -----------------------
        # Hybrid retrieval
        # -----------------------
        start = time.time()
        docs = vectorstore.hybrid_search(rewritten, k=10, embeddings=embeddings)
        logger.info(
            "retrieval_done",
            documents_retrieved=len(docs),
            time_taken=time.time() - start
        )

        # -----------------------
        # Cross-encoder reranking
        # -----------------------
        if cross_encoder and docs:
            start = time.time()
            pairs = [(rewritten, d.page_content) for d in docs]
            scores = cross_encoder.predict(pairs)
            docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
            logger.info(
                "reranking_done",
                time_taken=time.time() - start
            )

        # -----------------------
        # Context building
        # -----------------------
        top_docs = docs[:4]
        context = "\n\n".join(d.page_content for d in top_docs)

        # -----------------------
        # LLM answer generation
        # -----------------------
        start = time.time()
        prompt = f"""
Answer ONLY from context.
If not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
        answer = llm.invoke(prompt).content
        logger.info(
            "llm_answer_generated",
            answer_tokens=len(answer.split()),
            time_taken=time.time() - start
        )

        # -----------------------
        # Answer validation
        # -----------------------
        start = time.time()
        validation_prompt = f"""
Is the answer fully supported by context?
Answer yes or no.

Context:
{context}

Answer:
{answer}
"""
        validation = llm.invoke(validation_prompt).content.lower()
        logger.info(
            "answer_validated",
            validation_result=validation,
            time_taken=time.time() - start
        )

        if "no" in validation:
            logger.warning("answer_rejected")
            return "I don't know"

        logger.info("request_completed")
        return answer

    return run
