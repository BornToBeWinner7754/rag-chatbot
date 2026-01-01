import time
from app.rag.loader import load_docs
from app.rag.splitter import split_docs
from app.rag.embeddings import get_embeddings
from app.rag.vectorestore import save_index, load_index

def run_worker():
    print("Background indexer started")

    while True:
        print("Checking for new docs...")
        docs = load_docs("data/docs.pdf")
        chunks = split_docs(docs)

        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(
            [c.page_content for c in chunks]
        )

        index = load_index(len(vectors[0]))
        index.add(np.array(vectors).astype("float32"))
        save_index(index)

        print("Index updated")
        time.sleep(300)  # every 5 minutes

if __name__ == "__main__":
    run_worker()
