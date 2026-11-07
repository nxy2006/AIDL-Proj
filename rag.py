"""
Step 1 — RAG: read the knowledge base, store it in ChromaDB, retrieve chunks.

Run once to build the database:   python rag.py
agent.py then calls retrieve(question) to get the top chunks.
"""
import re
from pathlib import Path
import chromadb

DATA_DIR = Path(__file__).parent / "data"
DB_DIR = Path(__file__).parent / "chroma_db"

# Chroma's built-in embedding model (all-MiniLM-L6-v2). Free, runs locally,
# no API key. Downloads ~80 MB the first time.
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection("dental_kb")


def split_into_chunks(text: str) -> list[str]:
    """The .txt files separate entries with long lines of dashes,
    and OralHealthEducation.txt uses '--- END CHUNK 001 ---' markers."""
    parts = re.split(r"\n\s*-{20,}\s*\n|--- END CHUNK \d+ ---", text)
    # drop tiny leftovers (blank bits, '=====' banners)
    return [p.strip() for p in parts if len(p.strip()) > 150]


def build_index():
    """Read every .txt in data/, chunk it, embed it, store it."""
    ids, docs, metas = [], [], []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_into_chunks(text)):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metas.append({"source": path.name})
    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(docs)} chunks from {DATA_DIR}")


def retrieve(query: str, k: int = 3) -> str:
    """Return the k most relevant chunks as one string for the prompt."""
    if collection.count() == 0:
        build_index()
    res = collection.query(query_texts=[query], n_results=k)
    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append(f"[Source: {meta['source']}]\n{doc}")
    return "\n\n---\n\n".join(chunks)


if __name__ == "__main__":
    build_index()
    print("\n=== Test: How much does scaling cost? ===\n")
    print(retrieve("How much does scaling cost?")[:1500])