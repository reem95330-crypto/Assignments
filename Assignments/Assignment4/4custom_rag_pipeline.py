import requests
from bs4 import BeautifulSoup
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit(
        "FAISS import failed. Try: pip install faiss-cpu\n"
        f"Original error: {e}"
    )

from sentence_transformers import SentenceTransformer


# Documents of our choice: AI-related Wikipedia articles
URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
]


def load_url(url: str) -> dict:
    """Load text from a given URL."""
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "table", "sup"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cleaned = "\n".join(lines)

    return {"source": url, "text": cleaned}


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into smaller chunks with overlap."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def build_index(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """Embed text chunks and build a FAISS index."""
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return model, index, embeddings


def retrieve(query: str, model: SentenceTransformer, index, chunks: list[str], metadatas: list[dict], k: int = 5):
    """Retrieve top-k relevant chunks based on a query."""
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "score": float(score),
            "text": chunks[idx],
            "source": metadatas[idx]["source"],
        })
    return results


def main():
    print("1. Loading custom documents...")
    docs = [load_url(u) for u in URLS]

    print("2. Chunking custom documents...")
    chunks: list[str] = []
    metadatas: list[dict] = []
    for d in docs:
        c = chunk_text(d["text"], chunk_size=1000, overlap=150)
        chunks.extend(c)
        metadatas.extend([{"source": d["source"]}] * len(c))

    print(f"Total chunks created: {len(chunks)}")

    print("3. Embedding chunks and storing in FAISS vector index...")
    model, index, _ = build_index(chunks)

    query = "How do artificial neural networks compare to the structure of the human brain?"
    print("\n4. Retrieving relevant chunks for the Query:", query)

    results = retrieve(query, model, index, chunks, metadatas, k=3)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} | score={r['score']:.3f} ---")
        print("Source:", r["source"])
        print(r["text"][:600] + "...\n")


if __name__ == "__main__":
    main()
