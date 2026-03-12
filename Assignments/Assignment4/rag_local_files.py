import glob
from typing import List

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit(
        "FAISS import failed. Try: pip install faiss-cpu\n"
        f"Original error: {e}"
    )

from sentence_transformers import SentenceTransformer

from local_loaders import load_any


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
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
    return model, index


def retrieve(
    query: str,
    model: SentenceTransformer,
    index,
    chunks: list[str],
    metadatas: list[dict],
    k: int = 5,
):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append(
            {
                "score": float(score),
                "text": chunks[idx],
                "source": metadatas[idx]["source"],
            }
        )
    return results


def main():
    # Put your documents into a folder named: documents/
    # Supported: .txt, .pdf, .docx
    paths: List[str] = []
    paths.extend(glob.glob("documents/*.txt"))
    paths.extend(glob.glob("documents/*.pdf"))
    paths.extend(glob.glob("documents/*.docx"))

    if not paths:
        raise SystemExit(
            "No documents found. Create a folder named 'documents' and add .txt/.pdf/.docx files."
        )

    docs = [load_any(p) for p in paths]

    chunks: list[str] = []
    metadatas: list[dict] = []
    for d in docs:
        c = chunk_text(d.text, chunk_size=900, overlap=150)
        chunks.extend(c)
        metadatas.extend([{"source": d.source}] * len(c))

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")

    model, index = build_index(chunks)

    query = "Summarize the key points across these documents."
    results = retrieve(query, model, index, chunks, metadatas, k=5)

    print("\nQuery:", query)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} | score={r['score']:.3f} ---")
        print("Source:", r["source"])
        print(r["text"][:600])


if __name__ == "__main__":
    main()
