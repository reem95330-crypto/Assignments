# Load → Embed → Store → Retrieve (RAG mini-project)

This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using documents:

1. **Load**: fetch/extract text from documents (URLs or local files)
2. **Chunk**: split text into overlapping chunks
3. **Embed**: create vector embeddings with `sentence-transformers`
4. **Store**: index vectors in **FAISS**
5. **Retrieve**: search top-k relevant chunks for a query

## Files
- `rag_pipeline.py` — RAG pipeline using URLs
- `rag_pipeline.ipynb` — notebook version (URLs)
- `local_loaders.py` — loaders for `.txt`, `.pdf`, `.docx`
- `rag_local_files.py` — RAG pipeline using local files in `documents/`
- `requirements.txt` — dependencies
- `run.bat` — Windows helper script
- `.gitignore` — typical Python ignores

## Setup (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
