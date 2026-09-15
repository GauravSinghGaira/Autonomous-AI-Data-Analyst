"""
Run once after installing dependencies:
    python scripts/setup.py

Creates the storage directories and ingests the sample RAG documents
(data dictionary, KPI definitions) into the vector store so the
document_retrieval route has something to search on a fresh install.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.ingest import ingest_directory
from app.utils.config import settings


def main() -> None:
    for path in [settings.upload_dir, settings.reports_dir, settings.chroma_persist_dir,
                 os.path.dirname(settings.log_file)]:
        os.makedirs(path, exist_ok=True)
        print(f"Ensured directory: {path}")

    n_chunks = ingest_directory()
    print(f"Ingested {n_chunks} document chunks into the vector store from {settings.docs_dir}")
    print("Setup complete.")


if __name__ == "__main__":
    main()
