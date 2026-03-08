from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
import json
import os


def load_chunks_from_indexer(json_path: str) -> list[Document]:
    """
    Load structured chunks produced by resume_indexer.py.
    Each chunk becomes a LangChain Document with section metadata.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for chunk in data["chunks"]:
        docs.append(Document(
            page_content=chunk["content"],
            metadata={
                "section": chunk["section"],
                "source":  data["source"],
            }
        ))
    print(f"  Loaded {len(docs)} chunks from '{json_path}'")
    return docs


def create_vector_store():
    """
    Reads documents, splits them into chunks, creates embeddings,
    and saves them to a local FAISS vector store.
    """
    all_docs = []

    # ── Primary source: structured resume chunks from resume_indexer.py ──
    resume_index_path = "me/index_data.json"
    if os.path.exists(resume_index_path):
        print("Loading resume chunks from resume_indexer output...")
        all_docs.extend(load_chunks_from_indexer(resume_index_path))
    else:
        print(f"Warning: '{resume_index_path}' not found.")
        print("Run resume_indexer.py first:")
        print("  python resume_indexer.py --pdf me/MohitKumarResume.pdf --out me/index_data.json")

    # ── Secondary source: plain text files (summary, etc.) ──
    txt_docs = ["me/summary.txt"]
    for doc_path in txt_docs:
        if os.path.exists(doc_path):
            print(f"Loading text file: {doc_path}")
            loader = TextLoader(doc_path, encoding="utf-8")
            all_docs.extend(loader.load())
        else:
            print(f"Warning: '{doc_path}' not found, skipping.")

    if not all_docs:
        print("No documents loaded. Aborting.")
        return

    print(f"\nTotal documents/chunks to index: {len(all_docs)}")

    # Create embeddings and build FAISS index
    print("Creating embeddings and building the vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(all_docs, embeddings)

    # Save the vector store locally
    db_path = "faiss_index"
    vector_store.save_local(db_path)
    print(f"\n✅ Vector store saved to '{db_path}'.")


if __name__ == "__main__":
    os.makedirs("me", exist_ok=True)
    create_vector_store()