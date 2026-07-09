"""
Jalon 2 — Chunking et indexation.

"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
PREPARED_PATH = DATA_DIR / "corpus_prepared.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "code_du_travail"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_prepared_documents() -> list[dict]:
    with open(PREPARED_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_chroma_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(force_reindex: bool = False) -> None:
    client = get_chroma_client()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force_reindex:
        collection = client.get_collection(COLLECTION_NAME)
        stored_model = collection.metadata.get("embedding_model") if collection.metadata else None
        print(
            f"Collection '{COLLECTION_NAME}' déjà présente sur disque "
            f"({collection.count()} chunks, modèle : {stored_model}). "
            f"Pas de réindexation (utilisez force_reindex=True pour forcer)."
        )
        return

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    print(f"Chargement du modèle d'embedding : {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    documents = load_prepared_documents()
    print(f"Encodage de {len(documents)} chunks...")

    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL_NAME},
    )

    collection.add(
        ids=[d["id"] for d in documents],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "numero": d["numero"],
                "titre": d["titre"],
                "theme": d["theme"],
                "source": d["source"],
            }
            for d in documents
        ],
    )

    print(f"Index construit et persisté -> {CHROMA_DIR} ({collection.count()} chunks)")

    # Contrôle qualité : vérifier qu'aucun chunk n'est coupé en plein milieu
    # d'une phrase (heuristique simple : le texte doit se terminer par une
    # ponctuation forte).
    bad = [d["id"] for d in documents if not d["text"].rstrip().endswith((".", "?", "!"))]
    if bad:
        print(f"ATTENTION : {len(bad)} chunk(s) ne se terminent pas par une ponctuation forte : {bad}")
    else:
        print("Contrôle qualité : tous les chunks se terminent proprement (pas de coupure en plein milieu de phrase).")


if __name__ == "__main__":
    build_index()
