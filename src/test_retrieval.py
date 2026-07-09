"""
Jalon 3 — Recherche seule (avant tout appel LLM).

"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

from build_index import get_chroma_client, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# (question, numéro d'article attendu)
TEST_QUESTIONS = [
    ("Quelle est la durée légale du préavis pour un CDI ?", "L1234-1"),
    ("Combien de jours de congés payés par mois de travail ?", "L3141-3"),
    ("Comment fonctionne la rupture conventionnelle ?", "L1237-11"),
    ("Qu'est-ce qu'un licenciement économique ?", "L1233-3"),
    ("Le harcèlement moral au travail est-il interdit ?", "L1152-1"),
]

TOP_K = 3


def run_retrieval_tests() -> bool:
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    all_passed = True
    print(f"=== Validation du retrieval (top-{TOP_K}) ===\n")

    for question, expected_article in TEST_QUESTIONS:
        query_embedding = model.encode([question]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

        retrieved_numeros = [m["numero"] for m in results["metadatas"][0]]
        passed = expected_article in retrieved_numeros
        all_passed &= passed

        status = "OK" if passed else "ECHEC"
        print(f"[{status}] \"{question}\"")
        print(f"       attendu : {expected_article} | top-{TOP_K} retourné : {retrieved_numeros}\n")

    if all_passed:
        print("Tous les tests de retrieval passent. On peut brancher le LLM (Jalon 4).")
    else:
        print("Au moins un test échoue : revoir le chunking, l'embedding ou le corpus avant de continuer.")

    return all_passed


if __name__ == "__main__":
    run_retrieval_tests()
