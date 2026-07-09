"""
Jalon 5 — Interface et finitions.

"""

import os
import sys

from dotenv import load_dotenv

from rag import CodeDuTravailRAG

# Seuil calibré empiriquement sur le corpus de démo (cf. README) : en dessous,
# on avertit l'utilisateur que la réponse repose sur un contexte peu pertinent.
CONFIDENCE_DISTANCE_THRESHOLD = 1.0


def print_sources(sources) -> None:
    if not sources:
        return
    print("\nSources consultées :")
    for c in sources:
        confidence_flag = " (⚠️ similarité faible)" if c.distance > CONFIDENCE_DISTANCE_THRESHOLD else ""
        print(f"  - Article {c.numero} — {c.titre} [{c.theme}]{confidence_flag}")


def main():
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        print("Erreur : la variable d'environnement GROQ_API_KEY n'est pas définie.")
        print("Copiez .env.example vers .env et renseignez votre clé.")
        sys.exit(1)

    print("Chargement de l'assistant Code du travail (RAG)...")
    try:
        assistant = CodeDuTravailRAG()
    except Exception as e:
        print(f"Erreur au chargement de l'index vectoriel : {e}")
        print("Avez-vous lancé `python src/prepare_corpus.py` puis `python src/build_index.py` ?")
        sys.exit(1)

    print("\nAssistant Code du travail — tapez 'quit' ou 'exit' pour quitter.\n")

    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Au revoir.")
            break

        response = assistant.ask(question)

        if response.blocked:
            print(f"\n[Message bloqué] {response.block_reason}\n")
            continue

        print(f"\nAssistant > {response.answer}")
        print_sources(response.sources)
        print()


if __name__ == "__main__":
    main()
