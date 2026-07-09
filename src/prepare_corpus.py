"""
Jalon 1 — Préparation des données.

"""

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "d.json"
PREPARED_PATH = Path(__file__).parent.parent / "data" / "corpus_prepared.json"


@dataclass
class Document:
    id: str
    text: str          # texte qui sera embeddé
    numero: str         # métadonnée : numéro d'article (source de vérité pour la citation)
    titre: str
    theme: str
    source: str


def clean_text(text: str) -> str:
    """Retire les scories courantes : espaces multiples, retours à la ligne
    parasites, espaces avant ponctuation (typo FR mal encodée)."""
    text = text.replace("\xa0", " ")          # espaces insécables mal décodées
    text = re.sub(r"\s+", " ", text)           # espaces/retours multiples
    text = re.sub(r"\s+([.,;:])", r"\1", text)  # espace avant ponctuation
    return text.strip()


def load_raw_corpus() -> list[dict]:
    with open(RAW_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_documents(raw_articles: list[dict]) -> list[Document]:
    documents = []
    for art in raw_articles:
        texte = clean_text(art["texte"])
        titre = clean_text(art["titre"])
        numero = art["numero"].strip()

        # Le texte à embedder combine numéro + titre + contenu : ça aide la
        # recherche par similarité à retrouver l'article même quand la
        # question mentionne son numéro ou reformule le titre.
        texte_embedding = f"Article {numero} — {titre}. {texte}"

        documents.append(
            Document(
                id=f"art-{numero}",
                text=texte_embedding,
                numero=numero,
                titre=titre,
                theme=art["theme"],
                source="Corpus de démo (Option C) — À remplacer par API Légifrance / LEGI en production",
            )
        )
    return documents


def quality_check(documents: list[Document], n: int = 10) -> None:
    """Jalon 1 - contrôle qualité : afficher n documents au hasard."""
    print(f"\n=== Contrôle qualité : {min(n, len(documents))} documents au hasard ===\n")
    sample = random.sample(documents, min(n, len(documents)))
    for doc in sample:
        print(f"[{doc.id}] ({doc.theme})")
        print(f"  {doc.text}")
        print()


def main():
    raw = load_raw_corpus()
    documents = build_documents(raw)

    themes = {d.theme for d in documents}
    print(f"{len(documents)} documents préparés, couvrant {len(themes)} thèmes :")
    for t in sorted(themes):
        print(f"  - {t}")

    quality_check(documents)

    with open(PREPARED_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in documents], f, ensure_ascii=False, indent=2)
    print(f"\nCorpus préparé sauvegardé -> {PREPARED_PATH}")


if __name__ == "__main__":
    main()
