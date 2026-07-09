"""
Jalon 4 — Génération avec citations.

"""

import os
from dataclasses import dataclass

from groq import Groq
from sentence_transformers import SentenceTransformer

from build_index import get_chroma_client, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from moderation import check_message

GENERATION_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4
TEMPERATURE = 0.1

LEGAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ Cet assistant ne fournit pas de conseil juridique. "
    "Consultez un avocat ou l'inspection du travail pour votre situation personnelle."
)

SYSTEM_PROMPT = """Tu es un assistant d'information sur le Code du travail français.

Règles strictes :
1. Tu réponds UNIQUEMENT à partir des extraits d'articles fournis dans le contexte ci-dessous.
   Tu n'inventes JAMAIS un numéro d'article, un délai ou un chiffre qui n'est pas dans le contexte.
2. Chaque affirmation de ta réponse doit être rattachée explicitement à un numéro d'article
   du contexte (ex. "selon l'article L1234-1...").
3. Si le contexte fourni ne contient pas l'information demandée, réponds exactement :
   "Je ne trouve pas cette information dans ma base." N'essaie pas de deviner.
4. Certaines questions dépendent de la taille de l'entreprise, de la convention collective,
   ou d'une situation individuelle : dans ce cas, donne la règle générale que tu peux sourcer,
   puis indique clairement les réserves ("cela peut varier selon...") au lieu d'affirmer
   une réponse unique.
5. Si la question demande une appréciation ou une interprétation d'un cas personnel
   (ex. "mon licenciement est-il abusif ?"), ne tranche pas : rappelle le cadre légal
   pertinent que tu peux sourcer, puis oriente explicitement vers un professionnel
   (avocat, inspection du travail, défenseur syndical).
6. Ne mentionne jamais l'avertissement juridique toi-même : il est ajouté automatiquement
   après ta réponse.
"""


@dataclass
class RetrievedChunk:
    numero: str
    titre: str
    theme: str
    text: str
    distance: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedChunk]
    blocked: bool = False
    block_reason: str = ""


class CodeDuTravailRAG:
    """Orchestrateur : modération -> retrieval -> génération -> assemblage final."""

    def __init__(self, groq_api_key: str | None = None):
        self.groq_client = Groq(api_key=groq_api_key or os.environ["GROQ_API_KEY"])
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        chroma_client = get_chroma_client()
        self.collection = chroma_client.get_collection(COLLECTION_NAME)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        query_embedding = self.embedding_model.encode([question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    numero=meta["numero"],
                    titre=meta["titre"],
                    theme=meta["theme"],
                    text=doc,
                    distance=dist,
                )
            )
        return chunks

    def _build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(f"[Extrait {i} - Article {c.numero} - {c.titre}]\n{c.text}")
        return "\n\n".join(blocks)

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = self._build_context_block(chunks)
        user_prompt = f"Contexte (extraits du Code du travail) :\n\n{context}\n\nQuestion : {question}"

        response = self.groq_client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str, skip_moderation: bool = False) -> RAGResponse:
        # Étape 1 : modération (bonus jalon 6)
        if not skip_moderation:
            moderation = check_message(question, client=self.groq_client)
            if not moderation.safe:
                return RAGResponse(
                    answer="Cette question a été bloquée par le filtre de sécurité.",
                    sources=[],
                    blocked=True,
                    block_reason=moderation.reason,
                )

        # Étape 2 : retrieval
        chunks = self.retrieve(question)

        # Étape 3 : génération
        raw_answer = self.generate(question, chunks)

        # Étape 4 : assemblage final — l'avertissement est ajouté ICI,
        # par le code, jamais laissé au LLM (cf. contrainte non négociable
        # du sujet).
        final_answer = raw_answer + LEGAL_DISCLAIMER

        return RAGResponse(answer=final_answer, sources=chunks)
