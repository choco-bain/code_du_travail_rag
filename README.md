Voici un résumé du README :

## Assistant Code du travail (RAG)

Assistant CLI qui répond aux questions de droit du travail français en citant systématiquement les articles sources, et refuse de répondre si l'info n'est pas dans sa base. **Corpus de démo** (16 articles écrits à la main, Option C) — à remplacer par une vraie extraction Légifrance/LEGI pour un usage réel.

**Réponses aux 5 questions de réflexion :**
1. **Chunking** : un chunk = un article (précision de citation max, articles déjà courts et autonomes). Regrouper par section diluerait la précision.
2. **Traçabilité** : le numéro d'article est stocké à la fois dans le texte embeddé et dans les métadonnées ChromaDB — c'est cette dernière, jamais le texte généré par le LLM, qui sert de source de vérité pour l'affichage final.
3. **Fraîcheur** : chaque doc a un champ `source` indiquant son caractère provisoire ; en prod, il porterait la date d'extraction.
4. **Réponses conditionnelles** : le prompt impose de donner la règle générale + réserves plutôt que de trancher.
5. **Frontière conseil juridique** : question factuelle → réponse directe sourcée ; question d'interprétation → rappel du cadre légal + orientation vers un professionnel, jamais de tranchage.

**Architecture** : `data/` (corpus brut, corpus préparé, base vectorielle) + `src/` (6 scripts correspondant aux jalons 1 à 6 : prepare_corpus, build_index, test_retrieval, moderation, rag, cli).

**Installation/lancement** : venv classique, `.env` avec clé Groq, puis exécution séquentielle des scripts (prepare → build_index → test_retrieval → cli). L'indexation ne se refait pas si la base existe déjà.

**Choix techniques** : embeddings multilingues `paraphrase-multilingual-MiniLM-L12-v2`, ChromaDB persistant, génération via Groq `llama-3.3-70b-versatile` (température basse), modération via un second appel Groq plus léger, avertissement juridique ajouté par le code (pas par le prompt) pour garantir sa présence à 100%.

**Limites** : corpus restreint, seuil de confiance non calibré sur données réelles, pas de recherche hybride lexicale/vectorielle, pas d'historique multi-tour.