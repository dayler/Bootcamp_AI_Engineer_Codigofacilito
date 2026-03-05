"""Módulo de caché semántico para RAG.

Evita llamadas repetidas al LLM almacenando respuestas previas
y buscándolas por similitud semántica con las queries entrantes.
"""

import time
from typing import Callable

import numpy as np

from access_control import User, retrieve_with_access


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCache:
    """Caché semántico que almacena respuestas del LLM y las busca por similitud.

    Cada entrada contiene: embedding, query, response, doc_ids, timestamp.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ):
        """Inicializa el caché semántico.

        Args:
            embed_fn: Función que recibe texto y devuelve un vector de embedding.
            threshold: Umbral de similitud coseno para considerar cache hit.
            ttl_seconds: Tiempo de vida de las entradas en segundos.
        """
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self._entries: list[dict] = []
        self._hits = 0
        self._misses = 0

    def get(self, query: str) -> str | None:
        """Busca una respuesta en caché para la query dada.

        Args:
            query: Texto de la consulta.

        Returns:
            Respuesta almacenada si hay cache hit, None si es miss.
        """
        query_embedding = np.array(self.embed_fn(query))
        now = time.time()

        for entry in self._entries:
            # Saltar entradas expiradas
            if entry["timestamp"] + self.ttl_seconds < now:
                continue

            sim = _cosine_similarity(query_embedding, np.array(entry["embedding"]))
            if sim >= self.threshold:
                return entry["response"]

        return None

    def put(
        self,
        query: str,
        response: str,
        doc_ids: list[str] = None,
    ) -> None:
        """Almacena una nueva entrada en el caché.

        Args:
            query: Texto de la consulta.
            response: Respuesta generada por el LLM.
            doc_ids: IDs de los documentos usados para generar la respuesta.
        """
        embedding = self.embed_fn(query)
        self._entries.append({
            "embedding": embedding,
            "query": query,
            "response": response,
            "doc_ids": doc_ids or [],
            "timestamp": time.time(),
        })

    def invalidate_by_doc(self, doc_id: str) -> None:
        """Elimina todas las entradas del caché asociadas a un doc_id.

        Args:
            doc_id: ID del documento a invalidar.
        """
        self._entries = [
            entry for entry in self._entries
            if doc_id not in entry.get("doc_ids", [])
        ]

    def cleanup_expired(self) -> int:
        """Elimina todas las entradas expiradas del caché.

        Returns:
            Número de entradas eliminadas.
        """
        now = time.time()
        before = len(self._entries)
        self._entries = [
            entry for entry in self._entries
            if entry["timestamp"] + self.ttl_seconds >= now
        ]
        removed = before - len(self._entries)
        return removed

    def hit_rate(self) -> dict:
        """Devuelve estadísticas de hit rate del caché.

        Returns:
            Diccionario con total_queries, hits, misses y hit_rate_percent.
        """
        total = self._hits + self._misses
        return {
            "total_queries": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": (self._hits / total * 100) if total > 0 else 0.0,
        }


def rag_query(
    query: str,
    user: User,
    vectorstore,
    cache: SemanticCache,
    llm_fn: Callable[[str], str],
    embed_fn: Callable[[str], list[float]],
    rerank_fn: Callable = None,
) -> str:
    """Ejecuta una query RAG con caché semántico y control de acceso.

    Pipeline:
    1. Buscar en caché
    2. Si miss, recuperar chunks con control de acceso
    3. Opcionalmente re-rankear
    4. Generar respuesta con LLM
    5. Guardar en caché

    Args:
        query: Texto de la consulta.
        user: Usuario que realiza la consulta.
        vectorstore: Colección de ChromaDB.
        cache: Instancia de SemanticCache.
        llm_fn: Función que recibe un prompt y devuelve texto.
        embed_fn: Función de embedding.
        rerank_fn: Función opcional de reranking.

    Returns:
        Respuesta generada.
    """
    # 1. Intentar cache hit
    cached = cache.get(query)
    if cached is not None:
        cache._hits += 1
        return cached

    # 2. Cache miss
    cache._misses += 1

    # 3. Recuperar chunks con control de acceso
    results = retrieve_with_access(query, user, vectorstore)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        response = llm_fn(f"No se encontraron documentos relevantes. Pregunta: {query}")
        cache.put(query, response, [])
        return response

    # 4. Reranking opcional
    if rerank_fn is not None:
        ranked = rerank_fn(query, documents)
        documents = ranked

    # 5. Construir prompt y llamar al LLM
    context = "\n\n---\n\n".join(documents)
    prompt = (
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {query}\n\n"
        f"Responde basándote únicamente en el contexto proporcionado."
    )

    response = llm_fn(prompt)

    # 6. Guardar en caché
    doc_ids = list({m.get("doc_id", "") for m in metadatas if m.get("doc_id")})
    cache.put(query, response, doc_ids)

    return response
