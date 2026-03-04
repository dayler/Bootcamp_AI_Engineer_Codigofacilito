"""
rag/retrieval.py — Clase 6: RAG Avanzado

Técnicas implementadas:
1. BM25 (keyword search)
2. Hybrid Retrieval (BM25 + vector)
3. Multi-query expansion
4. Reciprocal Rank Fusion (RRF)
5. Cross-encoder reranking
6. Context compression
7. Pipeline completo (advanced_rag_query)

LLM: GPT-OSS vía Groq (compatible OpenAI)
Embeddings: sentence-transformers (local)
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag.ingestion import Chunk
from rag.vectorstore import SearchResult, search as vector_search

load_dotenv()

# ---------------------------------------------------------------------------
# Configuración LLM (Groq vía OpenAI-compatible API)
# ---------------------------------------------------------------------------

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _get_groq_client() -> OpenAI:
    """Crea un cliente OpenAI apuntando a la API de Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY no está configurada. "
            "Exporta la variable de entorno: export GROQ_API_KEY='gsk_...'"
        )
    return OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


_groq_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Singleton del cliente Groq."""
    global _groq_client
    if _groq_client is None:
        _groq_client = _get_groq_client()
    return _groq_client


# ---------------------------------------------------------------------------
# Tracker de uso de tokens (para métricas en compare_rag.py)
# ---------------------------------------------------------------------------

_usage_tracker = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def reset_usage_tracker():
    """Reinicia el acumulador de tokens."""
    global _usage_tracker
    _usage_tracker = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def get_usage():
    """Retorna copia del acumulador actual de tokens."""
    return dict(_usage_tracker)


# ---------------------------------------------------------------------------
# Helper: LLM call wrapper
# ---------------------------------------------------------------------------

def call_llm(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """Llama al LLM (GPT-OSS vía Groq) y retorna el texto de respuesta.

    Acumula métricas de uso en el tracker interno.
    """
    global _usage_tracker
    client = _get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
    )

    # Acumular métricas
    if response.usage:
        _usage_tracker["input_tokens"] += response.usage.prompt_tokens or 0
        _usage_tracker["output_tokens"] += response.usage.completion_tokens or 0
    _usage_tracker["calls"] += 1

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 1. BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    """Índice de búsqueda por keywords usando BM25 (Okapi)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenización simple: lowercase + regex \\w+."""
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Busca por BM25 y retorna lista de (Chunk, score)."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(self.chunks[i], float(s)) for i, s in ranked if s > 0]


# ---------------------------------------------------------------------------
# 2. HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combina BM25 (keyword) y vector search (semántico) con score normalizado."""

    def __init__(
        self, collection, chunks: list[Chunk], alpha: float = 0.5
    ) -> None:
        """
        Args:
            collection: Colección de ChromaDB.
            chunks: Lista de Chunks indexados.
            alpha: Peso del vector search (1-alpha = peso BM25).
        """
        self.collection = collection
        self.chunks = chunks
        self.alpha = alpha
        self.bm25 = BM25Index(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Búsqueda híbrida con scores normalizados y combinados."""
        candidates = top_k * 2

        # --- BM25 ---
        bm25_results = self.bm25.search(query, top_k=candidates)
        bm25_scores: dict[str, float] = {}
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results)
            if max_bm25 > 0:
                for chunk, score in bm25_results:
                    bm25_scores[chunk.chunk_id] = score / max_bm25

        # --- Vector search ---
        vec_results = vector_search(self.collection, query, n_results=candidates)
        vec_scores: dict[str, float] = {}
        vec_content: dict[str, SearchResult] = {}
        if vec_results:
            max_vec = max(r.score for r in vec_results)
            if max_vec > 0:
                for r in vec_results:
                    vec_scores[r.chunk_id] = r.score / max_vec
                    vec_content[r.chunk_id] = r

        # --- Combinar scores ---
        all_ids = set(bm25_scores.keys()) | set(vec_scores.keys())
        combined: list[tuple[str, float]] = []
        for cid in all_ids:
            vec_norm = vec_scores.get(cid, 0.0)
            bm25_norm = bm25_scores.get(cid, 0.0)
            score = self.alpha * vec_norm + (1 - self.alpha) * bm25_norm
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)

        # --- Construir resultados ---
        results = []
        for cid, score in combined[:top_k]:
            if cid in vec_content:
                r = vec_content[cid]
                results.append(SearchResult(
                    content=r.content, metadata=r.metadata,
                    score=score, chunk_id=cid,
                ))
            elif cid in self._chunk_map:
                chunk = self._chunk_map[cid]
                results.append(SearchResult(
                    content=chunk.content, metadata=chunk.metadata,
                    score=score, chunk_id=cid,
                ))

        return results


# ---------------------------------------------------------------------------
# 3. Multi-query generation
# ---------------------------------------------------------------------------

def generate_multi_queries(original_query: str, n: int = 3) -> list[str]:
    """Genera n reformulaciones de la query usando el LLM.

    Siempre incluye la query original como primera entrada.
    """
    prompt = (
        f"Genera {n} reformulaciones diferentes de la siguiente pregunta. "
        "Cada reformulación debe usar vocabulario distinto pero mantener el mismo significado.\n\n"
        f"Pregunta original: {original_query}\n\n"
        'Responde SOLO con un JSON array de strings. Ejemplo:\n'
        '["reformulación 1", "reformulación 2", "reformulación 3"]'
    )

    try:
        response = call_llm(prompt)
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [original_query] + queries[:n]
    except Exception:
        pass

    return [original_query]


# ---------------------------------------------------------------------------
# 4. Multi-query search
# ---------------------------------------------------------------------------

def multi_query_search(
    collection, query: str, n_results: int = 5
) -> list[SearchResult]:
    """Busca con múltiples reformulaciones y acumula scores."""
    queries = generate_multi_queries(query)
    score_map: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for q in queries:
        results = vector_search(collection, q, n_results=n_results)
        for r in results:
            score_map[r.chunk_id] = score_map.get(r.chunk_id, 0.0) + r.score
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [
        SearchResult(
            content=result_map[cid].content,
            metadata=result_map[cid].metadata,
            score=score,
            chunk_id=cid,
        )
        for cid, score in ranked
        if cid in result_map
    ]


# ---------------------------------------------------------------------------
# 5. Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60
) -> list[tuple[str, float]]:
    """Fusiona múltiples listas de resultados usando RRF.

    Returns:
        Lista de (chunk_id, rrf_score) ordenada descendente.
    """
    rrf_scores: dict[str, float] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            rrf_scores[r.chunk_id] = (
                rrf_scores.get(r.chunk_id, 0.0) + 1 / (k + rank + 1)
            )
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 6. Reranking con Cross-Encoder
# ---------------------------------------------------------------------------

_cross_encoder = None


def _get_cross_encoder():
    """Carga el cross-encoder (lazy, singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        print("  Cargando cross-encoder (primera vez puede tardar unos minutos)...")
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def rerank(
    query: str, results: list[SearchResult], top_k: int = 5
) -> list[SearchResult]:
    """Re-ordena resultados usando un cross-encoder."""
    if not results:
        return []

    model = _get_cross_encoder()
    pairs = [(query, r.content) for r in results]
    scores = model.predict(pairs)

    reranked = []
    for r, score in zip(results, scores):
        reranked.append(SearchResult(
            content=r.content,
            metadata=r.metadata,
            score=float(score),
            chunk_id=r.chunk_id,
        ))

    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# 7. Compresión de contexto (LLM)
# ---------------------------------------------------------------------------

def compress_context(query: str, chunks: list[str]) -> str:
    """Extrae SOLO las oraciones relevantes de los chunks usando el LLM."""
    combined = "\n---\n".join(chunks)
    prompt = (
        f"Pregunta del usuario: {query}\n\n"
        f"Contexto recuperado:\n{combined}\n\n"
        "Extrae SOLO las oraciones relevantes para responder la pregunta. "
        "No inventes información. No parafrasees. "
        "Solo selecciona y devuelve las oraciones más relevantes."
    )
    return call_llm(prompt)


# ---------------------------------------------------------------------------
# 8. Compresión de contexto (reranker, sin LLM)
# ---------------------------------------------------------------------------

def compress_with_reranker(
    query: str, chunks: list[str], top_sentences: int = 10
) -> str:
    """Comprime el contexto rerankeando oraciones individuales."""
    sentences = []
    for chunk in chunks:
        for s in chunk.split(". "):
            s = s.strip()
            if len(s) > 20:
                sentences.append(s if s.endswith(".") else s + ".")

    if not sentences:
        return " ".join(chunks)

    model = _get_cross_encoder()
    pairs = [(query, s) for s in sentences]
    scores = model.predict(pairs)

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    top = [s for s, _ in ranked[:top_sentences]]
    return " ".join(top)


# ---------------------------------------------------------------------------
# 9. Pipeline completo: Advanced RAG Query
# ---------------------------------------------------------------------------

def advanced_rag_query(collection, chunks: list[Chunk], query: str) -> str:
    """Pipeline completo de RAG Avanzado.

    1. Multi-query: genera reformulaciones
    2. Búsqueda híbrida: por cada query, busca con HybridRetriever
    3. Deduplicar resultados por chunk_id
    4. Reranking: reordena candidatos únicos con cross-encoder
    5. Compresión: extrae solo lo relevante con LLM
    6. Genera respuesta final con LLM
    """
    # 1. Multi-query
    queries = generate_multi_queries(query)
    print(f"  Multi-query: {len(queries)} queries generadas")

    # 2. Búsqueda híbrida por cada query
    hybrid = HybridRetriever(collection, chunks)
    all_results: list[SearchResult] = []
    for q in queries:
        results = hybrid.search(q, top_k=5)
        all_results.extend(results)

    # 3. Deduplicar por chunk_id
    seen: set[str] = set()
    unique_results: list[SearchResult] = []
    for r in all_results:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            unique_results.append(r)
    print(f"  Candidatos únicos: {len(unique_results)}")

    # 4. Reranking
    reranked = rerank(query, unique_results, top_k=5)
    print(f"  Rerankeados: {len(reranked)}")

    # 5. Compresión de contexto
    chunk_texts = [r.content for r in reranked]
    compressed = compress_context(query, chunk_texts)
    print(f"  Contexto comprimido: {len(compressed)} chars")

    # 6. Generar respuesta final
    final_prompt = (
        "Responde la siguiente pregunta usando SOLO el contexto proporcionado.\n"
        'Si no puedes responder con el contexto dado, di "No tengo suficiente información".\n\n'
        f"Contexto:\n{compressed}\n\n"
        f"Pregunta: {query}\n\n"
        "Respuesta:"
    )
    return call_llm(final_prompt)
