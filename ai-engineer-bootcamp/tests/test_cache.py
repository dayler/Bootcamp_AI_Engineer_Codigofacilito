"""Tests para el módulo rag.cache."""

import hashlib
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag.access_control import User
from rag.cache import SemanticCache, rag_query


def _make_embed_fn():
    """Crea un embed_fn determinista basado en SHA-256.

    Genera vectores de dimensión 384 normalizados.
    """

    def _embed(text: str) -> list[float]:
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        repeated = hash_bytes * (384 // len(hash_bytes) + 1)
        raw = np.array([b / 255.0 for b in repeated[:384]], dtype=np.float32)
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        return raw.tolist()

    return _embed


@pytest.fixture
def embed_fn():
    return _make_embed_fn()


@pytest.fixture
def cache(embed_fn):
    return SemanticCache(embed_fn=embed_fn, threshold=0.95, ttl_seconds=3600)


class TestSemanticCache:
    def test_cache_miss_empty(self, cache):
        """En un caché vacío, get() devuelve None."""
        assert cache.get("cualquier query") is None

    def test_cache_put_and_get(self, cache):
        """Guardar una entrada y recuperarla con la misma query."""
        cache.put("¿Cuántas vacaciones tengo?", "Tienes 20 días", ["doc_01"])
        result = cache.get("¿Cuántas vacaciones tengo?")
        assert result == "Tienes 20 días"

    def test_cache_hit_similar_query(self):
        """Recuperar con una query semánticamente similar (cosine > threshold)."""
        # Crear embed_fn que devuelve vectores similares para queries cercanas
        base_vec = np.random.rand(384).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        noise = np.random.rand(384).astype(np.float32) * 0.01
        similar_vec = base_vec + noise
        similar_vec = similar_vec / np.linalg.norm(similar_vec)

        call_count = [0]

        def controlled_embed(text: str) -> list[float]:
            call_count[0] += 1
            if call_count[0] <= 2:  # put + first part of get
                return base_vec.tolist()
            return similar_vec.tolist()

        cache = SemanticCache(embed_fn=controlled_embed, threshold=0.95)
        cache.put("query original", "respuesta original")

        result = cache.get("query similar")
        assert result == "respuesta original"

    def test_cache_miss_different_query(self, cache):
        """Fallar al buscar con una query muy diferente."""
        cache.put("¿Cuál es el presupuesto?", "5 millones", ["doc_fin"])
        result = cache.get("¿Cómo se configura Kubernetes?")
        assert result is None

    def test_cache_ttl_expiration(self, embed_fn):
        """Entrada expirada devuelve None."""
        cache = SemanticCache(embed_fn=embed_fn, threshold=0.95, ttl_seconds=60)
        cache.put("query test", "respuesta test")

        # Manipular timestamp para que esté expirada
        cache._entries[0]["timestamp"] = time.time() - 120  # 2 min atrás

        result = cache.get("query test")
        assert result is None

    def test_invalidate_by_doc(self, cache):
        """Invalidar un doc_id elimina solo las entradas relacionadas."""
        cache.put("query A", "respuesta A", ["doc_01", "doc_02"])
        cache.put("query B", "respuesta B", ["doc_03"])
        cache.put("query C", "respuesta C", ["doc_01"])

        cache.invalidate_by_doc("doc_01")

        # Solo debe quedar la entrada de doc_03
        assert len(cache._entries) == 1
        assert cache._entries[0]["response"] == "respuesta B"

    def test_cleanup_expired(self, embed_fn):
        """Entradas con timestamps viejos se eliminan con cleanup."""
        cache = SemanticCache(embed_fn=embed_fn, threshold=0.95, ttl_seconds=60)
        cache.put("query vieja", "respuesta vieja")
        cache.put("query nueva", "respuesta nueva")

        # Hacer vieja la primera entrada
        cache._entries[0]["timestamp"] = time.time() - 120

        removed = cache.cleanup_expired()
        assert removed == 1
        assert len(cache._entries) == 1
        assert cache._entries[0]["response"] == "respuesta nueva"

    def test_hit_rate_tracking(self, cache):
        """Verificar contadores de hits y misses."""
        cache.put("query conocida", "respuesta conocida")

        # Simular hits y misses manualmente
        cache._hits = 3
        cache._misses = 7

        stats = cache.hit_rate()
        assert stats["total_queries"] == 10
        assert stats["hits"] == 3
        assert stats["misses"] == 7
        assert stats["hit_rate_percent"] == 30.0


class TestRagQuery:
    def test_rag_query_uses_cache(self, embed_fn):
        """La misma query dos veces solo llama a llm_fn una vez."""
        # Mock del vectorstore
        mock_vectorstore = MagicMock()
        mock_vectorstore.query.return_value = {
            "documents": [["Contenido del documento de prueba"]],
            "metadatas": [[{"doc_id": "doc_01", "is_current": True}]],
            "ids": [["chunk_01"]],
        }

        # Mock del LLM
        mock_llm = MagicMock(return_value="Respuesta generada por el LLM")

        user = User(
            user_id="test_user",
            roles=["employee"],
            department="general",
            access_level="public",
        )

        cache = SemanticCache(embed_fn=embed_fn, threshold=0.95, ttl_seconds=3600)

        # Primera query: cache miss, llama al LLM
        response1 = rag_query(
            query="¿Cuál es la política?",
            user=user,
            vectorstore=mock_vectorstore,
            cache=cache,
            llm_fn=mock_llm,
            embed_fn=embed_fn,
        )

        # Segunda query: cache hit, NO llama al LLM
        response2 = rag_query(
            query="¿Cuál es la política?",
            user=user,
            vectorstore=mock_vectorstore,
            cache=cache,
            llm_fn=mock_llm,
            embed_fn=embed_fn,
        )

        assert response1 == response2
        assert mock_llm.call_count == 1
        assert cache._hits == 1
        assert cache._misses == 1
