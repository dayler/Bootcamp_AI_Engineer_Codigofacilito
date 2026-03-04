"""Tests para el módulo rag.index_ops."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag.index_ops import (
    compute_doc_hash,
    deduplicate_chunks,
    generate_chunk_id,
    ingest_new_version,
    load_registry,
    save_registry,
    sync_documents,
)


class TestComputeDocHash:
    def test_compute_doc_hash_deterministic(self):
        """El mismo contenido siempre produce el mismo hash."""
        content = "Hola mundo"
        hash1 = compute_doc_hash(content)
        hash2 = compute_doc_hash(content)
        assert hash1 == hash2

    def test_compute_doc_hash_different(self):
        """Contenido distinto produce hashes distintos."""
        hash1 = compute_doc_hash("Contenido A")
        hash2 = compute_doc_hash("Contenido B")
        assert hash1 != hash2


class TestGenerateChunkId:
    def test_generate_chunk_id_format(self):
        """El ID tiene el formato correcto con separadores ::."""
        chunk_id = generate_chunk_id("doc_001", 0, "texto de prueba")
        parts = chunk_id.split("::")
        assert len(parts) == 3
        assert parts[0] == "doc_001"
        assert parts[1] == "chunk_0"
        assert len(parts[2]) == 8  # hash parcial MD5

    def test_generate_chunk_id_deterministic(self):
        """Mismo input produce el mismo ID."""
        id1 = generate_chunk_id("doc_001", 0, "texto")
        id2 = generate_chunk_id("doc_001", 0, "texto")
        assert id1 == id2


class TestRegistry:
    def test_load_registry_missing_file(self):
        """Si el archivo no existe, devuelve dict vacío."""
        result = load_registry(Path("/tmp/nonexistent_registry_xyz.json"))
        assert result == {}

    def test_save_and_load_registry(self):
        """Guardar y cargar un registry produce datos iguales."""
        registry = {
            "doc_001": {"hash": "abc123", "chunk_ids": ["c1", "c2"]},
            "doc_002": {"hash": "def456", "chunk_ids": ["c3"]},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = Path(tmp.name)

        save_registry(registry, path)
        loaded = load_registry(path)
        assert loaded == registry
        path.unlink()


class TestSyncDocuments:
    def test_sync_documents_new(self, chroma_collection):
        """Sincronizar docs nuevos los indexa y actualiza el registry."""
        docs = [
            {"id": "doc_a", "content": "Contenido del documento A"},
            {"id": "doc_b", "content": "Contenido del documento B"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            registry_path = tmp.name

        counters = sync_documents(docs, chroma_collection, registry_path)
        assert counters["new"] == 2
        assert counters["modified"] == 0
        assert counters["deleted"] == 0

        registry = load_registry(Path(registry_path))
        assert "doc_a" in registry
        assert "doc_b" in registry
        Path(registry_path).unlink()

    def test_sync_documents_modified(self, chroma_collection):
        """Cambiar contenido de un doc y re-sincronizar detecta el cambio."""
        docs = [{"id": "doc_a", "content": "Contenido original"}]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            registry_path = tmp.name

        sync_documents(docs, chroma_collection, registry_path)

        docs[0]["content"] = "Contenido modificado"
        counters = sync_documents(docs, chroma_collection, registry_path)
        assert counters["modified"] == 1
        assert counters["new"] == 0
        Path(registry_path).unlink()

    def test_sync_documents_deleted(self, chroma_collection):
        """Quitar un doc de la lista lo elimina del registry."""
        docs = [
            {"id": "doc_a", "content": "Doc A"},
            {"id": "doc_b", "content": "Doc B"},
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            registry_path = tmp.name

        sync_documents(docs, chroma_collection, registry_path)

        # Quitar doc_b
        docs_reduced = [{"id": "doc_a", "content": "Doc A"}]
        counters = sync_documents(docs_reduced, chroma_collection, registry_path)
        assert counters["deleted"] == 1

        registry = load_registry(Path(registry_path))
        assert "doc_b" not in registry
        Path(registry_path).unlink()


class TestDeduplicateChunks:
    def test_deduplicate_chunks_exact(self):
        """Chunks con texto idéntico se eliminan."""
        chunks = [
            {"text": "Texto duplicado"},
            {"text": "Texto duplicado"},
            {"text": "Texto único"},
        ]
        embeddings = [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
        ]
        result = deduplicate_chunks(chunks, embeddings)
        assert len(result) == 2
        texts = [c["text"] for c in result]
        assert "Texto duplicado" in texts
        assert "Texto único" in texts

    def test_deduplicate_chunks_semantic(self):
        """Chunks con embeddings muy similares (cosine > 0.95) se eliminan."""
        base_vec = np.random.rand(384)
        base_vec = base_vec / np.linalg.norm(base_vec)

        # Crear un vector casi idéntico (cosine > 0.95)
        noise = np.random.rand(384) * 0.01
        similar_vec = base_vec + noise
        similar_vec = similar_vec / np.linalg.norm(similar_vec)

        # Crear un vector diferente
        diff_vec = np.random.rand(384)
        diff_vec = diff_vec / np.linalg.norm(diff_vec)

        chunks = [
            {"text": "Texto original uno"},
            {"text": "Texto casi igual dos"},
            {"text": "Texto completamente diferente tres"},
        ]
        embeddings = [base_vec.tolist(), similar_vec.tolist(), diff_vec.tolist()]

        result = deduplicate_chunks(chunks, embeddings, sim_threshold=0.95)
        # El segundo chunk debería ser eliminado por similitud semántica
        assert len(result) <= 2


class TestIngestNewVersion:
    def test_ingest_new_version(self, chroma_collection):
        """Ingestar v1 y luego v2, solo v2 tiene is_current=True."""
        doc = {"id": "doc_versioned"}

        v1 = ingest_new_version(doc, chroma_collection, ["Contenido versión 1"])
        assert v1 == 1

        v2 = ingest_new_version(doc, chroma_collection, ["Contenido versión 2"])
        assert v2 == 2

        # Verificar que solo v2 tiene is_current=True
        all_docs = chroma_collection.get(where={"doc_id": "doc_versioned"})
        for meta in all_docs["metadatas"]:
            if meta["version"] == 1:
                assert meta["is_current"] is False
            elif meta["version"] == 2:
                assert meta["is_current"] is True
