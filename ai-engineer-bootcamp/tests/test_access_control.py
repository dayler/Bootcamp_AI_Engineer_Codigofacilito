"""Tests para el módulo rag.access_control."""

import hashlib
import uuid

import chromadb
import numpy as np
import pytest

from rag.access_control import (
    LEVEL_HIERARCHY,
    User,
    build_access_filter,
    ingest_document_with_access,
    retrieve_with_access,
)


def _deterministic_embed(text: str) -> list[float]:
    """Genera un embedding determinista basado en hash para tests."""
    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    repeated = hash_bytes * (384 // len(hash_bytes) + 1)
    raw = np.array([b / 255.0 for b in repeated[:384]], dtype=np.float32)
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm
    return raw.tolist()


class DeterministicEmbeddingFunction(chromadb.EmbeddingFunction):
    """Embedding function determinista para ChromaDB en tests."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [_deterministic_embed(text) for text in input]


@pytest.fixture
def access_collection():
    """Colección ChromaDB con embedding function determinista."""
    client = chromadb.Client()
    name = f"test_access_{uuid.uuid4().hex[:12]}"
    collection = client.create_collection(
        name=name,
        embedding_function=DeterministicEmbeddingFunction(),
    )
    yield collection
    try:
        client.delete_collection(name=name)
    except Exception:
        pass


class TestBuildAccessFilter:
    def test_build_access_filter_public_employee(self):
        """Usuario public solo puede ver docs public."""
        user = User(
            user_id="u1", roles=["employee"],
            department="rrhh", access_level="public",
        )
        f = build_access_filter(user)
        assert "$and" in f

        # Extraer allowed levels
        level_filter = f["$and"][1]
        allowed = level_filter["access_level"]["$in"]
        assert allowed == ["public"]

    def test_build_access_filter_internal(self):
        """Usuario internal puede ver public e internal."""
        user = User(
            user_id="u2", roles=["employee"],
            department="ingenieria", access_level="internal",
        )
        f = build_access_filter(user)
        level_filter = f["$and"][1]
        allowed = level_filter["access_level"]["$in"]
        assert "public" in allowed
        assert "internal" in allowed
        assert "confidential" not in allowed

    def test_build_access_filter_confidential_admin(self):
        """Usuario confidential puede ver los 3 niveles."""
        user = User(
            user_id="u3", roles=["admin"],
            department="finanzas", access_level="confidential",
        )
        f = build_access_filter(user)
        level_filter = f["$and"][1]
        allowed = level_filter["access_level"]["$in"]
        assert set(allowed) == {"public", "internal", "confidential"}

    def test_build_access_filter_department(self):
        """El filtro incluye el departamento del usuario y 'general'."""
        user = User(
            user_id="u4", roles=["employee"],
            department="ingenieria", access_level="internal",
        )
        f = build_access_filter(user)
        dept_filter = f["$and"][0]
        assert "$or" in dept_filter
        departments = [
            list(cond.values())[0] for cond in dept_filter["$or"]
        ]
        assert "ingenieria" in departments
        assert "general" in departments


class TestRetrieveWithAccess:
    def test_retrieve_with_access_filters_correctly(self, access_collection):
        """Solo se obtienen docs del departamento y nivel permitido."""
        # Ingestar docs de distintos departamentos y niveles
        docs_data = [
            ("doc_fin", "Datos financieros confidenciales", "finanzas", "confidential"),
            ("doc_ing", "Arquitectura de microservicios", "ingenieria", "internal"),
            ("doc_pub", "Política de vacaciones pública", "rrhh", "public"),
            ("doc_gen", "Misión de la empresa", "general", "public"),
        ]

        for doc_id, content, dept, level in docs_data:
            ingest_document_with_access(
                doc={"id": doc_id},
                chunks=[content],
                vectorstore=access_collection,
                department=dept,
                access_level=level,
            )

        # Usuario de ingeniería con nivel internal
        user = User(
            user_id="eng_01", roles=["employee"],
            department="ingenieria", access_level="internal",
        )
        results = retrieve_with_access(
            "arquitectura", user, access_collection, top_k=10
        )

        # Debe ver docs de ingeniería y general, con nivel public o internal
        doc_ids_returned = []
        if results["metadatas"][0]:
            doc_ids_returned = [m["doc_id"] for m in results["metadatas"][0]]

        # No debe ver docs confidenciales de finanzas
        assert "doc_fin" not in doc_ids_returned

    def test_retrieve_with_access_excludes_old_versions(self, access_collection):
        """Docs con is_current=False no aparecen en resultados."""
        # Insertar doc con is_current=False directamente
        access_collection.add(
            ids=["old_chunk_1"],
            documents=["Documento de versión antigua"],
            metadatas=[{
                "doc_id": "doc_old",
                "department": "general",
                "access_level": "public",
                "is_current": False,
                "version": 1,
            }],
        )
        # Insertar doc con is_current=True
        access_collection.add(
            ids=["current_chunk_1"],
            documents=["Documento de versión actual"],
            metadatas=[{
                "doc_id": "doc_current",
                "department": "general",
                "access_level": "public",
                "is_current": True,
                "version": 2,
            }],
        )

        user = User(
            user_id="u1", roles=["employee"],
            department="general", access_level="public",
        )
        results = retrieve_with_access("documento", user, access_collection)

        if results["metadatas"][0]:
            for meta in results["metadatas"][0]:
                assert meta["is_current"] is True


class TestIngestDocumentWithAccess:
    def test_ingest_document_with_access_metadata(self, access_collection):
        """La metadata contiene department, access_level, allowed_roles, is_current y version."""
        ingest_document_with_access(
            doc={"id": "doc_test"},
            chunks=["Contenido de prueba para verificar metadata"],
            vectorstore=access_collection,
            department="ingenieria",
            access_level="internal",
            allowed_roles=["admin", "engineer"],
        )

        result = access_collection.get(where={"doc_id": "doc_test"})
        assert len(result["ids"]) == 1

        meta = result["metadatas"][0]
        assert meta["department"] == "ingenieria"
        assert meta["access_level"] == "internal"
        assert meta["allowed_roles"] == "admin,engineer"
        assert meta["is_current"] is True
        assert meta["version"] == 1
