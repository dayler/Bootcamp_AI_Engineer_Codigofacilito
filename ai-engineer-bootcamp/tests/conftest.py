"""Fixtures compartidos de pytest para los tests del sistema RAG."""

import hashlib
import uuid

import chromadb
import numpy as np
import pytest


@pytest.fixture
def chroma_collection():
    """Crea un cliente ChromaDB en memoria con una colección única por test."""
    client = chromadb.Client()
    collection_name = f"test_{uuid.uuid4().hex[:12]}"
    collection = client.create_collection(name=collection_name)
    yield collection
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


@pytest.fixture
def dummy_embed_fn():
    """Función de embedding determinista basada en SHA-256.

    Genera vectores de dimensión 384 normalizados a norma 1.
    El mismo texto siempre produce el mismo vector.
    """

    def _embed(text: str) -> list[float]:
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        # Expandir el hash a 384 dimensiones repitiendo y truncando
        repeated = hash_bytes * (384 // len(hash_bytes) + 1)
        raw = np.array(
            [b / 255.0 for b in repeated[:384]], dtype=np.float32
        )
        # Normalizar a norma 1
        norm = np.linalg.norm(raw)
        if norm > 0:
            raw = raw / norm
        return raw.tolist()

    return _embed


@pytest.fixture
def sample_docs():
    """Lista de 5 documentos de prueba con temas variados."""
    return [
        {
            "id": "doc_001",
            "content": "El presupuesto anual de la empresa es de 5 millones. "
                       "Se distribuye entre operaciones, I+D y marketing.",
        },
        {
            "id": "doc_002",
            "content": "La arquitectura del sistema usa microservicios con Kubernetes. "
                       "El API gateway maneja autenticación y rate limiting.",
        },
        {
            "id": "doc_003",
            "content": "La política de vacaciones permite 20 días hábiles al año. "
                       "Se pueden acumular hasta 5 días para el siguiente período.",
        },
        {
            "id": "doc_004",
            "content": "El plan de capacitación incluye cursos online y talleres presenciales. "
                       "Cada empleado tiene un presupuesto de formación anual.",
        },
        {
            "id": "doc_005",
            "content": "Las normas de seguridad informática requieren autenticación de dos factores. "
                       "Las contraseñas deben cambiarse cada 90 días.",
        },
    ]
