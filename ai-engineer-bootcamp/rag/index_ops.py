"""Módulo de operaciones de índice para RAG en producción.

Provee sincronización de índices con hash-based diff, deduplicación
de chunks y versionamiento de documentos.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np


def compute_doc_hash(content: str) -> str:
    """Calcula el SHA-256 de un string y devuelve el hexdigest."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Genera un ID determinista para cada chunk.

    Formato: {doc_id}::chunk_{index}::{hash_parcial}
    donde hash_parcial son los primeros 8 caracteres del MD5 del texto.
    """
    hash_parcial = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}::chunk_{index}::{hash_parcial}"


def load_registry(path: Path) -> dict:
    """Carga un JSON desde disco. Si no existe, devuelve dict vacío."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_registry(registry: dict, path: Path) -> None:
    """Guarda el diccionario como JSON con indent=2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def sync_documents(
    docs: list[dict],
    vectorstore,
    registry_path: str,
) -> dict:
    """Sincroniza documentos con el vector store usando hash-based diff.

    Args:
        docs: Lista de documentos con keys "id" y "content".
        vectorstore: Colección de ChromaDB.
        registry_path: Ruta al archivo JSON de registry.

    Returns:
        Diccionario con contadores: new, modified, deleted, unchanged.
    """
    registry = load_registry(Path(registry_path))
    current_doc_ids = {doc["id"] for doc in docs}

    counters = {"new": 0, "modified": 0, "deleted": 0, "unchanged": 0}

    # Procesar cada documento actual
    for doc in docs:
        doc_id = doc["id"]
        content = doc["content"]
        current_hash = compute_doc_hash(content)

        if doc_id not in registry:
            # Documento nuevo
            chunk_id = generate_chunk_id(doc_id, 0, content)
            vectorstore.add(
                ids=[chunk_id],
                documents=[content],
                metadatas=[{"doc_id": doc_id, "is_current": True}],
            )
            registry[doc_id] = {
                "hash": current_hash,
                "chunk_ids": [chunk_id],
                "updated_at": time.time(),
            }
            counters["new"] += 1

        elif registry[doc_id]["hash"] != current_hash:
            # Documento modificado: eliminar chunks viejos y re-indexar
            old_chunk_ids = registry[doc_id].get("chunk_ids", [])
            if old_chunk_ids:
                try:
                    vectorstore.delete(ids=old_chunk_ids)
                except Exception:
                    pass

            chunk_id = generate_chunk_id(doc_id, 0, content)
            vectorstore.add(
                ids=[chunk_id],
                documents=[content],
                metadatas=[{"doc_id": doc_id, "is_current": True}],
            )
            registry[doc_id] = {
                "hash": current_hash,
                "chunk_ids": [chunk_id],
                "updated_at": time.time(),
            }
            counters["modified"] += 1

        else:
            # Sin cambios
            counters["unchanged"] += 1

    # Detectar documentos eliminados
    deleted_ids = set(registry.keys()) - current_doc_ids
    for doc_id in deleted_ids:
        old_chunk_ids = registry[doc_id].get("chunk_ids", [])
        if old_chunk_ids:
            try:
                vectorstore.delete(ids=old_chunk_ids)
            except Exception:
                pass
        del registry[doc_id]
        counters["deleted"] += 1

    save_registry(registry, Path(registry_path))

    print(
        f"Sync completado: {counters['new']} nuevos, "
        f"{counters['modified']} modificados, "
        f"{counters['deleted']} eliminados, "
        f"{counters['unchanged']} sin cambios"
    )

    return counters


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud coseno entre dos vectores."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def deduplicate_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    sim_threshold: float = 0.95,
) -> list[dict]:
    """Deduplica chunks por hash exacto y similitud semántica.

    Args:
        chunks: Lista de dicts con al menos key "text".
        embeddings: Lista de embeddings correspondientes a cada chunk.
        sim_threshold: Umbral de similitud coseno para dedup semántica.

    Returns:
        Lista de chunks únicos.
    """
    # Paso 1: Deduplicación exacta por hash MD5
    seen_hashes: set[str] = set()
    exact_unique: list[tuple[dict, list[float]]] = []

    for chunk, emb in zip(chunks, embeddings):
        text_hash = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            exact_unique.append((chunk, emb))

    # Paso 2: Deduplicación semántica por similitud coseno
    semantic_unique: list[dict] = []
    unique_embeddings: list[np.ndarray] = []

    for chunk, emb in exact_unique:
        emb_array = np.array(emb)
        is_duplicate = False

        for existing_emb in unique_embeddings:
            sim = _cosine_similarity(emb_array, existing_emb)
            if sim >= sim_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            semantic_unique.append(chunk)
            unique_embeddings.append(emb_array)

    return semantic_unique


def ingest_new_version(
    doc: dict,
    vectorstore,
    chunks: list[str],
) -> int:
    """Ingesta una nueva versión de un documento.

    Marca chunks de versiones anteriores como is_current=False e indexa
    los nuevos chunks con la versión incrementada.

    Args:
        doc: Diccionario con key "id".
        vectorstore: Colección de ChromaDB.
        chunks: Lista de strings (textos de los chunks).

    Returns:
        Número de la nueva versión.
    """
    doc_id = doc["id"]

    # Determinar la versión máxima existente
    max_version = 0
    try:
        existing = vectorstore.get(
            where={"doc_id": doc_id},
        )
        if existing and existing["ids"]:
            for meta in existing["metadatas"]:
                v = meta.get("version", 0)
                if v > max_version:
                    max_version = v

            # Marcar chunks anteriores como no actuales
            vectorstore.update(
                ids=existing["ids"],
                metadatas=[
                    {**meta, "is_current": False}
                    for meta in existing["metadatas"]
                ],
            )
    except Exception:
        pass

    new_version = max_version + 1

    # Indexar nuevos chunks
    new_ids = []
    new_documents = []
    new_metadatas = []

    for i, chunk_text in enumerate(chunks):
        chunk_id = generate_chunk_id(doc_id, i, chunk_text)
        # Agregar version al ID para evitar colisiones
        versioned_id = f"{chunk_id}::v{new_version}"
        new_ids.append(versioned_id)
        new_documents.append(chunk_text)
        new_metadatas.append({
            "doc_id": doc_id,
            "version": new_version,
            "is_current": True,
            "created_at": time.time(),
        })

    vectorstore.add(
        ids=new_ids,
        documents=new_documents,
        metadatas=new_metadatas,
    )

    return new_version
