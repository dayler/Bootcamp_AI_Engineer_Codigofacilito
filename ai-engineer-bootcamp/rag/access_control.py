"""Módulo de control de acceso basado en roles (RBAC) para RAG.

Implementa filtros de acceso por departamento y nivel de confidencialidad
que se aplican como metadata filters en el vector store.
"""

from dataclasses import dataclass, field

from index_ops import generate_chunk_id


@dataclass
class User:
    """Representa un usuario con roles y permisos."""
    user_id: str
    roles: list[str] = field(default_factory=list)
    department: str = "general"
    access_level: str = "public"


LEVEL_HIERARCHY: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
}


def build_access_filter(user: User) -> dict:
    """Construye un filtro compatible con Chroma basado en los permisos del usuario.

    El filtro combina:
    - Condición de departamento: docs del departamento del usuario O "general"
    - Condición de nivel: docs con access_level igual o menor al del usuario

    Returns:
        Diccionario de filtro compatible con ChromaDB.
    """
    user_level = LEVEL_HIERARCHY.get(user.access_level, 0)
    allowed_levels = [
        level for level, rank in LEVEL_HIERARCHY.items() if rank <= user_level
    ]

    department_filter = {
        "$or": [
            {"department": user.department},
            {"department": "general"},
        ]
    }

    level_filter = {"access_level": {"$in": allowed_levels}}

    return {"$and": [department_filter, level_filter]}


def retrieve_with_access(
    query: str,
    user: User,
    vectorstore,
    top_k: int = 5,
) -> dict:
    """Ejecuta una query con filtros de control de acceso.

    Args:
        query: Texto de la consulta.
        user: Usuario que realiza la consulta.
        vectorstore: Colección de ChromaDB.
        top_k: Número máximo de resultados.

    Returns:
        Resultados de la query filtrados por acceso.
    """
    access_filter = build_access_filter(user)

    combined_filter = {
        "$and": [
            access_filter,
            {"is_current": True},
        ]
    }

    results = vectorstore.query(
        query_texts=[query],
        n_results=top_k,
        where=combined_filter,
    )

    return results


def ingest_document_with_access(
    doc: dict,
    chunks: list[str],
    vectorstore,
    department: str,
    access_level: str,
    allowed_roles: list[str] = None,
) -> list[str]:
    """Indexa chunks de un documento con metadata de control de acceso.

    Args:
        doc: Diccionario con key "id".
        chunks: Lista de textos de los chunks.
        vectorstore: Colección de ChromaDB.
        department: Departamento al que pertenece el documento.
        access_level: Nivel de acceso requerido.
        allowed_roles: Lista de roles que pueden acceder al documento.

    Returns:
        Lista de IDs de los chunks indexados.
    """
    roles_str = ",".join(allowed_roles) if allowed_roles else ""

    ids = []
    documents = []
    metadatas = []

    for i, chunk_text in enumerate(chunks):
        chunk_id = generate_chunk_id(doc["id"], i, chunk_text)
        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append({
            "doc_id": doc["id"],
            "department": department,
            "access_level": access_level,
            "allowed_roles": roles_str,
            "is_current": True,
            "version": 1,
        })

    vectorstore.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return ids
