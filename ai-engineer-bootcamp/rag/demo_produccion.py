"""Script de demostración del sistema RAG en producción.

Demuestra:
- Sincronización de índices con detección de cambios
- Control de acceso basado en roles
- Caché semántico
- Versionamiento de documentos
"""

import os
import tempfile

import chromadb

from sentence_transformers import SentenceTransformer

from access_control import User, ingest_document_with_access, retrieve_with_access
from cache import SemanticCache, rag_query
from index_ops import ingest_new_version, sync_documents


from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
# --- Configuración de embeddings ---

_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def embed_fn(text: str) -> list[float]:
    """Genera un embedding para un texto usando sentence-transformers."""
    return _model.encode(text).tolist()


class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
    """Wrapper de sentence-transformers compatible con ChromaDB."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return _model.encode(input).tolist()


# --- Configuración del LLM ---

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)


def llm_fn(prompt: str) -> str:
    """Genera una respuesta usando Groq."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error LLM]: {e}"


def main():
    """Ejecuta la demostración completa del sistema RAG en producción."""

    print("=" * 60)
    print("  DEMO: RAG en Producción")
    print("=" * 60)

    # 1. Inicializar ChromaDB en memoria
    client = chromadb.Client()
    embedding_function = SentenceTransformerEmbeddingFunction()
    collection = client.create_collection(
        name="demo_produccion",
        embedding_function=embedding_function,
    )

    print("\n[1] ChromaDB inicializado en memoria")

    # 2. Crear usuarios de prueba
    admin_finanzas = User(
        user_id="admin_01",
        roles=["admin"],
        department="finanzas",
        access_level="confidential",
    )
    employee_ingenieria = User(
        user_id="emp_02",
        roles=["employee"],
        department="ingenieria",
        access_level="internal",
    )
    employee_rrhh = User(
        user_id="emp_03",
        roles=["employee"],
        department="rrhh",
        access_level="public",
    )

    print("[2] Usuarios creados:")
    print(f"    - Admin Finanzas (confidential): {admin_finanzas.user_id}")
    print(f"    - Employee Ingeniería (internal): {employee_ingenieria.user_id}")
    print(f"    - Employee RRHH (public): {employee_rrhh.user_id}")

    # 3. Documentos de prueba
    documents = [
        {
            "id": "doc_finanzas_01",
            "content": "El presupuesto anual de la empresa para 2025 es de 5 millones de dólares. "
                       "Se asignan 2 millones a operaciones, 1.5 millones a I+D y 1.5 millones a marketing.",
            "department": "finanzas",
            "access_level": "confidential",
        },
        {
            "id": "doc_ingenieria_01",
            "content": "La arquitectura del sistema usa microservicios con Kubernetes. "
                       "El API gateway maneja autenticación OAuth2 y rate limiting.",
            "department": "ingenieria",
            "access_level": "internal",
        },
        {
            "id": "doc_rrhh_01",
            "content": "La política de vacaciones permite 20 días hábiles al año. "
                       "Se pueden acumular hasta 5 días para el año siguiente.",
            "department": "rrhh",
            "access_level": "public",
        },
        {
            "id": "doc_general_01",
            "content": "La misión de la empresa es liderar la innovación tecnológica "
                       "con un enfoque en sostenibilidad y responsabilidad social.",
            "department": "general",
            "access_level": "public",
        },
        {
            "id": "doc_finanzas_02",
            "content": "Los salarios del equipo directivo se revisan trimestralmente. "
                       "El bono anual depende del cumplimiento de KPIs departamentales.",
            "department": "finanzas",
            "access_level": "confidential",
        },
    ]

    # 4. Ingestar documentos con control de acceso
    print("\n[3] Ingesta de documentos con control de acceso:")
    for doc in documents:
        ingest_document_with_access(
            doc=doc,
            chunks=[doc["content"]],
            vectorstore=collection,
            department=doc["department"],
            access_level=doc["access_level"],
            allowed_roles=["admin", "employee"],
        )
        print(f"    - {doc['id']} ({doc['department']}/{doc['access_level']})")

    # 5. Sync documents
    print("\n[4] Sincronización de documentos:")
    sync_docs = [{"id": doc["id"], "content": doc["content"]} for doc in documents]

    # Usar colección separada para sync (sin metadata de acceso)
    sync_collection = client.create_collection(
        name="sync_demo",
        embedding_function=embedding_function,
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        registry_path = tmp.name

    sync_documents(sync_docs, sync_collection, registry_path)

    # 6. Modificar un documento y re-sincronizar
    print("\n[5] Modificar documento y re-sincronizar:")
    sync_docs[1]["content"] = (
        "La arquitectura del sistema migró a serverless con AWS Lambda. "
        "Se usa API Gateway con autenticación JWT y rate limiting adaptativo."
    )
    sync_documents(sync_docs, sync_collection, registry_path)

    # 7. Queries con control de acceso
    print("\n[6] Queries con control de acceso:")
    query = "¿Cuál es el presupuesto de la empresa?"

    cache = SemanticCache(embed_fn=embed_fn, threshold=0.95, ttl_seconds=3600)

    print(f"\n    Query: '{query}'")
    for user, label in [
        (admin_finanzas, "Admin Finanzas"),
        (employee_ingenieria, "Employee Ingeniería"),
        (employee_rrhh, "Employee RRHH"),
    ]:
        results = retrieve_with_access(query, user, collection)
        n_results = len(results.get("documents", [[]])[0])
        print(f"\n    [{label}] Resultados: {n_results}")
        if results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                doc_id = results["metadatas"][0][i].get("doc_id", "?")
                print(f"      - {doc_id}: {doc_text[:80]}...")

    # 8. Demo de caché semántico
    print("\n[7] Demo de caché semántico:")
    query_cache = "¿Cuántos días de vacaciones tengo?"

    print(f"    Query 1: '{query_cache}'")
    response1 = rag_query(
        query=query_cache,
        user=employee_rrhh,
        vectorstore=collection,
        cache=cache,
        llm_fn=llm_fn,
        embed_fn=embed_fn,
    )
    print(f"    Respuesta: {response1[:100]}...")

    print(f"\n    Query 2 (misma pregunta): '{query_cache}'")
    response2 = rag_query(
        query=query_cache,
        user=employee_rrhh,
        vectorstore=collection,
        cache=cache,
        llm_fn=llm_fn,
        embed_fn=embed_fn,
    )
    print(f"    Respuesta (desde caché): {response2[:100]}...")

    # 9. Estadísticas de caché
    print("\n[8] Estadísticas de caché:")
    stats = cache.hit_rate()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # 10. Demo de versionamiento
    print("\n[9] Demo de versionamiento:")
    version_collection = client.create_collection(
        name="version_demo",
        embedding_function=embedding_function,
    )

    doc_version = {"id": "doc_version_test"}
    v1_chunks = ["Versión 1 del documento de prueba. Contenido original."]
    v1 = ingest_new_version(doc_version, version_collection, v1_chunks)
    print(f"    Ingesta versión {v1}")

    v2_chunks = ["Versión 2 del documento de prueba. Contenido actualizado con mejoras."]
    v2 = ingest_new_version(doc_version, version_collection, v2_chunks)
    print(f"    Ingesta versión {v2}")

    # Verificar que solo la versión actual está marcada como current
    all_docs = version_collection.get(where={"doc_id": "doc_version_test"})
    print(f"\n    Total chunks en store: {len(all_docs['ids'])}")
    for i, (doc_id, meta) in enumerate(zip(all_docs["ids"], all_docs["metadatas"])):
        print(f"      - {doc_id}: version={meta['version']}, is_current={meta['is_current']}")

    current_docs = version_collection.get(
        where={"$and": [{"doc_id": "doc_version_test"}, {"is_current": True}]}
    )
    print(f"\n    Chunks con is_current=True: {len(current_docs['ids'])}")
    for doc_text in current_docs["documents"]:
        print(f"      - {doc_text[:80]}...")

    # Cleanup
    try:
        os.unlink(registry_path)
    except OSError:
        pass

    print("\n" + "=" * 60)
    print("  Demo completada exitosamente")
    print("=" * 60)


if __name__ == "__main__":
    main()
