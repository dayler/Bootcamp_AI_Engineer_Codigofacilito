"""
main_rag_avanzado.py — Clase 6: RAG Avanzado (paso a paso)

Demostración interactiva de las 4 técnicas de RAG Avanzado:
  1. Hybrid Search (BM25 + Vector)
  2. Multi-Query Expansion
  3. Re-Ranking con Cross-Encoder
  4. Context Compression

Ejecutar:  python3 main_rag_avanzado.py
"""

import os
import shutil
import time

from dotenv import load_dotenv
from openai import OpenAI

from rag.ingestion import load_directory, chunk_by_paragraphs
from rag.vectorstore import create_vectorstore, index_chunks, search as vector_search
from rag.retrieval import (
    BM25Index,
    HybridRetriever,
    generate_multi_queries,
    multi_query_search,
    rerank,
    compress_context,
    compress_with_reranker,
    advanced_rag_query,
    call_llm,
)

load_dotenv()

# --- Colores ANSI ---
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
DIM = "\033[2m"
WHITE = "\033[97m"

# --- Configuración del LLM (Groq) ---
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

CHROMA_DIR = "./chroma_db_avanzado"


# =========================================================================
# Helpers de presentación
# =========================================================================

def header(paso: str, titulo: str, color: str = CYAN) -> None:
    print(f"\n{color}{BOLD}{'=' * 80}")
    print(f"  {paso}: {titulo}")
    print(f"{'=' * 80}{RESET}")


def subheader(texto: str, color: str = YELLOW) -> None:
    print(f"\n  {color}{BOLD}{texto}{RESET}")


def info(texto: str) -> None:
    print(f"  {DIM}{texto}{RESET}")


def result_line(idx: int, score: float, source: str, preview: str,
                color: str = YELLOW) -> None:
    print(f"  {color}{idx}. [{score:.4f}] [{source}]{RESET}")
    print(f"     {DIM}{preview}{RESET}")


def pause(msg: str = "Presiona Enter para continuar...") -> None:
    input(f"\n  {DIM}>>> {msg}{RESET}")


def rag_generate(context: str, question: str) -> str:
    """Genera respuesta con el LLM usando contexto y pregunta."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responde basándote ÚNICAMENTE en el contexto proporcionado. "
                    "Si no encuentras la respuesta, di 'No tengo información suficiente'. "
                    "Responde en español."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def print_chunks(results, color=YELLOW, max_preview=150) -> None:
    for i, r in enumerate(results, 1):
        source = os.path.basename(r.metadata.get("source", "?"))
        preview = r.content[:max_preview].replace("\n", " ")
        if len(r.content) > max_preview:
            preview += "..."
        result_line(i, r.score, source, preview, color)


def print_bm25_chunks(results, color=MAGENTA, max_preview=150) -> None:
    for i, (chunk, score) in enumerate(results, 1):
        source = os.path.basename(chunk.metadata.get("source", "?"))
        preview = chunk.content[:max_preview].replace("\n", " ")
        if len(chunk.content) > max_preview:
            preview += "..."
        result_line(i, score, source, preview, color)


# =========================================================================
# MAIN
# =========================================================================

def main() -> None:

    # =====================================================================
    # INTRO
    # =====================================================================
    print(f"\n{CYAN}{BOLD}{'=' * 80}")
    print(f"   CLASE 6: RAG AVANZADO — Demostración paso a paso")
    print(f"{'=' * 80}{RESET}")
    print(f"""
  Hoy vamos a explorar 4 técnicas que mejoran un pipeline RAG básico:

    {YELLOW}{BOLD}1. Hybrid Search{RESET}      — Combina BM25 (keywords) + Vector Search (semántico)
    {MAGENTA}{BOLD}2. Multi-Query{RESET}        — Reformula la pregunta para cubrir más vocabulario
    {BLUE}{BOLD}3. Re-Ranking{RESET}         — Un cross-encoder reordena por relevancia real
    {RED}{BOLD}4. Compress Context{RESET}   — Extrae solo las oraciones útiles del contexto

  {DIM}LLM:        {MODEL} (Groq)
  Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (local)
  Reranker:   cross-encoder/ms-marco-MiniLM-L-6-v2 (local){RESET}
""")

    if not os.environ.get("GROQ_API_KEY"):
        print(f"  {RED}{BOLD}ERROR: GROQ_API_KEY no configurada.{RESET}")
        print(f"  {DIM}export GROQ_API_KEY='gsk_...'{RESET}\n")
        return

    pause("Enter para comenzar...")

    # =====================================================================
    # PASO 1: Carga de documentos
    # =====================================================================
    header("PASO 1", "Carga de documentos", GREEN)

    docs_dir = "./data/docs"
    if not os.path.isdir(docs_dir):
        print(f"  {RED}No se encontró {docs_dir}. Asegúrate de tener los documentos.{RESET}")
        return

    documents = load_directory(docs_dir)
    print(f"\n  Documentos cargados: {BOLD}{len(documents)}{RESET}")
    for doc in documents:
        name = os.path.basename(doc.metadata["source"])
        words = len(doc.content.split())
        print(f"    - {name} ({words} palabras, {len(doc.content)} chars)")

    pause("Enter para hacer chunking...")

    # =====================================================================
    # PASO 2: Chunking
    # =====================================================================
    header("PASO 2", "Chunking por párrafos (max 800 chars)", MAGENTA)

    all_chunks = []
    for doc in documents:
        chunks = chunk_by_paragraphs(doc, max_chunk_size=800)
        all_chunks.extend(chunks)
        name = os.path.basename(doc.metadata["source"])
        print(f"    {MAGENTA}{name}: {BOLD}{len(chunks)} chunks{RESET}")

    print(f"\n  Total: {MAGENTA}{BOLD}{len(all_chunks)} chunks{RESET}")

    pause("Enter para indexar en ChromaDB...")

    # =====================================================================
    # PASO 3: Indexación en ChromaDB
    # =====================================================================
    header("PASO 3", "Indexación en ChromaDB (embeddings locales)", CYAN)

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    collection = create_vectorstore("rag_avanzado", persist_dir=CHROMA_DIR)
    indexed = index_chunks(collection, all_chunks)
    print(f"\n  {CYAN}{BOLD}{indexed} chunks indexados{RESET} en {CHROMA_DIR}")

    pause("Enter para ver RAG Básico (línea base)...")

    # =====================================================================
    # PASO 4: RAG Básico (línea base)
    # =====================================================================
    header("PASO 4", "RAG Básico — Solo Vector Search (línea base)", GREEN)

    question = "¿Cuál es el código de la estrategia de liberación para pedidos de compra en SAP?"

    print(f"\n  {WHITE}{BOLD}Pregunta:{RESET} {question}")
    info("Buscando con vector search (similitud semántica)...\n")

    t0 = time.time()
    vec_results = vector_search(collection, question, n_results=5)
    vec_time = time.time() - t0

    subheader(f"Chunks recuperados (vector search) — {vec_time:.2f}s:")
    print_chunks(vec_results)

    context = "\n\n---\n\n".join(r.content for r in vec_results)
    answer_basic = rag_generate(context, question)
    print(f"\n  {GREEN}{BOLD}Respuesta RAG Básico:{RESET}\n  {GREEN}{answer_basic}{RESET}")

    info(
        "Observa: la búsqueda semántica entiende el significado general, "
        "pero puede fallar con códigos exactos como 'ZRL01' o 'ZRL02'."
    )

    pause("Enter para ver BM25 (búsqueda por keywords)...")

    # =====================================================================
    # PASO 5: BM25 — Búsqueda por keywords
    # =====================================================================
    header("PASO 5", "BM25 — Búsqueda por keywords exactas", MAGENTA)

    print(f"\n  {WHITE}{BOLD}Misma pregunta:{RESET} {question}")
    info("Buscando con BM25 (coincidencia exacta de palabras)...\n")

    bm25_index = BM25Index(all_chunks)

    t0 = time.time()
    bm25_results = bm25_index.search(question, top_k=5)
    bm25_time = time.time() - t0

    subheader(f"Chunks recuperados (BM25) — {bm25_time:.3f}s:")
    print_bm25_chunks(bm25_results)

    info(
        "\nBM25 es excelente para términos exactos (códigos, nombres técnicos). "
        "Pero falla con sinónimos: si preguntas 'reglas de liberación' en vez de "
        "'estrategia de liberación', no lo encuentra."
    )

    pause("Enter para Hybrid Search (lo mejor de ambos mundos)...")

    # =====================================================================
    # PASO 6: Hybrid Search — BM25 + Vector
    # =====================================================================
    header("PASO 6", "Hybrid Search — BM25 + Vector combinados", YELLOW)

    print(f"\n  {WHITE}{BOLD}Pregunta:{RESET} {question}")
    print(f"""
  {DIM}Formula: score = alpha * vector_score + (1 - alpha) * bm25_score

  alpha = 0.5  → peso equilibrado entre semántico y keywords
  alpha = 0.0  → solo BM25 (keywords puras)
  alpha = 1.0  → solo vector (semántico puro){RESET}
""")

    hybrid = HybridRetriever(collection, all_chunks, alpha=0.5)

    t0 = time.time()
    hybrid_results = hybrid.search(question, top_k=5)
    hybrid_time = time.time() - t0

    subheader(f"Chunks recuperados (Hybrid, alpha=0.5) — {hybrid_time:.2f}s:")
    print_chunks(hybrid_results, YELLOW)

    context = "\n\n---\n\n".join(r.content for r in hybrid_results)
    answer_hybrid = rag_generate(context, question)
    print(f"\n  {GREEN}{BOLD}Respuesta Hybrid Search:{RESET}\n  {GREEN}{answer_hybrid}{RESET}")

    info(
        "Hybrid combina lo mejor: los términos exactos de BM25 con la "
        "comprensión semántica del vector search."
    )

    pause("Enter para ver el efecto de alpha...")

    # --- Comparativa de alphas ---
    subheader("Comparativa de alphas (misma query):")
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        h = HybridRetriever(collection, all_chunks, alpha=alpha)
        res = h.search(question, top_k=3)
        scores_str = " | ".join(f"{r.score:.3f}" for r in res)
        label = ""
        if alpha == 0.0:
            label = " (solo BM25)"
        elif alpha == 1.0:
            label = " (solo vector)"
        print(f"    alpha={alpha:.1f}{label:<16} → top-3 scores: [{scores_str}]")

    pause("Enter para Multi-Query Expansion...")

    # =====================================================================
    # PASO 7: Multi-Query — Reformulación de preguntas
    # =====================================================================
    header("PASO 7", "Multi-Query — Reformulación de preguntas con LLM", MAGENTA)

    question_mq = "¿Cuáles son las reglas de acceso en la empresa?"

    print(f"\n  {WHITE}{BOLD}Pregunta (ambigua):{RESET} {question_mq}")
    info(
        'Esta pregunta es ambigua: ¿acceso a edificios? ¿a sistemas? '
        '¿a la VPN? Multi-query genera reformulaciones para cubrir todas.\n'
    )

    # Primero: búsqueda simple
    subheader("A) Búsqueda simple (una sola query):")
    simple_results = vector_search(collection, question_mq, n_results=5)
    print_chunks(simple_results)

    # Ahora: multi-query
    subheader("B) Generando reformulaciones con el LLM...")
    t0 = time.time()
    queries = generate_multi_queries(question_mq, n=3)
    mq_time = time.time() - t0

    for i, q in enumerate(queries):
        label = "(original)" if i == 0 else f"(reformulación {i})"
        print(f"    {MAGENTA}{i + 1}. {q} {DIM}{label}{RESET}")

    print(f"    {DIM}Generación: {mq_time:.2f}s{RESET}")

    subheader("C) Búsqueda Multi-Query (acumula scores de todas las reformulaciones):")
    t0 = time.time()
    mq_results = multi_query_search(collection, question_mq, n_results=5)
    mq_search_time = time.time() - t0
    print_chunks(mq_results, MAGENTA)
    print(f"    {DIM}Búsqueda: {mq_search_time:.2f}s{RESET}")

    info(
        "\nMulti-query amplía la cobertura: al reformular la pregunta con vocabulario "
        "distinto, encuentra chunks que la query original no alcanzaba."
    )

    pause("Enter para Re-Ranking con Cross-Encoder...")

    # =====================================================================
    # PASO 8: Re-Ranking — Cross-Encoder
    # =====================================================================
    header("PASO 8", "Re-Ranking — Cross-Encoder reordena por relevancia", BLUE)

    question_rr = "¿Qué pasos debo seguir si pierdo mi laptop corporativa con información confidencial?"

    print(f"\n  {WHITE}{BOLD}Pregunta (compleja):{RESET} {question_rr}")
    info(
        "Esta pregunta toca múltiples documentos: FAQ (reportar pérdida), "
        "Seguridad (clasificación de info), y más.\n"
    )

    # Búsqueda inicial (vector)
    subheader("A) Vector search inicial (top 10 candidatos):")
    candidates = vector_search(collection, question_rr, n_results=10)
    print_chunks(candidates[:5])
    if len(candidates) > 5:
        info(f"    ... y {len(candidates) - 5} más")

    # Reranking
    subheader("B) Después de Re-Ranking con Cross-Encoder:")
    info(
        "El cross-encoder evalúa cada par (pregunta, chunk) de forma cruzada. "
        "Es más lento pero MUCHO más preciso que la similitud coseno.\n"
    )

    t0 = time.time()
    reranked = rerank(question_rr, candidates, top_k=5)
    rr_time = time.time() - t0

    print_chunks(reranked, BLUE)
    print(f"    {DIM}Re-ranking: {rr_time:.2f}s{RESET}")

    # Comparativa visual
    subheader("C) Comparativa de posiciones (antes → después):")
    candidate_ids = [r.chunk_id for r in candidates]
    for i, r in enumerate(reranked, 1):
        old_pos = candidate_ids.index(r.chunk_id) + 1 if r.chunk_id in candidate_ids else "?"
        source = os.path.basename(r.metadata.get("source", "?"))
        movement = ""
        if isinstance(old_pos, int):
            diff = old_pos - i
            if diff > 0:
                movement = f"{GREEN}+{diff} subió{RESET}"
            elif diff < 0:
                movement = f"{RED}{diff} bajó{RESET}"
            else:
                movement = f"{DIM}= igual{RESET}"
        print(f"    {BLUE}#{i}{RESET} (era #{old_pos}) [{source}] {movement}")

    context = "\n\n---\n\n".join(r.content for r in reranked)
    answer_rr = rag_generate(context, question_rr)
    print(f"\n  {GREEN}{BOLD}Respuesta con Re-Ranking:{RESET}\n  {GREEN}{answer_rr}{RESET}")

    pause("Enter para Compress Context...")

    # =====================================================================
    # PASO 9: Compress Context — Reducir ruido
    # =====================================================================
    header("PASO 9", "Compress Context — Extraer solo lo relevante", RED)

    question_cc = "¿Cuántos días de vacaciones me corresponden en mi segundo año?"

    print(f"\n  {WHITE}{BOLD}Pregunta:{RESET} {question_cc}")

    results_cc = vector_search(collection, question_cc, n_results=5)
    chunk_texts = [r.content for r in results_cc]

    # Mostrar contexto original
    total_chars = sum(len(t) for t in chunk_texts)
    subheader(f"A) Contexto original ({len(chunk_texts)} chunks, {total_chars} chars):")
    for i, text in enumerate(chunk_texts, 1):
        preview = text[:120].replace("\n", " ")
        print(f"    {RED}{i}. {DIM}({len(text)} chars){RESET} {preview}...")

    # Compresión con LLM
    subheader("B) Compresión con LLM (extrae solo oraciones relevantes):")
    t0 = time.time()
    compressed_llm = compress_context(question_cc, chunk_texts)
    compress_llm_time = time.time() - t0

    print(f"\n    {RED}{compressed_llm}{RESET}")
    print(f"\n    {DIM}Chars: {total_chars} → {len(compressed_llm)} "
          f"({100 - len(compressed_llm) * 100 // total_chars}% reducción) | "
          f"{compress_llm_time:.2f}s{RESET}")

    # Compresión con reranker
    subheader("C) Compresión con Reranker (sin llamada al LLM):")
    info("Divide en oraciones, las rankea con cross-encoder, toma las mejores.\n")

    t0 = time.time()
    compressed_rr = compress_with_reranker(question_cc, chunk_texts, top_sentences=5)
    compress_rr_time = time.time() - t0

    print(f"    {RED}{compressed_rr}{RESET}")
    print(f"\n    {DIM}Chars: {total_chars} → {len(compressed_rr)} "
          f"({100 - len(compressed_rr) * 100 // total_chars}% reducción) | "
          f"{compress_rr_time:.2f}s | Sin costo de LLM{RESET}")

    # Generar respuesta con contexto comprimido
    answer_compressed = rag_generate(compressed_llm, question_cc)
    print(f"\n  {GREEN}{BOLD}Respuesta con contexto comprimido:{RESET}\n  {GREEN}{answer_compressed}{RESET}")

    info(
        "\nLa compresión reduce tokens enviados al LLM = menor costo y respuestas "
        "más enfocadas. El reranker es gratis (local) pero el LLM es más preciso."
    )

    pause("Enter para el Pipeline Completo...")

    # =====================================================================
    # PASO 10: Pipeline Completo — Todo junto
    # =====================================================================
    header("PASO 10", "Pipeline Completo — Las 4 técnicas integradas", CYAN)

    question_full = (
        "Si un colaborador nuevo quiere trabajar remoto desde otro país "
        "y necesita acceso a los sistemas, ¿qué debe hacer?"
    )

    print(f"\n  {WHITE}{BOLD}Pregunta compleja:{RESET} {question_full}")
    print(f"""
  {DIM}El pipeline ejecuta en orden:
    1. Multi-query   → genera reformulaciones de la pregunta
    2. Hybrid search → BM25 + vector por cada reformulación
    3. Deduplicar    → elimina chunks repetidos
    4. Re-ranking    → cross-encoder reordena por relevancia
    5. Compress      → LLM extrae solo oraciones relevantes
    6. Generar       → LLM responde con contexto limpio{RESET}
""")

    t0 = time.time()
    answer_full = advanced_rag_query(collection, all_chunks, question_full)
    full_time = time.time() - t0

    print(f"\n  {GREEN}{BOLD}Respuesta del Pipeline Completo:{RESET}")
    print(f"  {GREEN}{answer_full}{RESET}")
    print(f"\n  {DIM}Tiempo total del pipeline: {full_time:.2f}s{RESET}")

    pause("Enter para la comparación final...")

    # =====================================================================
    # PASO 11: Comparación — Básico vs Avanzado
    # =====================================================================
    header("PASO 11", "Comparación lado a lado — Básico vs Avanzado", WHITE)

    question_cmp = (
        "¿Qué proceso debo seguir para comprar software nuevo y qué aprobaciones necesito?"
    )

    print(f"\n  {WHITE}{BOLD}Pregunta:{RESET} {question_cmp}")

    # --- RAG Básico ---
    subheader("A) RAG Básico (solo vector search):", GREEN)
    t0 = time.time()
    basic_results = vector_search(collection, question_cmp, n_results=5)
    basic_context = "\n\n---\n\n".join(r.content for r in basic_results)
    answer_basic_cmp = rag_generate(basic_context, question_cmp)
    basic_time = time.time() - t0

    print_chunks(basic_results[:3], GREEN)
    print(f"\n  {GREEN}{answer_basic_cmp}{RESET}")
    print(f"  {DIM}Tiempo: {basic_time:.2f}s{RESET}")

    # --- RAG Avanzado ---
    subheader("B) RAG Avanzado (pipeline completo):", RED)
    t0 = time.time()
    answer_adv_cmp = advanced_rag_query(collection, all_chunks, question_cmp)
    adv_time = time.time() - t0

    print(f"\n  {RED}{answer_adv_cmp}{RESET}")
    print(f"  {DIM}Tiempo: {adv_time:.2f}s{RESET}")

    # --- Tabla resumen ---
    subheader("C) Resumen:", WHITE)
    print(f"""
    {'Métrica':<25} {'RAG Básico':>15} {'RAG Avanzado':>15}
    {'-' * 25} {'-' * 15} {'-' * 15}
    {'Tiempo total':<25} {f'{basic_time:.2f}s':>15} {f'{adv_time:.2f}s':>15}
    {'Técnicas usadas':<25} {'1 (vector)':>15} {'4 (completo)':>15}
    {'Llamadas al LLM':<25} {'1':>15} {'3+':>15}
""")

    info(
        "RAG Avanzado es más lento y costoso, pero produce respuestas más precisas "
        "y completas, especialmente en preguntas complejas o ambiguas."
    )

    # =====================================================================
    # FIN
    # =====================================================================
    print(f"\n{CYAN}{BOLD}{'=' * 80}")
    print(f"  Pipeline RAG Avanzado completado")
    print(f"{'=' * 80}{RESET}")
    print(f"""
  {DIM}Resumen de técnicas demostradas:

  Paso 5: BM25            — Búsqueda por keywords (rápida, exacta)
  Paso 6: Hybrid Search   — BM25 + Vector combinados con alpha
  Paso 7: Multi-Query     — Reformulación de preguntas con LLM
  Paso 8: Re-Ranking      — Cross-encoder reordena por relevancia real
  Paso 9: Compress        — Reduce contexto para ahorrar tokens
  Paso 10: Pipeline       — Todo integrado en un solo flujo
  Paso 11: Comparación    — Básico vs Avanzado lado a lado{RESET}
""")

    # Limpiar ChromaDB
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        info(f"Base de datos temporal eliminada: {CHROMA_DIR}\n")


if __name__ == "__main__":
    main()
