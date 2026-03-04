#!/usr/bin/env python3
"""
scripts/compare_rag.py — Clase 6: Comparación RAG Básico vs RAG Avanzado

Demostración interactiva con pausas, scores visuales y análisis
de cómo cada técnica mejora progresivamente los resultados.

Ejecutar desde la raíz del proyecto:
    python3 scripts/compare_rag.py
"""

import os
import sys
import shutil
import time

# Asegurar que el proyecto raíz esté en el path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from rag.ingestion import load_directory, chunk_by_paragraphs, Chunk
from rag.vectorstore import (
    create_vectorstore,
    index_chunks,
    search as vector_search,
    SearchResult,
)
from rag.retrieval import (
    BM25Index,
    HybridRetriever,
    generate_multi_queries,
    multi_query_search,
    rerank,
    compress_context,
    advanced_rag_query,
    call_llm,
    reset_usage_tracker,
    get_usage,
)

# =========================================================================
# Colores ANSI
# =========================================================================
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"
BLUE = "\033[94m"
WHITE = "\033[97m"

# =========================================================================
# Pricing estimado
# =========================================================================
INPUT_COST_PER_1M = 0.05
OUTPUT_COST_PER_1M = 0.10


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_COST_PER_1M + \
           (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M


# =========================================================================
# 10 queries de prueba
# =========================================================================
QUERIES = [
    {
        "query": "¿Cuál es el código de la estrategia de liberación ZRL01 para solicitudes de pedido en SAP?",
        "type": "técnico",
    },
    {
        "query": "¿Qué transacción SAP se usa para verificación de factura MIRO?",
        "type": "técnico",
    },
    {
        "query": "¿Cuál es el código del permiso por matrimonio PER-MAT según la política RH-POL-003?",
        "type": "técnico",
    },
    {
        "query": "Mi computadora va muy lenta, ¿qué puedo hacer antes de llamar a soporte?",
        "type": "conversacional",
    },
    {
        "query": "¿Cómo protege la empresa la información personal de sus empleados?",
        "type": "conversacional",
    },
    {
        "query": "Acabo de entrar a trabajar como desarrollador, ¿qué herramientas necesito instalar?",
        "type": "conversacional",
    },
    {
        "query": "¿Cuáles son las reglas de acceso en la empresa?",
        "type": "ambigua",
    },
    {
        "query": "¿Cómo funciona el proceso de evaluación?",
        "type": "ambigua",
    },
    {
        "query": "Si pierdo mi laptop corporativa con información confidencial, ¿qué pasos debo seguir?",
        "type": "compleja",
    },
    {
        "query": "¿Cuáles son todos los días libres adicionales y beneficios especiales que ofrece TechCorp más allá de las vacaciones?",
        "type": "compleja",
    },
]


# =========================================================================
# Helpers de presentación
# =========================================================================

def pause(msg: str = "Presiona Enter para continuar...") -> None:
    input(f"\n  {DIM}>>> {msg}{RESET}")


def header(text: str, color: str = CYAN) -> None:
    print(f"\n{color}{BOLD}{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}{RESET}")


def subheader(text: str, color: str = YELLOW) -> None:
    print(f"\n  {color}{BOLD}{text}{RESET}")


def info(text: str) -> None:
    print(f"  {DIM}{text}{RESET}")


def bar(value: float, max_value: float, width: int = 30, color: str = GREEN) -> str:
    """Genera una barra ASCII proporcional."""
    if max_value <= 0:
        filled = 0
    else:
        filled = int(round(value / max_value * width))
    filled = max(0, min(filled, width))
    empty = width - filled
    return f"{color}{'█' * filled}{'░' * empty}{RESET}"


def score_bar(label: str, value: float, max_value: float,
              width: int = 25, color: str = GREEN) -> str:
    """Barra con etiqueta y valor numérico."""
    b = bar(value, max_value, width, color)
    return f"    {label:<28} {b} {value:.4f}"


def print_chunks_compact(results, color=YELLOW, limit=5) -> None:
    """Imprime chunks de forma compacta."""
    for i, r in enumerate(results[:limit], 1):
        source = os.path.basename(r.metadata.get("source", "?"))
        preview = r.content[:100].replace("\n", " ")
        print(f"    {color}{i}. [{r.score:.4f}] [{source}]{RESET} {DIM}{preview}...{RESET}")


def print_sources_overlap(results_a: list, results_b: list,
                          label_a: str, label_b: str) -> None:
    """Muestra el solapamiento de chunks entre dos métodos."""
    ids_a = {r.chunk_id for r in results_a}
    ids_b = {r.chunk_id for r in results_b}
    shared = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a
    print(f"    Compartidos:       {len(shared)} chunks")
    print(f"    Solo en {label_a:<12} {len(only_a)} chunks")
    print(f"    Solo en {label_b:<12} {len(only_b)} chunks {'← nuevos hallazgos' if only_b else ''}")


def improvement_pct(old: float, new: float) -> str:
    """Retorna string con porcentaje de mejora y flecha."""
    if old == 0:
        return f"{GREEN}  NEW{RESET}"
    pct = (new - old) / abs(old) * 100
    if pct > 0:
        return f"{GREEN}↑ +{pct:.0f}%{RESET}"
    elif pct < 0:
        return f"{RED}↓ {pct:.0f}%{RESET}"
    return f"{DIM}= 0%{RESET}"


def avg_score(results: list) -> float:
    """Promedio de scores de una lista de SearchResult."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


# =========================================================================
# Métodos RAG
# =========================================================================

def rag_basico(collection, query: str) -> tuple[str, list[SearchResult]]:
    results = vector_search(collection, query, n_results=5)
    context = "\n\n---\n\n".join(r.content for r in results)
    prompt = (
        "Responde la siguiente pregunta usando SOLO el contexto proporcionado.\n"
        'Si no puedes responder, di "No tengo suficiente información".\n\n'
        f"Contexto:\n{context}\n\nPregunta: {query}\n\nRespuesta:"
    )
    answer = call_llm(prompt)
    return answer, results


def rag_hibrido(
    collection, chunks: list[Chunk], query: str, alpha: float = 0.5
) -> tuple[str, list[SearchResult]]:
    hybrid = HybridRetriever(collection, chunks, alpha=alpha)
    results = hybrid.search(query, top_k=5)
    context = "\n\n---\n\n".join(r.content for r in results)
    prompt = (
        "Responde la siguiente pregunta usando SOLO el contexto proporcionado.\n"
        'Si no puedes responder, di "No tengo suficiente información".\n\n'
        f"Contexto:\n{context}\n\nPregunta: {query}\n\nRespuesta:"
    )
    answer = call_llm(prompt)
    return answer, results


def rag_multiquery_rerank(
    collection, query: str
) -> tuple[str, list[SearchResult]]:
    mq_results = multi_query_search(collection, query, n_results=10)
    reranked = rerank(query, mq_results, top_k=5)
    context = "\n\n---\n\n".join(r.content for r in reranked)
    prompt = (
        "Responde la siguiente pregunta usando SOLO el contexto proporcionado.\n"
        'Si no puedes responder, di "No tengo suficiente información".\n\n'
        f"Contexto:\n{context}\n\nPregunta: {query}\n\nRespuesta:"
    )
    answer = call_llm(prompt)
    return answer, reranked


def rag_avanzado_completo(
    collection, chunks: list[Chunk], query: str
) -> tuple[str, list]:
    answer = advanced_rag_query(collection, chunks, query)
    return answer, []


# =========================================================================
# Ejecución con tracking
# =========================================================================

def run_method(method_name, method_fn, *args) -> dict:
    reset_usage_tracker()
    start = time.time()
    try:
        answer, results = method_fn(*args)
    except Exception as e:
        answer = f"[ERROR: {e}]"
        results = []
    elapsed = time.time() - start
    usage = get_usage()

    return {
        "method": method_name,
        "answer": answer,
        "results": results,
        "latency_s": elapsed,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        "llm_calls": usage["calls"],
        "cost_usd": estimate_cost(usage["input_tokens"], usage["output_tokens"]),
        "avg_score": avg_score(results),
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    header("CLASE 6: COMPARACIÓN RAG BÁSICO vs RAG AVANZADO")
    print(f"""
  Este script compara 4 métodos de RAG en 10 queries distintas,
  mostrando paso a paso cómo cada técnica mejora los resultados.

    {GREEN}1. RAG Básico{RESET}         — Solo vector search
    {MAGENTA}2. RAG Híbrido{RESET}        — BM25 + vector search combinados
    {BLUE}3. Multi-query + Rerank{RESET} — Reformulación + cross-encoder
    {RED}4. RAG Avanzado{RESET}        — Pipeline completo (4 técnicas)
""")

    if not os.environ.get("GROQ_API_KEY"):
        print(f"  {RED}{BOLD}ERROR: GROQ_API_KEY no configurada.{RESET}")
        print(f"  {DIM}export GROQ_API_KEY='gsk_...'{RESET}\n")
        sys.exit(1)

    pause("Enter para comenzar la preparación...")

    # =================================================================
    # PASO 1: Ingesta
    # =================================================================
    header("PASO 1: Carga de documentos", GREEN)

    docs_dir = os.path.join(PROJECT_ROOT, "data", "docs")
    if not os.path.isdir(docs_dir):
        print(f"  {RED}No se encontró {docs_dir}{RESET}")
        sys.exit(1)

    documents = load_directory(docs_dir)
    print(f"\n  Documentos cargados: {BOLD}{len(documents)}{RESET}")
    for doc in documents:
        name = os.path.basename(doc.metadata["source"])
        words = len(doc.content.split())
        print(f"    - {name} ({words} palabras)")

    pause("Enter para chunking...")

    # =================================================================
    # PASO 2: Chunking
    # =================================================================
    header("PASO 2: Chunking por párrafos (max 800 chars)", MAGENTA)

    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_by_paragraphs(doc, max_chunk_size=800)
        all_chunks.extend(chunks)
        name = os.path.basename(doc.metadata["source"])
        print(f"    {name}: {BOLD}{len(chunks)} chunks{RESET}")

    print(f"\n  Total: {MAGENTA}{BOLD}{len(all_chunks)} chunks{RESET}")

    pause("Enter para indexar en ChromaDB...")

    # =================================================================
    # PASO 3: Indexación
    # =================================================================
    header("PASO 3: Indexación en ChromaDB", BLUE)

    chroma_dir = os.path.join(PROJECT_ROOT, "chroma_db_clase6")
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    collection = create_vectorstore("clase6_advanced_rag", persist_dir=chroma_dir)
    indexed = index_chunks(collection, all_chunks)
    print(f"\n  {BOLD}{indexed} chunks indexados{RESET}")

    pause("Enter para precargar el cross-encoder...")

    # =================================================================
    # PASO 4: Precarga cross-encoder
    # =================================================================
    header("PASO 4: Precarga del cross-encoder", YELLOW)
    try:
        from rag.retrieval import _get_cross_encoder
        _get_cross_encoder()
        print(f"  {GREEN}Cross-encoder cargado.{RESET}")
    except Exception as e:
        print(f"  {RED}Error: {e}{RESET}")
        sys.exit(1)

    pause("Enter para la demostración progresiva...")

    # =================================================================
    # PASO 5: Demostración progresiva (1 query, técnica por técnica)
    # =================================================================
    header("PASO 5: Demostración progresiva — cómo mejora cada técnica", WHITE)

    demo_query = "Si pierdo mi laptop corporativa con información confidencial, ¿qué pasos debo seguir?"

    print(f"\n  {WHITE}{BOLD}Query de demostración:{RESET}")
    print(f"  {CYAN}\"{demo_query}\"{RESET}")
    info(
        "Esta pregunta es compleja: toca FAQ (reportar pérdida), Seguridad\n"
        "  (clasificación de info, cifrado), y procedimientos de incidentes."
    )

    pause("Enter para ver vector search básico...")

    # --- Etapa A: Solo vector search ---
    subheader("ETAPA A: Solo Vector Search", GREEN)
    info("Busca por similitud semántica con el embedding de la query.\n")

    t0 = time.time()
    vec_results = vector_search(collection, demo_query, n_results=5)
    vec_time = time.time() - t0

    print_chunks_compact(vec_results, GREEN)
    print(f"\n    {DIM}Tiempo: {vec_time:.2f}s | Score promedio: {avg_score(vec_results):.4f}{RESET}")

    pause("Enter para agregar BM25 (Hybrid Search)...")

    # --- Etapa B: + Hybrid Search ---
    subheader("ETAPA B: + Hybrid Search (BM25 + Vector)", MAGENTA)
    info(
        "BM25 busca coincidencias exactas de palabras clave.\n"
        "  Hybrid combina ambos: alpha*vector + (1-alpha)*bm25\n"
    )

    hybrid = HybridRetriever(collection, all_chunks, alpha=0.5)
    t0 = time.time()
    hybrid_results = hybrid.search(demo_query, top_k=5)
    hybrid_time = time.time() - t0

    print_chunks_compact(hybrid_results, MAGENTA)
    print(f"\n    {DIM}Tiempo: {hybrid_time:.2f}s | Score promedio: {avg_score(hybrid_results):.4f}{RESET}")

    subheader("Análisis de solapamiento (Vector vs Hybrid):")
    print_sources_overlap(vec_results, hybrid_results, "Vector", "Hybrid")

    vec_ids = {r.chunk_id for r in vec_results}
    new_in_hybrid = [r for r in hybrid_results if r.chunk_id not in vec_ids]
    if new_in_hybrid:
        print(f"\n    {MAGENTA}{BOLD}Chunks nuevos encontrados por Hybrid:{RESET}")
        for r in new_in_hybrid:
            src = os.path.basename(r.metadata.get("source", "?"))
            print(f"      [{src}] {DIM}{r.content[:90].replace(chr(10), ' ')}...{RESET}")

    pause("Enter para agregar Multi-Query + Re-Ranking...")

    # --- Etapa C: + Multi-Query + Rerank ---
    subheader("ETAPA C: + Multi-Query + Re-Ranking", BLUE)
    info("Multi-Query reformula la pregunta para ampliar cobertura.")
    info("Rerank usa un cross-encoder para reordenar por relevancia real.\n")

    t0 = time.time()
    mq_queries = generate_multi_queries(demo_query, n=3)
    mq_time_gen = time.time() - t0

    print(f"    Reformulaciones generadas ({mq_time_gen:.2f}s):")
    for j, q in enumerate(mq_queries):
        tag = "(original)" if j == 0 else f"(reformulación {j})"
        print(f"      {BLUE}{j+1}. {q} {DIM}{tag}{RESET}")

    t0 = time.time()
    mq_results = multi_query_search(collection, demo_query, n_results=10)
    mq_reranked = rerank(demo_query, mq_results, top_k=5)
    mq_rr_time = time.time() - t0

    print(f"\n    Candidatos pre-rerank: {len(mq_results)} → post-rerank: {len(mq_reranked)}")
    print_chunks_compact(mq_reranked, BLUE)
    print(f"\n    {DIM}Tiempo: {mq_rr_time:.2f}s | Score promedio (rerank): {avg_score(mq_reranked):.4f}{RESET}")

    subheader("Análisis de solapamiento (Hybrid vs Multi+Rerank):")
    print_sources_overlap(hybrid_results, mq_reranked, "Hybrid", "MQ+Rerank")

    pause("Enter para ver el pipeline completo con compresión...")

    # --- Etapa D: Pipeline completo ---
    subheader("ETAPA D: Pipeline Completo (+ Compress Context)", RED)
    info(
        "Agrega compresión de contexto: el LLM extrae solo las oraciones\n"
        "  relevantes antes de generar la respuesta final.\n"
    )

    t0 = time.time()
    answer_full = advanced_rag_query(collection, all_chunks, demo_query)
    full_time = time.time() - t0

    print(f"\n  {GREEN}{BOLD}Respuesta del pipeline completo:{RESET}")
    print(f"  {GREEN}{answer_full}{RESET}")
    print(f"\n    {DIM}Tiempo total: {full_time:.2f}s{RESET}")

    # --- Barras de score promedio ---
    subheader("Comparativa de scores por etapa:")

    max_s = max(avg_score(vec_results), avg_score(hybrid_results),
                avg_score(mq_reranked), 0.01)
    print(score_bar("A) Vector Search", avg_score(vec_results), max_s, 25, GREEN))
    print(score_bar("B) Hybrid Search", avg_score(hybrid_results), max_s, 25, MAGENTA))
    print(score_bar("C) Multi-Query + Rerank", avg_score(mq_reranked), max_s, 25, BLUE))
    print(f"    {'D) Avanzado Completo':<28} {RED}{'█' * 25}{RESET} (incluye compresión)")

    # --- Barras de latencia ---
    subheader("Comparativa de latencia por etapa:")
    max_t = max(vec_time, hybrid_time, mq_rr_time + mq_time_gen, full_time, 0.01)
    print(score_bar("A) Vector Search", vec_time, max_t, 25, GREEN))
    print(score_bar("B) Hybrid Search", hybrid_time, max_t, 25, MAGENTA))
    print(score_bar("C) Multi-Query + Rerank", mq_rr_time + mq_time_gen, max_t, 25, BLUE))
    print(score_bar("D) Avanzado Completo", full_time, max_t, 25, RED))

    info(
        "\nMás técnicas = mayor latencia y costo, pero mejores resultados.\n"
        "  El trade-off depende del caso de uso."
    )

    pause("Enter para ejecutar las 10 queries con los 4 métodos...")

    # =================================================================
    # PASO 6: Ejecución completa (10 queries × 4 métodos)
    # =================================================================
    header("PASO 6: Ejecución completa — 10 queries × 4 métodos", CYAN)

    METHOD_COLORS = {
        "RAG Básico": GREEN,
        "RAG Híbrido (α=0.5)": MAGENTA,
        "Multi-query + Rerank": BLUE,
        "RAG Avanzado Completo": RED,
    }

    all_results = []

    for i, q_info in enumerate(QUERIES, 1):
        query = q_info["query"]
        qtype = q_info["type"]

        header(f"Query {i}/{len(QUERIES)} [{qtype}]", CYAN)
        print(f"\n  {WHITE}{BOLD}\"{query}\"{RESET}")

        method_results = []

        # a) RAG Básico
        print(f"\n    {GREEN}→ RAG Básico...{RESET}", end="", flush=True)
        r = run_method("RAG Básico", rag_basico, collection, query)
        method_results.append(r)
        print(f" {r['latency_s']:.1f}s")

        # b) RAG Híbrido
        print(f"    {MAGENTA}→ RAG Híbrido (α=0.5)...{RESET}", end="", flush=True)
        r = run_method("RAG Híbrido (α=0.5)", rag_hibrido, collection, all_chunks, query, 0.5)
        method_results.append(r)
        print(f" {r['latency_s']:.1f}s")

        # c) Multi-query + Rerank
        print(f"    {BLUE}→ Multi-query + Rerank...{RESET}", end="", flush=True)
        r = run_method("Multi-query + Rerank", rag_multiquery_rerank, collection, query)
        method_results.append(r)
        print(f" {r['latency_s']:.1f}s")

        # d) RAG Avanzado
        print(f"    {RED}→ RAG Avanzado Completo...{RESET}", end="", flush=True)
        r = run_method("RAG Avanzado Completo", rag_avanzado_completo, collection, all_chunks, query)
        method_results.append(r)
        print(f" {r['latency_s']:.1f}s")

        # --- Scores por método ---
        subheader("Scores de relevancia (top-5 promedio):")
        basico_avg = method_results[0]["avg_score"]
        max_avg = max((mr["avg_score"] for mr in method_results), default=0.01) or 0.01

        for mr in method_results:
            m = mr["method"]
            sc = mr["avg_score"]
            c = METHOD_COLORS.get(m, CYAN)
            imp = improvement_pct(basico_avg, sc) if m != "RAG Básico" else ""
            b = bar(sc, max_avg, 25, c)
            print(f"    {c}{m:<28}{RESET} {b} {sc:.4f} {imp}")

        # --- Chunks encontrados ---
        has_results = [(mr["method"], mr["results"]) for mr in method_results if mr["results"]]
        if len(has_results) >= 2:
            subheader("Solapamiento de chunks (Básico vs Híbrido):")
            basico_res = method_results[0]["results"]
            hibrido_res = method_results[1]["results"]
            mq_res = method_results[2]["results"]
            print_sources_overlap(basico_res, hibrido_res, "Básico", "Híbrido")
            if mq_res:
                subheader("Solapamiento de chunks (Básico vs MQ+Rerank):")
                print_sources_overlap(basico_res, mq_res, "Básico", "MQ+Rerank")

        # --- Latencia ---
        subheader("Latencia:")
        max_lat = max((mr["latency_s"] for mr in method_results), default=0.01) or 0.01
        for mr in method_results:
            m = mr["method"]
            c = METHOD_COLORS.get(m, CYAN)
            b = bar(mr["latency_s"], max_lat, 25, c)
            print(f"    {c}{m:<28}{RESET} {b} {mr['latency_s']:.2f}s "
                  f"({mr['llm_calls']} calls, {mr['total_tokens']} tok)")

        # --- Respuestas ---
        subheader("Respuestas:")
        for mr in method_results:
            m = mr["method"]
            c = METHOD_COLORS.get(m, CYAN)
            ans = mr["answer"][:250].replace("\n", " ")
            if len(mr["answer"]) > 250:
                ans += "..."
            print(f"\n    {c}{BOLD}[{m}]{RESET}")
            print(f"    {ans}")

        all_results.append({
            "query_idx": i,
            "query": query,
            "query_type": qtype,
            "method_results": method_results,
        })

        if i < len(QUERIES):
            pause(f"Enter para la siguiente query ({i+1}/{len(QUERIES)})...")

    pause("Enter para ver la tabla comparativa final...")

    # =================================================================
    # PASO 7: Tabla comparativa final
    # =================================================================
    header("TABLA COMPARATIVA FINAL", WHITE)

    # --- Tabla detallada ---
    print(f"\n  {'Q':<4} {'Tipo':<15} {'Método':<28} "
          f"{'Latencia':>8} {'Tokens':>7} {'Calls':>5} {'Score':>7} {'Costo':>10}")
    print(f"  {'─'*4} {'─'*15} {'─'*28} {'─'*8} {'─'*7} {'─'*5} {'─'*7} {'─'*10}")

    for entry in all_results:
        q_idx = entry["query_idx"]
        qtype = entry["query_type"]
        for mr in entry["method_results"]:
            m = mr["method"]
            c = METHOD_COLORS.get(m, RESET)
            print(
                f"  {q_idx:<4} {qtype:<15} {c}{m:<28}{RESET} "
                f"{mr['latency_s']:>7.2f}s {mr['total_tokens']:>7} "
                f"{mr['llm_calls']:>5} {mr['avg_score']:>7.4f} "
                f"${mr['cost_usd']:>9.6f}"
            )
        print()

    # --- Resumen por método ---
    subheader("Resumen por método (promedios):")
    method_agg: dict[str, dict] = {}
    for entry in all_results:
        for mr in entry["method_results"]:
            m = mr["method"]
            if m not in method_agg:
                method_agg[m] = {"lat": [], "tok": [], "score": [], "cost": [], "calls": []}
            method_agg[m]["lat"].append(mr["latency_s"])
            method_agg[m]["tok"].append(mr["total_tokens"])
            method_agg[m]["score"].append(mr["avg_score"])
            method_agg[m]["cost"].append(mr["cost_usd"])
            method_agg[m]["calls"].append(mr["llm_calls"])

    print(f"\n  {'Método':<28} {'Lat.':>7} {'Tokens':>7} {'Score':>7} "
          f"{'Calls':>5} {'Costo tot.':>10}")
    print(f"  {'─'*28} {'─'*7} {'─'*7} {'─'*7} {'─'*5} {'─'*10}")

    basico_avg_sc = 0
    for m in ["RAG Básico", "RAG Híbrido (α=0.5)", "Multi-query + Rerank", "RAG Avanzado Completo"]:
        if m not in method_agg:
            continue
        d = method_agg[m]
        n = len(d["lat"])
        avg_lat = sum(d["lat"]) / n
        avg_tok = sum(d["tok"]) / n
        avg_sc = sum(d["score"]) / n
        avg_calls = sum(d["calls"]) / n
        total_cost = sum(d["cost"])
        c = METHOD_COLORS.get(m, RESET)

        if m == "RAG Básico":
            basico_avg_sc = avg_sc

        imp = ""
        if m != "RAG Básico" and basico_avg_sc > 0:
            imp = improvement_pct(basico_avg_sc, avg_sc)

        print(
            f"  {c}{m:<28}{RESET} {avg_lat:>6.2f}s {avg_tok:>7.0f} "
            f"{avg_sc:>7.4f} {avg_calls:>5.1f} ${total_cost:>9.6f} {imp}"
        )

    # --- Barras visuales de score promedio ---
    subheader("Score promedio por método:")
    all_avg_scores = {}
    for m in ["RAG Básico", "RAG Híbrido (α=0.5)", "Multi-query + Rerank", "RAG Avanzado Completo"]:
        if m in method_agg:
            all_avg_scores[m] = sum(method_agg[m]["score"]) / len(method_agg[m]["score"])

    max_avg_s = max(all_avg_scores.values(), default=0.01) or 0.01
    for m, s in all_avg_scores.items():
        c = METHOD_COLORS.get(m, CYAN)
        b = bar(s, max_avg_s, 30, c)
        print(f"    {c}{m:<28}{RESET} {b} {s:.4f}")

    # --- Barras de latencia promedio ---
    subheader("Latencia promedio por método:")
    all_avg_lat = {}
    for m in ["RAG Básico", "RAG Híbrido (α=0.5)", "Multi-query + Rerank", "RAG Avanzado Completo"]:
        if m in method_agg:
            all_avg_lat[m] = sum(method_agg[m]["lat"]) / len(method_agg[m]["lat"])

    max_avg_l = max(all_avg_lat.values(), default=0.01) or 0.01
    for m, l in all_avg_lat.items():
        c = METHOD_COLORS.get(m, CYAN)
        b = bar(l, max_avg_l, 30, c)
        print(f"    {c}{m:<28}{RESET} {b} {l:.2f}s")

    # --- Score por tipo de query ---
    subheader("Score promedio por TIPO de query:")
    type_method_scores: dict[str, dict[str, list]] = {}
    for entry in all_results:
        qt = entry["query_type"]
        if qt not in type_method_scores:
            type_method_scores[qt] = {}
        for mr in entry["method_results"]:
            m = mr["method"]
            if m not in type_method_scores[qt]:
                type_method_scores[qt][m] = []
            type_method_scores[qt][m].append(mr["avg_score"])

    for qt in ["técnico", "conversacional", "ambigua", "compleja"]:
        if qt not in type_method_scores:
            continue
        print(f"\n    {WHITE}{BOLD}[{qt}]{RESET}")
        ms = type_method_scores[qt]
        max_s = max(
            (sum(v)/len(v) for v in ms.values()),
            default=0.01
        ) or 0.01
        for m in ["RAG Básico", "RAG Híbrido (α=0.5)", "Multi-query + Rerank", "RAG Avanzado Completo"]:
            if m not in ms:
                continue
            s = sum(ms[m]) / len(ms[m])
            c = METHOD_COLORS.get(m, CYAN)
            b = bar(s, max_s, 20, c)
            print(f"      {c}{m:<28}{RESET} {b} {s:.4f}")

    pause("Enter para el experimento de alpha...")

    # =================================================================
    # PASO 8: Experimento con alpha
    # =================================================================
    header("PASO 8: Experimento con alpha (0.3, 0.5, 0.7)", MAGENTA)

    print(f"""
  {DIM}alpha = peso del vector search en la combinación híbrida:
    0.3 = más peso a BM25 (keywords exactas)
    0.5 = equilibrado
    0.7 = más peso a vector (semántico){RESET}
""")

    alpha_values = [0.3, 0.5, 0.7]
    alpha_results = []

    for i, q_info in enumerate(QUERIES, 1):
        query = q_info["query"]
        qtype = q_info["type"]
        print(f"  [{i}/{len(QUERIES)}] ({qtype}) {query[:55]}...", end="", flush=True)

        alpha_runs = []
        for alpha in alpha_values:
            reset_usage_tracker()
            start = time.time()
            try:
                answer, results = rag_hibrido(collection, all_chunks, query, alpha=alpha)
            except Exception as e:
                answer = f"[ERROR: {e}]"
                results = []
            elapsed = time.time() - start
            usage = get_usage()

            alpha_runs.append({
                "alpha": alpha,
                "latency_s": elapsed,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                "cost_usd": estimate_cost(usage["input_tokens"], usage["output_tokens"]),
                "answer": answer,
                "avg_score": avg_score(results),
                "results": results,
            })

        # Mini-barra inline
        scores_str = "  ".join(
            f"α={r['alpha']}: {r['avg_score']:.3f}" for r in alpha_runs
        )
        print(f"  {DIM}{scores_str}{RESET}")

        alpha_results.append({
            "query_idx": i,
            "query": query,
            "query_type": qtype,
            "alpha_runs": alpha_runs,
        })

    # --- Tabla de alpha ---
    subheader("Resultados del experimento de alpha:")

    print(f"\n  {'Q':<4} {'Tipo':<15}", end="")
    for a in alpha_values:
        print(f" {'α='+str(a):>10}", end="")
    print(f"  {'Mejor':>7}")
    print(f"  {'─'*4} {'─'*15}", end="")
    for _ in alpha_values:
        print(f" {'─'*10}", end="")
    print(f"  {'─'*7}")

    for entry in alpha_results:
        q_idx = entry["query_idx"]
        qt = entry["query_type"]
        runs = entry["alpha_runs"]
        print(f"  {q_idx:<4} {qt:<15}", end="")
        best_score = max(r["avg_score"] for r in runs)
        for r in runs:
            s = r["avg_score"]
            marker = f" {BOLD}*{RESET}" if s == best_score and s > 0 else "  "
            print(f" {s:>8.4f}{marker}", end="")
        best_alpha = max(runs, key=lambda r: r["avg_score"])["alpha"]
        print(f"  {MAGENTA}α={best_alpha}{RESET}")

    # --- Resumen: mejor alpha por tipo ---
    subheader("Mejor alpha por tipo de query:")
    type_alpha: dict[str, dict[float, list]] = {}
    for entry in alpha_results:
        qt = entry["query_type"]
        if qt not in type_alpha:
            type_alpha[qt] = {}
        for r in entry["alpha_runs"]:
            a = r["alpha"]
            if a not in type_alpha[qt]:
                type_alpha[qt][a] = []
            type_alpha[qt][a].append(r["avg_score"])

    for qt in ["técnico", "conversacional", "ambigua", "compleja"]:
        if qt not in type_alpha:
            continue
        avgs = {a: sum(v)/len(v) for a, v in type_alpha[qt].items()}
        best_a = max(avgs, key=avgs.get)
        max_s = max(avgs.values()) or 0.01
        print(f"\n    {WHITE}{BOLD}[{qt}]{RESET}")
        for a in alpha_values:
            s = avgs.get(a, 0)
            marker = f" {MAGENTA}{BOLD}← mejor{RESET}" if a == best_a else ""
            b = bar(s, max_s, 20, MAGENTA)
            print(f"      α={a}  {b} {s:.4f}{marker}")

    pause("Enter para el resumen final...")

    # =================================================================
    # PASO 9: Resumen final
    # =================================================================
    header("RESUMEN FINAL", GREEN)

    total_queries = len(QUERIES)
    total_methods = 4

    print(f"""
  {BOLD}Configuración:{RESET}
    Documentos:    {len(documents)}
    Chunks:        {len(all_chunks)}
    Queries:       {total_queries}
    Métodos:       {total_methods}
    LLM:           {os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')} (Groq)
    Embeddings:    paraphrase-multilingual-MiniLM-L12-v2 (local)
    Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
""")

    # Veredicto por tipo
    subheader("Veredicto por tipo de query:")
    print(f"""
    {YELLOW}{BOLD}Técnicas (códigos exactos):{RESET}
      Hybrid Search brilla — BM25 atrapa los códigos que vector search pierde.
      alpha bajo (0.3) suele ser mejor: más peso a keywords.

    {YELLOW}{BOLD}Conversacionales (sinónimos):{RESET}
      Vector Search ya es bueno — entiende significado.
      Hybrid con alpha alto (0.7) puede dar un boost adicional.

    {YELLOW}{BOLD}Ambiguas:{RESET}
      Multi-Query marca la diferencia — las reformulaciones cubren
      las múltiples interpretaciones de la pregunta.

    {YELLOW}{BOLD}Complejas (multi-chunk):{RESET}
      El pipeline completo (Avanzado) gana — necesita multi-query para
      encontrar chunks dispersos, rerank para ordenarlos, y compresión
      para darle al LLM solo lo que importa.
""")

    subheader("Cuándo usar cada técnica:")
    print(f"""
    {GREEN}RAG Básico{RESET}         → Consultas simples, baja latencia, bajo costo
    {MAGENTA}Hybrid Search{RESET}      → Cuando hay vocabulario técnico mezclado con lenguaje natural
    {BLUE}Multi-Query+Rerank{RESET} → Preguntas ambiguas o que requieren alta precisión
    {RED}Pipeline Completo{RESET}   → Preguntas complejas donde la calidad importa más que la velocidad
""")

    print(f"  {GREEN}{BOLD}Comparación completada exitosamente.{RESET}")
    print(f"  {DIM}Base de datos vectorial: {chroma_dir}{RESET}\n")


if __name__ == "__main__":
    main()
