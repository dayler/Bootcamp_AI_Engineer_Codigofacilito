"""Simulación de Hash-Based Diff para sincronización de documentos.

Demuestra el flujo completo:
1. Calcular hash SHA-256 del contenido de cada documento
2. Comparar con registro JSON (registry)
3. Clasificar: NUEVO, MODIFICADO, ELIMINADO, SIN CAMBIO
"""

import hashlib
import json
import tempfile
from pathlib import Path


def compute_hash(content: str) -> str:
    """Calcula SHA-256 de un string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_registry(path: Path) -> dict:
    """Carga el registry desde disco. Devuelve dict vacío si no existe."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_registry(registry: dict, path: Path) -> None:
    """Guarda el registry como JSON."""
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def sync(docs: list[dict], registry_path: Path) -> dict:
    """Ejecuta el hash-based diff y clasifica cada documento.

    Args:
        docs: Lista de dicts con keys "id" y "content".
        registry_path: Ruta al JSON de registry.

    Returns:
        Dict con listas de doc_ids por categoría.
    """
    registry = load_registry(registry_path)
    current_ids = {doc["id"] for doc in docs}

    result = {"new": [], "modified": [], "deleted": [], "unchanged": []}

    # Clasificar cada documento actual
    for doc in docs:
        doc_id = doc["id"]
        current_hash = compute_hash(doc["content"])
        print(current_hash)
        if doc_id not in registry:
            result["new"].append(doc_id)
            registry[doc_id] = current_hash
        elif registry[doc_id] != current_hash:
            result["modified"].append(doc_id)
            registry[doc_id] = current_hash
        else:
            result["unchanged"].append(doc_id)

    # Detectar eliminados
    for doc_id in list(registry.keys()):
        if doc_id not in current_ids:
            result["deleted"].append(doc_id)
            del registry[doc_id]

    save_registry(registry, registry_path)
    return result


def print_result(label: str, result: dict) -> None:
    """Imprime el resultado de una sincronización."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    for category, doc_ids in result.items():
        icon = {"new": "+", "modified": "~", "deleted": "-", "unchanged": "="}[category]
        status = category.upper()
        print(f"  [{icon}] {status:12s} → {len(doc_ids)} docs: {doc_ids}")


def main():
    print("=" * 50)
    print("  HASH-BASED DIFF — Simulación")
    print("=" * 50)

    registry_path = Path(tempfile.mktemp(suffix=".json"))

    # --- SYNC 1: Todo es nuevo ---
    docs_v1 = [
        {"id": "doc_001", "content": "Política de vacaciones: 20 días hábiles al año."},
        {"id": "doc_002", "content": "Arquitectura: microservicios con Kubernetes."},
        {"id": "doc_003", "content": "Presupuesto 2025: 5 millones de dólares."},
    ]

    r1 = sync(docs_v1, registry_path)
    print_result("SYNC 1 — Primera carga (todo nuevo)", r1)

    print(f"\n  Registry en disco:")
    reg = load_registry(registry_path)
    for doc_id, h in reg.items():
        print(f"    {doc_id} → {h[:16]}...")

    # --- SYNC 2: Un doc modificado, uno sin cambio, uno nuevo ---
    docs_v2 = [
        {"id": "doc_001", "content": "Política de vacaciones: 20 días hábiles al año."},  # sin cambio
        {"id": "doc_002", "content": "Arquitectura: migración a serverless con Lambda."},  # modificado
        {"id": "doc_003", "content": "Presupuesto 2025: 5 millones de dólares."},           # sin cambio
        {"id": "doc_004", "content": "Guía de onboarding para nuevos empleados."},          # nuevo
    ]

    r2 = sync(docs_v2, registry_path)
    print_result("SYNC 2 — Cambios parciales", r2)

    # --- SYNC 3: Un doc eliminado ---
    docs_v3 = [
        {"id": "doc_001", "content": "Política de vacaciones: 20 días hábiles al año."},
        {"id": "doc_002", "content": "Arquitectura: migración a serverless con Lambda."},
        {"id": "doc_004", "content": "Guía de onboarding para nuevos empleados."},
        # doc_003 eliminado
    ]

    r3 = sync(docs_v3, registry_path)
    print_result("SYNC 3 — doc_003 eliminado", r3)

    # --- SYNC 4: Sin cambios ---
    r4 = sync(docs_v3, registry_path)
    print_result("SYNC 4 — Sin cambios (todo unchanged)", r4)

    # Cleanup
    registry_path.unlink(missing_ok=True)

    print(f"\n{'=' * 50}")
    print("  Simulación completada")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
