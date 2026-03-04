#!/bin/bash
# Script para ejecutar los tests del sistema RAG en producción.

set -e

echo "========================================="
echo "  Instalando dependencias..."
echo "========================================="
pip install pytest chromadb numpy sentence-transformers groq

echo ""
echo "========================================="
echo "  Ejecutando tests..."
echo "========================================="
cd "$(dirname "$0")/.."
pytest tests/ -v --tb=short

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Todos los tests pasaron exitosamente"
else
    echo "  Algunos tests fallaron"
fi
echo "========================================="

exit $EXIT_CODE
