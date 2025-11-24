#!/bin/bash
# Script de ejemplo para ejecutar el pipeline LLM Judge
# Asegúrate de configurar .env antes de ejecutar

echo "=================================="
echo "PIPELINE LLM JUDGE - EJEMPLO DE USO"
echo "=================================="

# Configurar variables
ENTITIES_JSON="examples/entities_example.json"
DOCS_DIR="examples/documents"
OUTPUT_CSV="outputs/llm_judgments_example.csv"
RULES_FILE="guias-anotacion.json"

echo ""
echo "Configuración:"
echo "  - Entidades: $ENTITIES_JSON"
echo "  - Documentos: $DOCS_DIR"
echo "  - Salida: $OUTPUT_CSV"
echo "  - Reglas: $RULES_FILE"
echo ""

# Verificar que existe .env
if [ ! -f ".env" ]; then
    echo "ERROR: No se encontró el archivo .env"
    echo ""
    echo "Por favor, ejecuta:"
    echo "  cp .env.example .env"
    echo ""
    echo "Y luego edita .env con tus credenciales reales."
    exit 1
fi

echo "✓ Archivo .env encontrado"
echo ""

# Ejecutar el pipeline
echo "Ejecutando pipeline..."
echo ""

python src/pipeline-nuevos-textos/llm_entity_judge.py \
    --entities "$ENTITIES_JSON" \
    --docs "$DOCS_DIR" \
    --rules-file "$RULES_FILE" \
    --output "$OUTPUT_CSV" \
    --left-window 80 \
    --right-window 80

# Verificar resultado
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Pipeline completado exitosamente"
    echo "=================================="
    echo ""
    echo "Resultados guardados en: $OUTPUT_CSV"
    echo ""
    
    # Mostrar primeras líneas del CSV
    if [ -f "$OUTPUT_CSV" ]; then
        echo "Primeras líneas del resultado:"
        echo "------------------------------"
        head -n 3 "$OUTPUT_CSV"
        echo "------------------------------"
        echo ""
        echo "Para ver el archivo completo:"
        echo "  cat $OUTPUT_CSV"
        echo "  # o"
        echo "  open $OUTPUT_CSV  # en macOS"
    fi
else
    echo ""
    echo "=================================="
    echo "✗ Error al ejecutar el pipeline"
    echo "=================================="
    exit 1
fi
