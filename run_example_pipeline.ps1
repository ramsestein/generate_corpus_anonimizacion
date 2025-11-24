# Script de ejemplo para ejecutar el pipeline LLM Judge en PowerShell
# Asegúrate de configurar .env antes de ejecutar

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "PIPELINE LLM JUDGE - EJEMPLO DE USO" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Configurar variables
$ENTITIES_JSON = "examples/entities_example.json"
$DOCS_DIR = "examples/documents"
$OUTPUT_CSV = "outputs/llm_judgments_example.csv"
$RULES_FILE = "guias-anotacion.json"

Write-Host ""
Write-Host "Configuración:"
Write-Host "  - Entidades: $ENTITIES_JSON"
Write-Host "  - Documentos: $DOCS_DIR"
Write-Host "  - Salida: $OUTPUT_CSV"
Write-Host "  - Reglas: $RULES_FILE"
Write-Host ""

# Verificar que existe .env
if (-not (Test-Path ".env")) {
    Write-Host "ERROR: No se encontró el archivo .env" -ForegroundColor Red
    Write-Host ""
    Write-Host "Por favor, ejecuta:"
    Write-Host "  Copy-Item .env.example .env"
    Write-Host ""
    Write-Host "Y luego edita .env con tus credenciales reales."
    exit 1
}

Write-Host "✓ Archivo .env encontrado" -ForegroundColor Green
Write-Host ""

# Ejecutar el pipeline
Write-Host "Ejecutando pipeline..." -ForegroundColor Yellow
Write-Host ""

& .\.venv\Scripts\python.exe src\pipeline-nuevos-textos\llm_entity_judge.py `
    --entities $ENTITIES_JSON `
    --docs $DOCS_DIR `
    --rules-file $RULES_FILE `
    --output $OUTPUT_CSV `
    --left-window 80 `
    --right-window 80

# Verificar resultado
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Green
    Write-Host "✓ Pipeline completado exitosamente" -ForegroundColor Green
    Write-Host "==================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Resultados guardados en: $OUTPUT_CSV" -ForegroundColor Cyan
    Write-Host ""
    
    # Mostrar primeras líneas del CSV
    if (Test-Path $OUTPUT_CSV) {
        Write-Host "Primeras líneas del resultado:" -ForegroundColor Yellow
        Write-Host "------------------------------"
        Get-Content $OUTPUT_CSV -Head 3
        Write-Host "------------------------------"
        Write-Host ""
        Write-Host "Para ver el archivo completo:"
        Write-Host "  Get-Content $OUTPUT_CSV"
        Write-Host "  # o"
        Write-Host "  notepad $OUTPUT_CSV"
    }
} else {
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Red
    Write-Host "✗ Error al ejecutar el pipeline" -ForegroundColor Red
    Write-Host "==================================" -ForegroundColor Red
    exit 1
}
