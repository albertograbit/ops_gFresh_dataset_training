#!/bin/bash
# Script para activar el entorno virtual y ejecutar comandos del Dataset Manager

echo "🚀 Dataset Manager - Activador de Entorno"
echo "=========================================="

if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Error: Entorno virtual no encontrado"
    echo "   Ejecute primero: python setup.py"
    exit 1
fi

echo "✅ Activando entorno virtual..."
source venv/bin/activate

echo ""
echo "🎯 Entorno virtual activado. Comandos disponibles:"
echo "   python main.py status               - Verificar estado"
echo "   python main.py analyze 122          - Análisis básico"
echo "   python main.py process 122          - Procesamiento completo"
echo "   python main.py --help              - Ver ayuda completa"
echo ""
echo "Para salir del entorno virtual, escriba: deactivate"
echo ""

# Ejecutar comando si se proporciona como parámetro
if [ $# -eq 0 ]; then
    bash
else
    python main.py "$@"
fi
