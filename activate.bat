@echo off
REM Script para activar el entorno virtual y ejecutar comandos del Dataset Manager

echo 🚀 Dataset Manager - Activador de Entorno
echo ==========================================

if not exist "venv\Scripts\activate.bat" (
    echo ❌ Error: Entorno virtual no encontrado
    echo    Ejecute primero: python setup.py
    pause
    exit /b 1
)

echo ✅ Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo 🎯 Entorno virtual activado. Comandos disponibles:
echo    python main.py status               - Verificar estado
echo    python main.py analyze 122          - Análisis básico
echo    python main.py process 122          - Procesamiento completo
echo    python main.py --help              - Ver ayuda completa
echo.
echo Para salir del entorno virtual, escriba: deactivate
echo.

REM Ejecutar comando si se proporciona como parámetro
if "%1"=="" (
    cmd /k
) else (
    python main.py %*
)
