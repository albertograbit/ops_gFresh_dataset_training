"""
Script principal para ejecutar el Dataset Manager
Punto de entrada principal para la herramienta
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al path para imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Función principal que ejecuta el CLI del Dataset Manager"""
    try:
        from dataset_manager.cli import main as cli_main
        return cli_main()
    except ImportError as e:
        print(f"[ERROR] Error importando Dataset Manager: {e}")
        print("Asegúrese de que todas las dependencias estén instaladas:")
        print("pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"[ERROR] Error ejecutando Dataset Manager: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
