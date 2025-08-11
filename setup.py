"""
Script de instalación y configuración inicial del Dataset Manager
Automatiza la configuración básica del entorno
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header():
    """Imprime el header del instalador"""
    print("[LAUNCH] Dataset Manager - Instalador y Configurador")
    print("=" * 50)
    print("Configurando herramienta de gestión de datasets para retail")
    print()

def check_python_version():
    """Verifica la versión de Python"""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("[ERROR] Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        return False
    
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def create_virtual_environment():
    """Crea un entorno virtual para el proyecto"""
    print("\n🏠 Configurando entorno virtual...")
    
    venv_path = Path("venv")
    
    # Verificar si ya existe
    if venv_path.exists():
        response = input("[FOLDER] El entorno virtual ya existe. ¿Recrear? (y/N): ")
        if response.lower() == 'y':
            print("   Eliminando entorno virtual existente...")
            try:
                shutil.rmtree(venv_path)
            except Exception as e:
                print(f"   [ERROR] Error eliminando entorno virtual: {e}")
                return False
        else:
            print("   [OK] Usando entorno virtual existente")
            return True
    
    try:
        # Crear entorno virtual
        print("   Creando entorno virtual...")
        result = subprocess.run([
            sys.executable, "-m", "venv", "venv"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   [ERROR] Error creando entorno virtual: {result.stderr}")
            return False
        
        print("   [OK] Entorno virtual creado en ./venv")
        return True
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

def get_venv_python():
    """Obtiene la ruta al Python del entorno virtual"""
    if os.name == 'nt':  # Windows
        return Path("venv") / "Scripts" / "python.exe"
    else:  # Unix/Linux/Mac
        return Path("venv") / "bin" / "python"

def get_venv_pip():
    """Obtiene la ruta al pip del entorno virtual"""
    if os.name == 'nt':  # Windows
        return Path("venv") / "Scripts" / "pip.exe"
    else:  # Unix/Linux/Mac
        return Path("venv") / "bin" / "pip"

def install_dependencies():
    """Instala las dependencias de requirements.txt en el entorno virtual"""
    print("\n[PACKAGE] Instalando dependencias en el entorno virtual...")
    
    venv_python = get_venv_python()
    venv_pip = get_venv_pip()
    
    if not venv_python.exists():
        print("[ERROR] Error: Entorno virtual no encontrado")
        return False
    
    try:
        # Verificar que pip funcione en el entorno virtual
        result = subprocess.run([
            str(venv_pip), "--version"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] Error: pip no funciona en el entorno virtual")
            return False
        
        print(f"   Usando pip: {result.stdout.strip()}")
        
        # Actualizar pip primero
        print("   Actualizando pip...")
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True)
        
        # Instalar dependencias
        print("   Instalando dependencias desde requirements.txt...")
        result = subprocess.run([
            str(venv_pip), "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   [OK] Dependencias instaladas correctamente en el entorno virtual")
            return True
        else:
            print("   [ERROR] Error instalando dependencias:")
            print(f"   {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("[ERROR] Error: archivo requirements.txt no encontrado")
        return False
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

def setup_environment_file():
    """Configura el archivo .env"""
    print("\n[CONFIG] Configurando archivo de entorno...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if not env_example.exists():
        print("[WARNING] Archivo .env.example no encontrado")
        return False
    
    if env_file.exists():
        response = input("[DOC] El archivo .env ya existe. ¿Sobrescribir? (y/N): ")
        if response.lower() != 'y':
            print("   Manteniendo archivo .env existente")
            return True
    
    try:
        shutil.copy(env_example, env_file)
        print("[OK] Archivo .env creado desde .env.example")
        print("   [WARNING] Recuerde configurar las variables de entorno en .env")
        return True
    except Exception as e:
        print(f"[ERROR] Error creando .env: {e}")
        return False

def create_directories():
    """Crea los directorios necesarios"""
    print("\n[FOLDER] Creando directorios...")
    
    directories = [
        "logs",
        "output",
        "output/images",
        "output/reports",
        "temp"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   [OK] {directory}")
        except Exception as e:
            print(f"   [ERROR] Error creando {directory}: {e}")
    
    return True

def verify_config_files():
    """Verifica que los archivos de configuración existan"""
    print("\n[SEARCH] Verificando archivos de configuración...")
    
    config_files = [
        "dataset_manager/config/config.yaml",
        "config/credentials/credentials_elastic_prod.json"
    ]
    
    all_good = True
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            print(f"   [OK] {config_file}")
        else:
            print(f"   [ERROR] {config_file} - No encontrado")
            all_good = False
    
    return all_good

def test_import():
    """Prueba que el paquete se pueda importar en el entorno virtual"""
    print("\n[TEST] Probando importación del paquete en el entorno virtual...")
    
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print("[ERROR] Error: Entorno virtual no encontrado")
        return False
    
    try:
        # Crear script temporal para probar la importación
        test_script = '''
import sys
from pathlib import Path

# Agregar directorio actual al path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from dataset_manager.config.settings import Settings
    print("[OK] Settings importado correctamente")
    
    # Probar carga de configuración
    try:
        settings = Settings()
        print("[OK] Configuración cargada correctamente")
    except Exception as e:
        print(f"[WARNING] Error cargando configuración: {e}")
        print("(Esto puede ser normal si faltan variables de entorno)")
    
    print("SUCCESS")
    
except ImportError as e:
    print(f"[ERROR] Error importando: {e}")
    print("FAILED")
'''
        
        # Escribir script temporal
        test_file = Path("test_import.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_script)
        
        try:
            # Ejecutar en el entorno virtual
            result = subprocess.run([
                str(venv_python), "test_import.py"
            ], capture_output=True, text=True, cwd=Path.cwd(), encoding='utf-8')
            
            # Limpiar archivo temporal
            test_file.unlink()
            
            if "SUCCESS" in result.stdout:
                # Mostrar solo las líneas de estado
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('SUCCESS'):
                        print(f"   {line}")
                return True
            else:
                print("   [ERROR] Error en importación:")
                # Reemplazar caracteres problemáticos
                stdout_clean = result.stdout.replace('[ERROR]', 'X').replace('[OK]', 'OK')
                print(f"   {stdout_clean}")
                if result.stderr:
                    stderr_clean = result.stderr.replace('[ERROR]', 'X').replace('[OK]', 'OK')
                    print(f"   {stderr_clean}")
                return False
                
        except Exception as e:
            if test_file.exists():
                test_file.unlink()
            print(f"   [ERROR] Error ejecutando prueba: {e}")
            return False
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return False

def show_next_steps():
    """Muestra los próximos pasos"""
    print("\n[TARGET] Próximos pasos:")
    print("1. Activar el entorno virtual:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    print("2. Editar .env con sus credenciales de base de datos")
    print("3. Verificar credenciales de Elasticsearch en config/credentials/")
    print("4. Probar la instalación:")
    print("   python main.py status")
    print("5. Ejecutar primer análisis:")
    print("   python main.py analyze 122 --days-back 7")
    print()
    print("[IDEA] Tip: También puede ejecutar directamente con el Python del entorno:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\python.exe main.py status")
    else:
        print("   venv/bin/python main.py status")
    print()
    print("📚 Para más información, consulte README.md")

def main():
    """Función principal del instalador"""
    print_header()
    
    # Lista de verificaciones
    checks = [
        ("Versión de Python", check_python_version),
        ("Entorno virtual", create_virtual_environment),
        ("Dependencias", install_dependencies),
        ("Archivo de entorno", setup_environment_file),
        ("Directorios", create_directories),
        ("Archivos de configuración", verify_config_files),
        ("Importación del paquete", test_import)
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        if not check_func():
            failed_checks.append(check_name)
    
    print("\n" + "=" * 50)
    
    if failed_checks:
        print("[WARNING] Instalación completada con advertencias")
        print("Verificaciones fallidas:")
        for check in failed_checks:
            print(f"   • {check}")
        print()
        print("Por favor, revise los errores antes de usar la herramienta.")
    else:
        print("🎉 ¡Instalación completada exitosamente!")
        print("Dataset Manager está listo para usar.")
    
    show_next_steps()
    
    return 0 if not failed_checks else 1

if __name__ == '__main__':
    sys.exit(main())
