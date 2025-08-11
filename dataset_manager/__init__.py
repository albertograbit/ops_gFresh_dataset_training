"""
Dataset Manager Package

Una herramienta completa para gestionar y facilitar la creación y actualización 
de datasets utilizados en el entrenamiento de modelos de clasificación de productos para retail.

Módulos:
- core: Funcionalidades principales del procesamiento de datos
- data_extractor: Extracción de datos desde Elasticsearch y BD relacional
- reference_analyzer: Análisis de referencias y detección de clases nuevas
- image_downloader: Descarga y organización de imágenes para entrenamiento
- report_generator: Generación de informes y archivos Excel
- config: Configuración y parámetros del sistema
"""

__version__ = "1.0.0"
__author__ = "Alberto Gómez"

try:
    from .core.dataset_processor import DatasetProcessor
    from .config.settings import Settings
    
    __all__ = ['DatasetProcessor', 'Settings']
except ImportError:
    # En caso de que falten dependencias, permitir que el paquete se importe
    # pero sin las clases principales
    __all__ = []
