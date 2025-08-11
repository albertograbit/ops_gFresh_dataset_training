"""
Configuración y settings del Dataset Manager
Gestiona la carga de configuración desde archivos YAML y variables de entorno
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class DeploymentConfig:
    """Configuración de deployment"""
    default_days_back: int
    max_days_back: int
    batch_size: int

@dataclass
class ElasticsearchConfig:
    """Configuración de Elasticsearch (sin credenciales)"""
    timeout: int = 30
    max_retries: int = 3
    retry_on_timeout: bool = True
    max_results: int = 10000
    transaction_metrics: list = None
    index_pattern: str = "device_transaction-*"
    scroll_timeout: str = "5m"
    
    def __post_init__(self):
        if self.transaction_metrics is None:
            self.transaction_metrics = ["WSPROCESSCOMPLETED03", "GFRSUSERPROCESSCOM03", "WSREFERENCEIMAGEAC01"]

@dataclass
class DatabaseConfig:
    """Configuración de base de datos (sin credenciales)"""
    charset: str
    autocommit: bool
    connect_timeout: int

@dataclass
class PathsConfig:
    """Configuración de rutas"""
    base_output_dir: str
    base_log_dir: str
    temp_dir: str
    credentials_dir: str

@dataclass
class FilesConfig:
    """Configuración de archivos"""
    elastic_query_template: str
    elastic_credentials: str

@dataclass
class ImageProcessingConfig:
    """Configuración para procesamiento de imágenes"""
    max_workers: int
    timeout_seconds: int
    retry_attempts: int
    supported_formats: list
    max_file_size_mb: int

@dataclass
class ExcelConfig:
    """Configuración de Excel"""
    max_rows_per_sheet: int
    include_charts: bool
    freeze_header_row: bool

@dataclass
class ReportsConfig:
    """Configuración de informes"""
    enabled: bool
    excel_filename: str
    new_references_filename: str
    output_directory: str

@dataclass
class LoggingConfig:
    """Configuración de logging"""
    level: str
    format: str
    file_rotation: bool
    max_log_files: int
    max_log_size_mb: int

@dataclass
class EmailConfig:
    """Configuración de email"""
    enabled: bool
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    use_tls: bool
    from_address: str
    to_addresses: list

@dataclass
class ReferenceAnalysisConfig:
    """Configuración específica para análisis de referencias"""
    min_appearances: int = 3
    confidence_threshold: float = 0.7
    top_suggestions: int = 5

@dataclass  
class ImageReviewConfig:
    """Configuración para revisión de imágenes"""
    num_imagenes_revision: int = 5
    tipo_transacciones: str = "ambas"
    tipo_imagenes_bajar: str = "clase_y_similares" 
    clear_output_folder: bool = True
    carpeta_destino: str = "./output/revisar_imagenes"
    s3_bucket: str = "grabit-data"
    s3_region: str = "eu-west-2"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    download_timeout: int = 30
    max_retries: int = 3
    image_extensions: list = None
    
    def __post_init__(self):
        if self.image_extensions is None:
            self.image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]

@dataclass
class DatasetConfig:
    """Configuración para creación de datasets"""
    num_imagenes_dataset_default: int = 150
    devices_incluir_default: str = "todos"  # "todos", "devices_seleccionados"
    min_transacciones_cliente: int = 150  # Mínimo de transacciones de cliente para incluir en dataset
    color_columnas_especiales: str = "#87CEEB"  # Color azul claro para columnas especiales
    opciones_devices_incluir: list = None  # Lista de opciones válidas para devices_incluir
    opciones_si_no: list = None  # Lista de opciones sí/no para dropdowns
    num_imagenes_dataset_adicionales_default: int = 0  # Valor por defecto para imágenes adicionales
    
    def __post_init__(self):
        if self.opciones_devices_incluir is None:
            self.opciones_devices_incluir = ["todos", "devices_seleccionados"]
        if self.opciones_si_no is None:
            self.opciones_si_no = ["sí", "no"]

@dataclass
class AnalysisConfig:
    """Configuración de análisis"""
    min_confidence_threshold: float
    max_suggestions_per_reference: int
    similarity_threshold: float
    ignore_case: bool

class Settings:
    """
    Clase principal para gestionar toda la configuración del Dataset Manager
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la configuración
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._setup_configs()
        
    def _get_default_config_path(self) -> str:
        """Obtiene la ruta por defecto del archivo de configuración"""
        # El settings.yaml está en el directorio config del proyecto
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config" / "settings.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            logging.error(f"Archivo de configuración no encontrado: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error al parsear archivo YAML: {e}")
            raise
    
    def _setup_configs(self):
        """Configura todas las secciones de configuración"""
        self.deployment = DeploymentConfig(**self.config['deployment'])
        self.elasticsearch = ElasticsearchConfig(**self.config['elasticsearch'])
        self.database = DatabaseConfig(**self.config['database'])
        self.paths = PathsConfig(**self.config['paths'])
        self.files = FilesConfig(**self.config['files'])
        self.image_processing = ImageProcessingConfig(**self.config['image_processing'])
        self.excel = ExcelConfig(**self.config['excel'])
        self.reports = ReportsConfig(**self.config.get('reports', {
            'enabled': True,
            'excel_filename': 'dataset_analysis_{deployment_id}_{timestamp}.xlsx',
            'new_references_filename': 'new_references_{deployment_id}_{timestamp}.xlsx',
            'output_directory': './output/reports'
        }))
        self.logging_config = LoggingConfig(**self.config['logging'])
        self.email = EmailConfig(**self.config['email'])
        self.analysis = AnalysisConfig(**self.config['analysis'])
        self.reference_analysis = ReferenceAnalysisConfig(**self.config.get('reference_analysis', {}))
        self.image_review = ImageReviewConfig(**self.config.get('image_review', {}))
        self.dataset = DatasetConfig(**self.config.get('dataset', {}))
        
        # Cargar variables de entorno si existe .env
        self._load_environment_variables()
    
    def _load_environment_variables(self):
        """Carga variables de entorno desde archivo .env si existe"""
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            
    def get_elasticsearch_credentials(self) -> Dict[str, Any]:
        """
        Carga las credenciales de Elasticsearch desde archivo seguro
        
        Returns:
            Diccionario con credenciales de Elasticsearch
        """
        try:
            credentials_path = self.get_absolute_path(self.files.elastic_credentials)
            with open(credentials_path, 'r', encoding='utf-8') as file:
                credentials = json.load(file)
            return credentials
        except FileNotFoundError:
            logging.error(f"Archivo de credenciales de Elasticsearch no encontrado: {credentials_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error al parsear credenciales de Elasticsearch: {e}")
            raise
            
    def get_database_credentials(self) -> Dict[str, Any]:
        """
        Obtiene las credenciales de base de datos desde variables de entorno
        
        Returns:
            Diccionario con credenciales de base de datos
        """
        required_vars = {
            'host': 'DB_PROD_RO_HOST',
            'user': 'DB_PROD_RO_USER', 
            'password': 'DB_PROD_RO_PASSWORD',
            'database': 'DB_PROD_RO_DATABASE',
            'port': 'DB_PROD_RO_PORT'
        }
        
        credentials = {}
        missing_vars = []
        
        for key, env_var in required_vars.items():
            value = os.getenv(env_var)
            if value is None:
                missing_vars.append(env_var)
            else:
                # Convertir puerto a entero si es necesario
                if key == 'port':
                    try:
                        value = int(value)
                    except ValueError:
                        logging.error(f"Puerto de base de datos inválido: {value}")
                        raise
                credentials[key] = value
        
        if missing_vars:
            raise ValueError(f"Variables de entorno faltantes para base de datos: {missing_vars}")
            
        return credentials
        
    def get_absolute_path(self, relative_path: str) -> str:
        """
        Convierte una ruta relativa en absoluta desde el directorio del proyecto
        
        Args:
            relative_path: Ruta relativa
            
        Returns:
            Ruta absoluta
        """
        if os.path.isabs(relative_path):
            return relative_path
        else:
            project_root = Path(__file__).parent.parent.parent
            return str(project_root / relative_path)
    
    def get_elasticsearch_config_path(self) -> str:
        """Obtiene la ruta completa al archivo de configuración de Elasticsearch"""
        return self.get_absolute_path(self.files.elastic_credentials)
    
    def get_query_template_path(self) -> str:
        """Obtiene la ruta completa al archivo de template de query de Elasticsearch"""
        return self.get_absolute_path(self.files.elastic_query_template)
    
    def get_output_directory(self, subdir: str = "") -> str:
        """
        Obtiene el directorio de salida completo para un subdirectorio específico
        
        Args:
            subdir: Subdirectorio (ej: 'images', 'reports')
            
        Returns:
            Ruta completa al directorio
        """
        if subdir:
            output_dir = f"{self.paths.base_output_dir}/{subdir}"
        else:
            output_dir = self.paths.base_output_dir
        
        output_dir = self.get_absolute_path(output_dir)
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def get_log_directory(self) -> str:
        """Obtiene el directorio de logs y lo crea si no existe"""
        log_dir = self.get_absolute_path(self.paths.base_log_dir)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
        
    def get_temp_directory(self) -> str:
        """Obtiene el directorio temporal y lo crea si no existe"""
        temp_dir = self.get_absolute_path(self.paths.temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Actualiza un valor de configuración dinámicamente
        
        Args:
            section: Sección de configuración
            key: Clave a actualizar
            value: Nuevo valor
        """
        if section in self.config:
            self.config[section][key] = value
            # Recargar la configuración específica
            if hasattr(self, section):
                section_config = getattr(self, section)
                setattr(section_config, key, value)
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Guarda la configuración actual en un archivo YAML
        
        Args:
            output_path: Ruta donde guardar la configuración
        """
        output_path = output_path or self.config_path
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            logging.info(f"Configuración guardada en: {output_path}")
        except Exception as e:
            logging.error(f"Error al guardar configuración: {e}")
            raise

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Obtener toda la configuración como diccionario
        
        Returns:
            Diccionario completo de configuración
        """
        return self.config.copy()
    
    def get_setting(self, key_path: str, default: Any = None) -> Any:
        """
        Obtener un valor de configuración usando notación de punto
        
        Args:
            key_path: Ruta de la clave usando notación de punto (ej: 'paths.base_output_dir')
            default: Valor por defecto si no se encuentra la clave
            
        Returns:
            Valor de configuración o default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
