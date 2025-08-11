"""
Configurador dinámico que ajusta rutas según el proceso activo
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataset_manager.process_manager import ProcessManager
from dataset_manager.config.settings import Settings

class ProcessAwareSettings:
    """Configuración que se adapta al proceso activo"""
    
    def __init__(self, config_path: str = None):
        """
        Inicializar configuración con awareness de procesos
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.logger = logging.getLogger(__name__)
        self.base_settings = Settings(config_path)
        self.process_manager = ProcessManager()
        
        # Obtener proceso activo para usarlo en DatasetProcessor
        self.active_process = self.process_manager.get_active_process()
        
    def get_settings(self) -> Dict[str, Any]:
        """
        Obtener configuración adaptada al proceso activo
        
        Returns:
            Diccionario de configuración con rutas ajustadas
        """
        # Obtener configuración base
        settings = self.base_settings.get_all_settings()
        
        # Obtener proceso activo
        active_process = self.process_manager.get_active_process()
        
        if active_process and 'process_dir' in active_process:
            process_dir = Path(active_process['process_dir'])
            
            # Actualizar rutas para usar el directorio del proceso
            settings['paths']['base_output_dir'] = str(process_dir)
            settings['paths']['reports_dir'] = str(process_dir / "reports")
            settings['paths']['images_dir'] = str(process_dir / "images")
            settings['paths']['data_dir'] = str(process_dir / "data")
            settings['paths']['logs_dir'] = str(process_dir / "logs")
            
            # Configuración específica para descarga de imágenes
            if 'image_review' not in settings:
                settings['image_review'] = {}
            
            settings['image_review']['output_folder'] = str(process_dir / "images" / "revisar_imagenes")
            
            self.logger.info(f"Configuración adaptada para proceso: {active_process['name']}")
            self.logger.info(f"Directorio base: {process_dir}")
        else:
            # Sin proceso activo, usar configuración por defecto
            base_output = Path(settings['paths']['base_output_dir'])
            settings['paths']['reports_dir'] = str(base_output / "reports")
            settings['paths']['images_dir'] = str(base_output / "images")
            settings['paths']['data_dir'] = str(base_output / "data")
            
            if 'image_review' not in settings:
                settings['image_review'] = {}
            settings['image_review']['output_folder'] = "./output/revisar_imagenes"
            
            self.logger.info("Usando configuración por defecto (sin proceso activo)")
        
        return settings
    
    def get_process_info(self) -> Optional[Dict[str, Any]]:
        """
        Obtener información del proceso activo
        
        Returns:
            Información del proceso activo o None
        """
        return self.process_manager.get_active_process()
    
    def get_output_path(self, subdir: str = "") -> Path:
        """
        Obtener ruta de salida para el proceso activo
        
        Args:
            subdir: Subdirectorio opcional
            
        Returns:
            Path de salida
        """
        active_process = self.process_manager.get_active_process()
        
        if active_process and 'process_dir' in active_process:
            base_path = Path(active_process['process_dir'])
        else:
            base_path = Path(self.base_settings.get_setting('paths.base_output_dir', './output'))
        
        return base_path / subdir if subdir else base_path

    def get_log_directory(self) -> str:
        """Obtiene el directorio de logs del proceso activo"""
        active_process = self.process_manager.get_active_process()
        
        if active_process and 'process_dir' in active_process:
            log_dir = Path(active_process['process_dir']) / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            return str(log_dir)
        else:
            return self.base_settings.get_log_directory()
    
    def get_output_directory(self, subdir: str = "") -> str:
        """Obtiene el directorio de salida del proceso activo"""
        return str(self.get_output_path(subdir))
    
    def get_temp_directory(self) -> str:
        """Obtiene el directorio temporal del proceso activo"""
        active_process = self.process_manager.get_active_process()
        
        if active_process and 'process_dir' in active_process:
            temp_dir = Path(active_process['process_dir']) / 'temp'
            temp_dir.mkdir(parents=True, exist_ok=True)
            return str(temp_dir)
        else:
            return self.base_settings.get_temp_directory()
    
    def get_absolute_path(self, relative_path: str) -> str:
        """Convierte una ruta relativa en absoluta desde el directorio del proyecto"""
        return self.base_settings.get_absolute_path(relative_path)
    
    def get_elasticsearch_credentials(self):
        """Obtiene las credenciales de Elasticsearch"""
        return self.base_settings.get_elasticsearch_credentials()
    
    def get_database_credentials(self):
        """Obtiene las credenciales de base de datos"""
        return self.base_settings.get_database_credentials()
    
    def get_elasticsearch_config_path(self) -> str:
        """Obtiene la ruta completa al archivo de configuración de Elasticsearch"""
        return self.base_settings.get_elasticsearch_config_path()
    
    def get_query_template_path(self) -> str:
        """Obtiene la ruta completa al archivo de template de query de Elasticsearch"""
        return self.base_settings.get_query_template_path()
    
    def __getattr__(self, name):
        """Delegar cualquier atributo faltante a base_settings"""
        return getattr(self.base_settings, name)
