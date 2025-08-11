"""
Gestor de procesos para Dataset Manager
Maneja procesos activos y nomenclatura de procesos
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class ProcessManager:
    """Gestor de procesos para nomenclatura y seguimiento de procesos activos"""
    
    def __init__(self, base_dir: str = None):
        """
        Inicializar el gestor de procesos
        
        Args:
            base_dir: Directorio base del proyecto
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.process_file = self.base_dir / "config" / "active_process.json"
        self.logger = logging.getLogger(__name__)
        
        # Asegurar que existe el directorio config
        self.process_file.parent.mkdir(exist_ok=True)
    
    def generate_process_name(self, deployment_id: str, custom_name: str = None) -> str:
        """
        Generar nombre de proceso automáticamente
        
        Args:
            deployment_id: ID del deployment
            custom_name: Nombre personalizado (opcional)
            
        Returns:
            Nombre del proceso generado
        """
        if custom_name:
            return custom_name
            
        # Formato: dataset_{deployment_id}_{DDMMAA}_v{version}
        today = datetime.now().strftime("%d%m%y")
        
        # Buscar versión más alta para esta fecha y deployment
        version = self._get_next_version(deployment_id, today)
        
        return f"dataset_{deployment_id}_{today}_v{version}"
    
    def _get_next_version(self, deployment_id: str, date: str) -> int:
        """
        Obtener la siguiente versión para un deployment y fecha dados
        
        Args:
            deployment_id: ID del deployment
            date: Fecha en formato DDMMAA
            
        Returns:
            Siguiente número de versión
        """
        # Buscar en archivos existentes procesos con mismo deployment y fecha
        output_dir = self.base_dir / "output" / "processes"
        
        if not output_dir.exists():
            return 1
            
        max_version = 0
        pattern = f"dataset_{deployment_id}_{date}_v"
        
        try:
            for folder in output_dir.iterdir():
                if folder.is_dir() and folder.name.startswith(pattern):
                    # Extraer versión del nombre
                    try:
                        version_part = folder.name.replace(pattern, "").split("_")[0]
                        version = int(version_part)
                        max_version = max(max_version, version)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            self.logger.warning(f"Error buscando versiones existentes: {e}")
            
        return max_version + 1
    
    def set_active_process(self, process_name: str, metadata: Dict[str, Any] = None) -> None:
        """
        Establecer un proceso como activo y crear su estructura de carpetas
        
        Args:
            process_name: Nombre del proceso
            metadata: Metadatos adicionales del proceso
        """
        # Crear estructura de carpetas para el proceso
        process_dir = self.create_process_directory(process_name, metadata)
        
        process_data = {
            "name": process_name,
            "created_at": datetime.now().isoformat(),
            "process_dir": str(process_dir),
            "metadata": metadata or {}
        }
        
        try:
            with open(self.process_file, 'w', encoding='utf-8') as f:
                json.dump(process_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Proceso activo establecido: {process_name}")
            self.logger.info(f"Directorio del proceso: {process_dir}")
            
        except Exception as e:
            self.logger.error(f"Error guardando proceso activo: {e}")
            raise
    
    def create_process_directory(self, process_name: str, metadata: Dict[str, Any] = None) -> Path:
        """
        Crear estructura de directorios para un proceso
        
        Args:
            process_name: Nombre del proceso
            metadata: Metadatos del proceso
            
        Returns:
            Path del directorio del proceso creado
        """
        # Directorio base para procesos
        processes_dir = self.base_dir / "output" / "processes"
        process_dir = processes_dir / process_name
        
        # Crear directorios
        directories = [
            process_dir,
            process_dir / "reports",
            process_dir / "images",
            process_dir / "images" / "revisar_imagenes",
            process_dir / "data",
            process_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Guardar metadata del proceso
        metadata_file = process_dir / "process_metadata.json"
        process_metadata = {
            "name": process_name,
            "created_at": datetime.now().isoformat(),
            "directories": {
                "base": str(process_dir),
                "reports": str(process_dir / "reports"),
                "images": str(process_dir / "images"),
                "revisar_imagenes": str(process_dir / "images" / "revisar_imagenes"),
                "data": str(process_dir / "data"),
                "logs": str(process_dir / "logs")
            }
        }
        
        if metadata:
            process_metadata.update(metadata)
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(process_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error guardando metadata del proceso: {e}")
        
        return process_dir
    
    def get_process_directory(self, process_name: str = None) -> Optional[Path]:
        """
        Obtener el directorio de un proceso (activo si no se especifica nombre)
        
        Args:
            process_name: Nombre del proceso (opcional, usa el activo si no se especifica)
            
        Returns:
            Path del directorio del proceso o None si no existe
        """
        if process_name:
            process_dir = self.base_dir / "output" / "processes" / process_name
            return process_dir if process_dir.exists() else None
        else:
            active_process = self.get_active_process()
            if active_process and 'process_dir' in active_process:
                return Path(active_process['process_dir'])
            return None
    
    def get_active_process(self) -> Optional[Dict[str, Any]]:
        """
        Obtener información del proceso activo
        
        Returns:
            Información del proceso activo o None si no hay ninguno
        """
        if not self.process_file.exists():
            return None
            
        try:
            with open(self.process_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error leyendo proceso activo: {e}")
            return None
    
    def clear_active_process(self) -> None:
        """Limpiar el proceso activo"""
        if self.process_file.exists():
            try:
                self.process_file.unlink()
                self.logger.info("Proceso activo limpiado")
            except Exception as e:
                self.logger.error(f"Error limpiando proceso activo: {e}")
