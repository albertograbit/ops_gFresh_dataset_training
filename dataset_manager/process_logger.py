"""
Process Logger - Sistema de logging dual para procesos
Mantiene dos logs: uno detallado y otro de comandos/resumen
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler


class ProcessLogger:
    """
    Logger especializado para procesos que mantiene dos tipos de logs:
    1. Log detallado: Todo el output del proceso
    2. Log de comandos: Solo comandos ejecutados y resultados principales
    """
    
    def __init__(self, process_name: str, process_directory: str):
        """
        Inicializa el logger de proceso
        
        Args:
            process_name: Nombre del proceso
            process_directory: Directorio base del proceso
        """
        self.process_name = process_name
        self.process_directory = process_directory
        
        # Crear directorio de logs si no existe
        self.logs_dir = os.path.join(process_directory, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configurar loggers
        self._setup_detailed_logger()
        self._setup_command_logger()
        
        # Archivo de historial de comandos
        self.commands_history_file = os.path.join(self.logs_dir, f"{process_name}_commands_history.log")
        
        # Inicializar con marca de inicio
        self._log_process_start()
    
    def _setup_detailed_logger(self):
        """Configura el logger detallado"""
        detailed_log_file = os.path.join(self.logs_dir, f"{self.process_name}_detailed.log")
        
        self.detailed_logger = logging.getLogger(f"detailed_{self.process_name}")
        self.detailed_logger.setLevel(logging.DEBUG)
        
        # Evitar duplicar handlers
        if not self.detailed_logger.handlers:
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            
            detailed_handler = RotatingFileHandler(
                detailed_log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3
            )
            detailed_handler.setFormatter(detailed_formatter)
            self.detailed_logger.addHandler(detailed_handler)
    
    def _setup_command_logger(self):
        """Configura el logger de comandos"""
        command_log_file = os.path.join(self.logs_dir, f"{self.process_name}_commands.log")
        
        self.command_logger = logging.getLogger(f"commands_{self.process_name}")
        self.command_logger.setLevel(logging.INFO)
        
        # Evitar duplicar handlers
        if not self.command_logger.handlers:
            command_formatter = logging.Formatter(
                '%(asctime)s | %(message)s'
            )
            
            command_handler = RotatingFileHandler(
                command_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            command_handler.setFormatter(command_formatter)
            self.command_logger.addHandler(command_handler)
    
    def _log_process_start(self):
        """Registra el inicio del proceso"""
        start_message = f"=== INICIO DE PROCESO: {self.process_name} ==="
        self.detailed_logger.info(start_message)
        self.command_logger.info(start_message)
    
    def log_command_start(self, command: str, description: str = ""):
        """
        Registra el inicio de un comando
        
        Args:
            command: Comando que se está ejecutando
            description: Descripción adicional del comando
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Log detallado
        self.detailed_logger.info(f"[INICIANDO] COMANDO: {command}")
        if description:
            self.detailed_logger.info(f"   Descripción: {description}")
        
        # Log de comandos
        command_entry = f"[{timestamp}] EJECUTANDO: {command}"
        if description:
            command_entry += f" - {description}"
        self.command_logger.info(command_entry)
    
    def log_command_result(self, command: str, result_summary: str, success: bool = True):
        """
        Registra el resultado de un comando
        
        Args:
            command: Comando ejecutado
            result_summary: Resumen del resultado
            success: Si el comando fue exitoso
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        status = "[EXITOSO]" if success else "[ERROR]"
        
        # Log detallado
        self.detailed_logger.info(f"{status}: {command}")
        self.detailed_logger.info(f"   Resultado: {result_summary}")
        
        # Log de comandos
        self.command_logger.info(f"[{timestamp}] {status}: {command} -> {result_summary}")
    
    def log_step_completion(self, step_name: str, key_metrics: Dict[str, Any]):
        """
        Registra la finalización de un paso del proceso
        
        Args:
            step_name: Nombre del paso completado
            key_metrics: Métricas clave del paso
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Log detallado
        self.detailed_logger.info(f"[COMPLETADO] PASO: {step_name}")
        for metric, value in key_metrics.items():
            self.detailed_logger.info(f"   {metric}: {value}")
        
        # Log de comandos - formato compacto
        metrics_str = " | ".join([f"{k}: {v}" for k, v in key_metrics.items()])
        self.command_logger.info(f"[{timestamp}] COMPLETADO: {step_name} -> {metrics_str}")
    
    def log_detailed(self, message: str, level: str = "info"):
        """
        Log solo en el archivo detallado
        
        Args:
            message: Mensaje a loggear
            level: Nivel de log (info, debug, warning, error)
        """
        log_method = getattr(self.detailed_logger, level.lower(), self.detailed_logger.info)
        log_method(message)
    
    def log_process_end(self, total_time: float, final_summary: str):
        """
        Registra el final del proceso
        
        Args:
            total_time: Tiempo total de procesamiento en segundos
            final_summary: Resumen final del proceso
        """
        end_message = f"=== FIN DE PROCESO: {self.process_name} ==="
        time_message = f"Tiempo total: {total_time:.1f} segundos"
        directory_message = f"Directorio: {self.process_directory}"
        
        # Log detallado
        self.detailed_logger.info(end_message)
        self.detailed_logger.info(time_message)
        self.detailed_logger.info(directory_message)
        self.detailed_logger.info(f"Resumen: {final_summary}")
        
        # Log de comandos
        self.command_logger.info(end_message)
        self.command_logger.info(f"{time_message} | {directory_message} | {final_summary}")
    
    def get_logs_info(self) -> Dict[str, str]:
        """
        Obtiene información sobre los archivos de log
        
        Returns:
            Diccionario con rutas de los logs
        """
        return {
            'detailed_log': os.path.join(self.logs_dir, f"{self.process_name}_detailed.log"),
            'commands_log': os.path.join(self.logs_dir, f"{self.process_name}_commands.log"),
            'logs_directory': self.logs_dir
        }


class ProcessLoggerManager:
    """
    Gestor de loggers de proceso que mantiene instancias activas
    """
    
    _loggers: Dict[str, ProcessLogger] = {}
    
    @classmethod
    def get_logger(cls, process_name: str, process_directory: str) -> ProcessLogger:
        """
        Obtiene o crea un logger para un proceso
        
        Args:
            process_name: Nombre del proceso
            process_directory: Directorio del proceso
            
        Returns:
            ProcessLogger para el proceso
        """
        if process_name not in cls._loggers:
            cls._loggers[process_name] = ProcessLogger(process_name, process_directory)
        
        return cls._loggers[process_name]
    
    @classmethod
    def cleanup_logger(cls, process_name: str):
        """
        Limpia un logger específico
        
        Args:
            process_name: Nombre del proceso a limpiar
        """
        if process_name in cls._loggers:
            # Cerrar handlers
            logger = cls._loggers[process_name]
            for handler in logger.detailed_logger.handlers:
                handler.close()
            for handler in logger.command_logger.handlers:
                handler.close()
            
            del cls._loggers[process_name]
    
    @classmethod
    def cleanup_all(cls):
        """Limpia todos los loggers"""
        for process_name in list(cls._loggers.keys()):
            cls.cleanup_logger(process_name)
