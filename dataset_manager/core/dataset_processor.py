"""
Procesador principal del Dataset Manager
Orquesta todo el flujo de análisis desde extracción hasta generación de informes
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys

# Configurar logging
def setup_logging(settings):
    """Configura el sistema de logging"""
    from logging.handlers import RotatingFileHandler
    
    # Crear directorio de logs si no existe
    log_dir = settings.get_log_directory()
    log_file = os.path.join(log_dir, "dataset_manager.log")
    
    # Configurar logger principal
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, settings.logging_config.level.upper()))
    
    # Formatter
    formatter = logging.Formatter(settings.logging_config.format)
    
    # Handler para archivo con rotación
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=settings.logging_config.max_log_size_mb * 1024 * 1024,
        backupCount=settings.logging_config.max_log_files
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

class DatasetProcessor:
    """
    Procesador principal que orquesta todo el flujo de análisis del dataset
    """
    
    def __init__(self, settings=None, config_path: Optional[str] = None):
        """
        Inicializa el procesador principal
        
        Args:
            settings: Objeto Settings con configuración (opcional)
            config_path: Ruta al archivo de configuración (opcional)
        """
        # Importar aquí para evitar imports circulares
        from ..config.settings import Settings
        from ..data_extractor.extractor import DataExtractor
        from ..reference_analyzer.analyzer import ReferenceAnalyzer
        from ..image_downloader.downloader import ImageDownloader
        from ..report_generator.generator import ReportGenerator
        from ..process_aware_settings import ProcessAwareSettings
        
        # Configuración consciente de procesos
        if settings:
            self.settings = settings
        else:
            # Usar ProcessAwareSettings para ser consciente del proceso activo
            try:
                self.settings = ProcessAwareSettings(config_path)
            except:
                # Fallback a Settings normales si no hay proceso activo
                from ..config.settings import Settings
                self.settings = Settings(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.settings)
        
        # Setup process logger si hay proceso activo
        self.process_logger = None
        if hasattr(self.settings, 'active_process') and self.settings.active_process:
            from ..process_logger import ProcessLoggerManager
            process_info = self.settings.active_process
            # Usar process_dir si está disponible, sino construir el directorio
            process_directory = process_info.get('process_dir') or process_info.get('directory')
            if process_directory:
                self.process_logger = ProcessLoggerManager.get_logger(
                    process_info['name'], 
                    process_directory
                )
        
        # Inicializar componentes
        self.data_extractor = DataExtractor(self.settings)
        self.reference_analyzer = ReferenceAnalyzer(self.settings)
        self.image_downloader = ImageDownloader(self.settings)
        self.report_generator = ReportGenerator(self.settings)
        
        self.logger.info("Dataset Manager inicializado correctamente")
    
    def validate_inputs(self, deployment_id: int, **kwargs) -> bool:
        """
        Valida los parámetros de entrada
        
        Args:
            deployment_id: ID del deployment
            **kwargs: Parámetros adicionales
            
        Returns:
            True si las validaciones pasan
        """
        try:
            self.logger.info("Validando parámetros de entrada")
            
            # Validar deployment_id
            if not isinstance(deployment_id, int) or deployment_id <= 0:
                raise ValueError(f"deployment_id debe ser un entero positivo, recibido: {deployment_id}")
            
            # Validar configuración de Elasticsearch
            elastic_config_path = self.settings.get_elasticsearch_config_path()
            if not os.path.exists(elastic_config_path):
                raise FileNotFoundError(f"Archivo de configuración de Elasticsearch no encontrado: {elastic_config_path}")
            
            # Validar configuración de base de datos (variables de entorno)
            try:
                db_credentials = self.settings.get_database_credentials()
                self.logger.info("Credenciales de base de datos validadas")
            except ValueError as e:
                raise ValueError(f"Error en credenciales de base de datos: {e}")
            
            self.logger.info("Validacion de parametros exitosa")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en validacion: {e}")
            raise
    
    def process_deployment(self, deployment_id: int, 
                         days_back: int = 30,
                         download_images: bool = True,
                         generate_reports: bool = True,
                         base_model_id: int = None,
                         model_id: int = None) -> Dict[str, Any]:
        """
        Procesa un deployment completo siguiendo todo el flujo
        
        Args:
            deployment_id: ID del deployment a procesar
            days_back: Días hacia atrás para consulta de Elasticsearch
            download_images: Si descargar imágenes o no
            generate_reports: Si generar informes o no
            base_model_id: ID del modelo base específico a usar (opcional)
            model_id: ID del modelo específico a usar. Si no se especifica, usa el del deployment
            
        Returns:
            Diccionario con todos los resultados del procesamiento
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"Iniciando procesamiento completo para deployment {deployment_id}")
            
            # Log de proceso si está disponible
            if self.process_logger:
                self.process_logger.log_command_start(
                    f"download_info {deployment_id}",
                    f"Procesamiento completo: deployment {deployment_id}, {days_back} días, imágenes={'Sí' if download_images else 'No'}, reportes={'Sí' if generate_reports else 'No'}"
                )
            
            # 1. Validar entradas
            self.validate_inputs(deployment_id, days_back=days_back)
            
            # 2. Extraer datos
            self.logger.info("Paso 1: Extrayendo datos")
            if self.process_logger:
                self.process_logger.log_detailed("=== PASO 1: EXTRACCIÓN DE DATOS ===")
            
            extracted_data = self.data_extractor.extract_all_data(deployment_id, days_back, base_model_id, model_id)
            
            if self.process_logger:
                self.process_logger.log_step_completion("Extracción de datos", {
                    "Registros Elasticsearch": len(extracted_data.get('elastic_data', [])),
                    "Referencias": len(extracted_data.get('references_data', [])),
                    "Labels": len(extracted_data.get('labels_data', [])),
                    "Modelo activo": extracted_data.get('active_model_info', {}).get('model_id', 'N/A')
                })
            
            # 3. Analizar referencias
            self.logger.info("Paso 2: Analizando referencias")
            if self.process_logger:
                self.process_logger.log_detailed("=== PASO 2: ANÁLISIS DE REFERENCIAS ===")
            
            analysis_results = self.reference_analyzer.perform_complete_analysis(extracted_data)
            
            if self.process_logger:
                summary_stats = analysis_results.get('summary_stats', {})
                self.process_logger.log_step_completion("Análisis de referencias", {
                    "Referencias filtradas": summary_stats.get('total_filtered_references', 0),
                    "Sin asignar": summary_stats.get('unassigned_count', 0),
                    "No entrenadas": summary_stats.get('untrained_count', 0),
                    "Sugerencias alta confianza": summary_stats.get('high_confidence_suggestions', 0)
                })
            
            # 4. Descargar imágenes (opcional)
            download_results = {}
            if download_images:
                self.logger.info("📸 Paso 3: Descargando imágenes")
                if self.process_logger:
                    self.process_logger.log_detailed("=== PASO 3: DESCARGA DE IMÁGENES ===")
                
                download_results = self.image_downloader.download_all_images(
                    extracted_data, analysis_results, deployment_id
                )
                
                if self.process_logger:
                    download_stats = download_results.get('summary_stats', {})
                    self.process_logger.log_step_completion("Descarga de imágenes", {
                        "Referencias nuevas": download_stats.get('total_new_ref_images', 0),
                        "Clases similares": download_stats.get('total_similar_images', 0),
                        "Total descargadas": download_stats.get('total_downloaded', 0)
                    })
            else:
                self.logger.info(">> Paso 3: Descarga de imagenes omitida")
                if self.process_logger:
                    self.process_logger.log_detailed("=== PASO 3: DESCARGA DE IMÁGENES OMITIDA ===")
            
            # 5. Generar informes (opcional)
            report_files = {}
            if generate_reports:
                self.logger.info("Paso 4: Generando informes")
                if self.process_logger:
                    self.process_logger.log_detailed("=== PASO 4: GENERACIÓN DE INFORMES ===")
                
                report_files = self.report_generator.generate_all_reports(
                    extracted_data, analysis_results, download_results, deployment_id
                )
                
                if self.process_logger:
                    self.process_logger.log_step_completion("Generación de informes", {
                        "Informes generados": len(report_files),
                        "Archivos": list(report_files.keys())
                    })
            else:
                self.logger.info(">> Paso 4: Generacion de informes omitida")
                if self.process_logger:
                    self.process_logger.log_detailed("=== PASO 4: GENERACIÓN DE INFORMES OMITIDA ===")
            
            # 6. Compilar resultados finales
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            final_results = {
                'deployment_id': deployment_id,
                'processing_metadata': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'processing_time_seconds': processing_time,
                    'days_back': days_back,
                    'download_images': download_images,
                    'generate_reports': generate_reports
                },
                'extracted_data': extracted_data,
                'analysis_results': analysis_results,
                'download_results': download_results,
                'report_files': report_files,
                'summary': self._generate_processing_summary(
                    extracted_data, analysis_results, download_results, report_files
                )
            }
            
            self.logger.info(f"Procesamiento completo finalizado en {processing_time:.1f} segundos")
            self._log_final_summary(final_results)
            
            # Log final del proceso
            if self.process_logger:
                summary = final_results['summary']
                final_message = f"Deployment {deployment_id} procesado: {summary['data_extraction']['elastic_records']} registros ES, {summary['data_extraction']['references']} referencias, {len(summary['generated_reports'])} informes"
                self.process_logger.log_command_result(
                    f"download_info {deployment_id}",
                    final_message,
                    success=True
                )
                self.process_logger.log_process_end(processing_time, final_message)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error en procesamiento completo: {e}")
            raise
    
    def process_analysis_only(self, deployment_id: int, days_back: int = 30, model_id: int = None) -> Dict[str, Any]:
        """
        Ejecuta solo la extracción y análisis de datos, sin descarga de imágenes ni informes
        
        Args:
            deployment_id: ID del deployment
            days_back: Días hacia atrás para consulta
            model_id: ID del modelo específico a usar
            
        Returns:
            Resultados del análisis
        """
        return self.process_deployment(
            deployment_id=deployment_id,
            days_back=days_back,
            download_images=False,
            generate_reports=False,
            model_id=model_id
        )
    
    def process_with_custom_settings(self, deployment_id: int, 
                                   custom_settings: Dict[str, Any],
                                   **kwargs) -> Dict[str, Any]:
        """
        Procesa un deployment con configuraciones personalizadas
        
        Args:
            deployment_id: ID del deployment
            custom_settings: Configuraciones personalizadas
            **kwargs: Parámetros adicionales para process_deployment
            
        Returns:
            Resultados del procesamiento
        """
        try:
            self.logger.info("Aplicando configuraciones personalizadas")
            
            # Aplicar configuraciones personalizadas
            for section, settings_dict in custom_settings.items():
                for key, value in settings_dict.items():
                    self.settings.update_config(section, key, value)
            
            # Procesar con nueva configuración
            return self.process_deployment(deployment_id, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error procesando con configuraciones personalizadas: {e}")
            raise
    
    def _generate_processing_summary(self, extracted_data: Dict[str, Any],
                                   analysis_results: Dict[str, Any],
                                   download_results: Dict[str, Any],
                                   report_files: Dict[str, str]) -> Dict[str, Any]:
        """Genera resumen del procesamiento"""
        try:
            summary = {
                'data_extraction': {
                    'elastic_records': len(extracted_data.get('elastic_data', [])),
                    'references': len(extracted_data.get('references_data', [])),
                    'labels': len(extracted_data.get('labels_data', [])),
                    'active_model': extracted_data.get('active_model_info', {}).get('model_id', 'N/A')
                },
                'reference_analysis': analysis_results.get('summary_stats', {}),
                'consistency': {
                    'has_inconsistencies': analysis_results.get('consistency_analysis', {}).get('has_inconsistencies', False),
                    'consistency_ratio': analysis_results.get('consistency_analysis', {}).get('consistency_ratio', 0)
                },
                'image_download': download_results.get('summary_stats', {}) if download_results else {},
                'generated_reports': list(report_files.keys()) if report_files else []
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generando resumen: {e}")
            return {}
    
    def _log_final_summary(self, results: Dict[str, Any]):
        """Registra resumen final en logs"""
        try:
            summary = results['summary']
            
            summary_message = f"""
            
*** PROCESAMIENTO COMPLETADO EXITOSAMENTE ***
============================================

RESUMEN DE DATOS:
   • Registros Elasticsearch: {summary['data_extraction']['elastic_records']:,}
   • Referencias procesadas: {summary['data_extraction']['references']:,}
   • Labels disponibles: {summary['data_extraction']['labels']:,}
   • Modelo activo: {summary['data_extraction']['active_model']}

ANALISIS DE REFERENCIAS:
   • Referencias filtradas: {summary['reference_analysis'].get('total_filtered_references', 0):,}
   • Sin asignar: {summary['reference_analysis'].get('unassigned_count', 0):,}
   • No entrenadas: {summary['reference_analysis'].get('untrained_count', 0):,}
   • Sugerencias alta confianza: {summary['reference_analysis'].get('high_confidence_suggestions', 0):,}

CONSISTENCIA:
   • Ratio: {summary['consistency']['consistency_ratio']:.1%}
   • Inconsistencias: {'Si' if summary['consistency']['has_inconsistencies'] else 'No'}

IMAGENES DESCARGADAS:
   • Referencias nuevas: {summary['image_download'].get('total_new_ref_images', 0):,}
   • Clases similares: {summary['image_download'].get('total_similar_images', 0):,}

INFORMES GENERADOS: {len(summary['generated_reports'])}

TIEMPO TOTAL: {results['processing_metadata']['processing_time_seconds']:.1f} segundos
            """
            
            self.logger.info(summary_message)
            
        except Exception as e:
            self.logger.error(f"Error registrando resumen final: {e}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema
        
        Returns:
            Diccionario con información del estado
        """
        try:
            status = {
                'system_info': {
                    'version': '1.0.0',
                    'config_path': self.settings.config_path,
                    'elasticsearch_config': self.settings.get_elasticsearch_config_path(),
                    'output_directories': {
                        'images': self.settings.get_output_directory('images'),
                        'reports': self.settings.get_output_directory('reports')
                    }
                },
                'configuration': {
                    'min_appearances': self.settings.reference_analysis.min_appearances,
                    'confidence_threshold': self.settings.reference_analysis.confidence_threshold,
                    'images_per_class': getattr(self.settings.image_review, 'num_imagenes_revision', 5),
                    'parallel_workers': getattr(self.settings.image_processing, 'max_workers', 5)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estado del sistema: {e}")
            raise
