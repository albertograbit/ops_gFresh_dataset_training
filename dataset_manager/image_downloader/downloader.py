"""
Descargador y organizador de imágenes para entrenamiento
Maneja la descarga de imágenes desde URLs y su organización por clases
"""

import logging
import os
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ImageDownloader:
    """
    Descargador y organizador de imágenes para entrenamiento de modelos
    """
    
    def __init__(self, settings):
        """
        Inicializa el descargador de imágenes
        
        Args:
            settings: Objeto Settings con la configuración
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Dataset-Manager/1.0 (Image Downloader)'
        })
        
    def setup_output_directories(self, deployment_id: int) -> str:
        """
        Configura los directorios de salida para las imágenes
        
        Args:
            deployment_id: ID del deployment
            
        Returns:
            Ruta base del directorio de imágenes
        """
        try:
            base_dir = self.settings.get_output_directory('images')
            deployment_dir = os.path.join(base_dir, f"deployment_{deployment_id}")
            
            # Crear directorios principales
            os.makedirs(deployment_dir, exist_ok=True)
            os.makedirs(os.path.join(deployment_dir, "references_nuevas"), exist_ok=True)
            os.makedirs(os.path.join(deployment_dir, "clases_similares"), exist_ok=True)
            
            self.logger.info(f"Directorios de salida configurados en: {deployment_dir}")
            return deployment_dir
            
        except Exception as e:
            self.logger.error(f"Error configurando directorios: {e}")
            raise
    
    def extract_image_urls_from_elastic(self, elastic_df: pd.DataFrame, 
                                      reference_code: str) -> List[str]:
        """
        Extrae URLs de imágenes para una referencia específica desde datos de Elastic
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            reference_code: Código de referencia
            
        Returns:
            Lista de URLs de imágenes
        """
        try:
            # Filtrar datos para la referencia específica
            ref_data = elastic_df[
                (elastic_df.get('reference_code', '') == reference_code) |
                (elastic_df.get('product_code', '') == reference_code)
            ]
            
            if len(ref_data) == 0:
                return []
            
            # Buscar columnas que contengan URLs de imágenes
            image_url_columns = [col for col in ref_data.columns 
                               if 'image' in col.lower() and 'url' in col.lower()]
            
            urls = []
            for col in image_url_columns:
                col_urls = ref_data[col].dropna().tolist()
                urls.extend(col_urls)
            
            # También buscar en campos de rutas o paths
            if 'image_path' in ref_data.columns:
                paths = ref_data['image_path'].dropna().tolist()
                urls.extend(paths)
            
            # Limpiar y validar URLs
            valid_urls = []
            for url in urls:
                if isinstance(url, str) and (url.startswith('http') or url.startswith('/')):
                    valid_urls.append(url)
            
            # Eliminar duplicados manteniendo orden
            unique_urls = list(dict.fromkeys(valid_urls))
            
            self.logger.debug(f"Encontradas {len(unique_urls)} URLs para referencia {reference_code}")
            return unique_urls
            
        except Exception as e:
            self.logger.error(f"Error extrayendo URLs para {reference_code}: {e}")
            return []
    
    def download_single_image(self, url: str, output_path: str, 
                            timeout: int = 30) -> Dict[str, Any]:
        """
        Descarga una sola imagen desde una URL
        
        Args:
            url: URL de la imagen
            output_path: Ruta donde guardar la imagen
            timeout: Timeout en segundos
            
        Returns:
            Diccionario con resultado de la descarga
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Realizar descarga
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Verificar tamaño del archivo
            content_length = response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.settings.image_download.max_image_size_mb:
                    return {
                        'success': False,
                        'error': f'Imagen demasiado grande: {size_mb:.1f}MB',
                        'url': url,
                        'output_path': output_path
                    }
            
            # Guardar imagen
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verificar que el archivo se guardó correctamente
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return {
                    'success': True,
                    'url': url,
                    'output_path': output_path,
                    'file_size': os.path.getsize(output_path)
                }
            else:
                return {
                    'success': False,
                    'error': 'Archivo vacío o no se pudo guardar',
                    'url': url,
                    'output_path': output_path
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Error de descarga: {str(e)}',
                'url': url,
                'output_path': output_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error inesperado: {str(e)}',
                'url': url,
                'output_path': output_path
            }
    
    def download_images_for_reference(self, elastic_df: pd.DataFrame, 
                                    reference_code: str, reference_name: str,
                                    output_dir: str, max_images: Optional[int] = None) -> Dict[str, Any]:
        """
        Descarga imágenes para una referencia específica
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            reference_code: Código de referencia
            reference_name: Nombre de referencia (para carpeta)
            output_dir: Directorio base de salida
            max_images: Máximo número de imágenes a descargar
            
        Returns:
            Diccionario con resultados de la descarga
        """
        try:
            self.logger.info(f"Descargando imágenes para referencia: {reference_code}")
            
            # Obtener URLs de imágenes
            image_urls = self.extract_image_urls_from_elastic(elastic_df, reference_code)
            
            if not image_urls:
                self.logger.warning(f"No se encontraron URLs de imágenes para {reference_code}")
                return {
                    'reference_code': reference_code,
                    'total_urls': 0,
                    'downloaded': 0,
                    'failed': 0,
                    'results': []
                }
            
            # Limitar número de imágenes si se especifica
            if max_images:
                image_urls = image_urls[:max_images]
            
            # Crear directorio para esta referencia
            safe_ref_name = "".join(c for c in reference_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            ref_dir = os.path.join(output_dir, f"{reference_code}_{safe_ref_name}")
            os.makedirs(ref_dir, exist_ok=True)
            
            # Descargar imágenes en paralelo
            download_results = []
            successful_downloads = 0
            failed_downloads = 0
            
            with ThreadPoolExecutor(max_workers=self.settings.processing.parallel_workers) as executor:
                # Preparar tareas de descarga
                download_tasks = []
                for i, url in enumerate(image_urls):
                    # Determinar extensión del archivo
                    parsed_url = urlparse(url)
                    file_ext = os.path.splitext(parsed_url.path)[1]
                    if not file_ext or file_ext.lower() not in ['.jpg', '.jpeg', '.png', '.gif']:
                        file_ext = '.jpg'  # Extensión por defecto
                    
                    output_path = os.path.join(ref_dir, f"image_{i:04d}{file_ext}")
                    task = executor.submit(self.download_single_image, url, output_path)
                    download_tasks.append(task)
                
                # Procesar resultados
                for task in as_completed(download_tasks):
                    try:
                        result = task.result()
                        download_results.append(result)
                        
                        if result['success']:
                            successful_downloads += 1
                        else:
                            failed_downloads += 1
                            self.logger.debug(f"Fallo descarga: {result['error']}")
                            
                    except Exception as e:
                        failed_downloads += 1
                        self.logger.error(f"Error procesando tarea de descarga: {e}")
            
            # Resultado final
            download_summary = {
                'reference_code': reference_code,
                'reference_name': reference_name,
                'output_directory': ref_dir,
                'total_urls': len(image_urls),
                'downloaded': successful_downloads,
                'failed': failed_downloads,
                'success_rate': (successful_downloads / len(image_urls)) * 100 if image_urls else 0,
                'results': download_results
            }
            
            self.logger.info(f"Descarga completada para {reference_code}: "
                           f"{successful_downloads}/{len(image_urls)} imágenes")
            
            return download_summary
            
        except Exception as e:
            self.logger.error(f"Error descargando imágenes para {reference_code}: {e}")
            raise
    
    def download_images_for_similar_classes(self, elastic_df: pd.DataFrame, 
                                          label_suggestions: pd.DataFrame,
                                          labels_df: pd.DataFrame,
                                          output_dir: str) -> Dict[str, Any]:
        """
        Descarga imágenes para clases similares sugeridas
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            label_suggestions: DataFrame con sugerencias de labels
            labels_df: DataFrame con información de labels
            output_dir: Directorio base de salida
            
        Returns:
            Diccionario con resultados de descarga
        """
        try:
            self.logger.info("Descargando imágenes para clases similares")
            
            similar_classes_dir = os.path.join(output_dir, "clases_similares")
            download_results = []
            
            # Crear mapeo de label_id a label_name
            label_mapping = dict(zip(labels_df['label_id'], labels_df['label_name']))
            
            for _, suggestion_row in label_suggestions.iterrows():
                if suggestion_row.get('all_suggestions'):
                    for suggestion in suggestion_row['all_suggestions']:
                        label_id = suggestion['label_id']
                        label_name = label_mapping.get(label_id, f"label_{label_id}")
                        
                        # Buscar imágenes de esta clase en datos de Elastic
                        class_data = elastic_df[
                            (elastic_df.get('predicted_label_id') == label_id) |
                            (elastic_df.get('label_id') == label_id)
                        ]
                        
                        if len(class_data) > 0:
                            # Obtener URLs de imágenes para esta clase
                            class_urls = []
                            image_columns = [col for col in class_data.columns 
                                           if 'image' in col.lower() and 'url' in col.lower()]
                            
                            for col in image_columns:
                                urls = class_data[col].dropna().tolist()
                                class_urls.extend(urls)
                            
                            if class_urls:
                                # Limitar número de imágenes
                                max_images = min(len(class_urls), self.settings.image_download.images_per_class)
                                selected_urls = class_urls[:max_images]
                                
                                # Crear directorio para esta clase
                                safe_label_name = "".join(c for c in label_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                class_dir = os.path.join(similar_classes_dir, f"label_{label_id}_{safe_label_name}")
                                os.makedirs(class_dir, exist_ok=True)
                                
                                # Descargar imágenes
                                class_results = []
                                for i, url in enumerate(selected_urls):
                                    parsed_url = urlparse(url)
                                    file_ext = os.path.splitext(parsed_url.path)[1]
                                    if not file_ext:
                                        file_ext = '.jpg'
                                    
                                    output_path = os.path.join(class_dir, f"image_{i:04d}{file_ext}")
                                    result = self.download_single_image(url, output_path)
                                    class_results.append(result)
                                
                                successful = sum(1 for r in class_results if r['success'])
                                download_results.append({
                                    'label_id': label_id,
                                    'label_name': label_name,
                                    'directory': class_dir,
                                    'downloaded': successful,
                                    'total': len(selected_urls)
                                })
            
            self.logger.info(f"Descarga de clases similares completada: {len(download_results)} clases")
            return {'similar_classes_downloads': download_results}
            
        except Exception as e:
            self.logger.error(f"Error descargando imágenes de clases similares: {e}")
            raise
    
    def download_all_images(self, extracted_data: Dict[str, Any], 
                          analysis_results: Dict[str, Any],
                          deployment_id: int) -> Dict[str, Any]:
        """
        Descarga todas las imágenes necesarias para el análisis
        
        Args:
            extracted_data: Datos extraídos
            analysis_results: Resultados del análisis de referencias
            deployment_id: ID del deployment
            
        Returns:
            Diccionario con todos los resultados de descarga
        """
        try:
            self.logger.info("Iniciando descarga completa de imágenes")
            
            # Configurar directorios
            output_dir = self.setup_output_directories(deployment_id)
            
            elastic_df = extracted_data['elastic_data']
            labels_df = extracted_data['labels_data']
            untrained_refs = analysis_results['untrained_references']
            label_suggestions = analysis_results['label_suggestions']
            
            # 1. Descargar imágenes para referencias no entrenadas
            new_references_dir = os.path.join(output_dir, "references_nuevas")
            new_ref_downloads = []
            
            for _, ref_row in untrained_refs.iterrows():
                download_result = self.download_images_for_reference(
                    elastic_df=elastic_df,
                    reference_code=ref_row['reference_code'],
                    reference_name=ref_row['reference_name'],
                    output_dir=new_references_dir,
                    max_images=self.settings.image_download.images_per_class
                )
                new_ref_downloads.append(download_result)
            
            # 2. Descargar imágenes para clases similares
            similar_downloads = self.download_images_for_similar_classes(
                elastic_df=elastic_df,
                label_suggestions=label_suggestions,
                labels_df=labels_df,
                output_dir=output_dir
            )
            
            # Compilar resultados
            download_summary = {
                'deployment_id': deployment_id,
                'output_directory': output_dir,
                'new_references_downloads': new_ref_downloads,
                'similar_classes_downloads': similar_downloads,
                'summary_stats': {
                    'total_new_references': len(new_ref_downloads),
                    'total_new_ref_images': sum(d['downloaded'] for d in new_ref_downloads),
                    'total_similar_classes': len(similar_downloads.get('similar_classes_downloads', [])),
                    'total_similar_images': sum(d['downloaded'] for d in similar_downloads.get('similar_classes_downloads', []))
                }
            }
            
            self.logger.info("Descarga completa de imágenes finalizada")
            return download_summary
            
        except Exception as e:
            self.logger.error(f"Error en descarga completa de imágenes: {e}")
            raise
        finally:
            # Cerrar sesión de requests
            self.session.close()
