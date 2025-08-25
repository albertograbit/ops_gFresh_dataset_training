"""
Extractor de datos desde Elasticsearch y base de datos relacional
Maneja la descarga y procesamiento inicial de datos necesarios para el análisis
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json

# Importar utilidades existentes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.elastic_utils import load_config, connect_es, download_data, convert_to_dataframe
from utils.file_manager import write_csv_sc, ensure_parent_dir
from utils.bbdd_utils import get_db_connection, add_image_columns_to_dataframe

class DataExtractor:
    """
    Extractor principal de datos desde Elasticsearch y base de datos relacional
    """
    
    def __init__(self, settings):
        """
        Inicializa el extractor de datos
        
        Args:
            settings: Objeto Settings con la configuración
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.es_client = None
        self.db_connection = None
        
    def setup_connections(self):
        """Configura las conexiones a Elasticsearch y base de datos"""
        try:
            # Conexión a Elasticsearch usando el nuevo método seguro
            elastic_config = self.settings.get_elasticsearch_credentials()
            
            self.es_client = connect_es(
                host=elastic_config['host'],
                username=elastic_config.get('username', ''),
                password=elastic_config.get('password', ''),
                verify_certs=elastic_config.get('verify_certs', True),
                auth_method=elastic_config.get('auth_method'),
                api_key_id=elastic_config.get('api_key_id'),
                api_key_secret=elastic_config.get('api_key_secret'),
                port=elastic_config.get('port', 9200)
            )
            # Ajustar protocolo según use_ssl si está definido y verify_certs es False pero se quiere https
            if elastic_config.get('use_ssl') and not elastic_config.get('verify_certs', True):
                # Reconectar forzando https sin verificación de certs
                self.logger.debug('Reconectando a Elasticsearch forzando HTTPS sin verify_certs')
                self.es_client = connect_es(
                    host=elastic_config['host'],
                    username=elastic_config.get('username', ''),
                    password=elastic_config.get('password', ''),
                    verify_certs=False,
                    auth_method=elastic_config.get('auth_method'),
                    api_key_id=elastic_config.get('api_key_id'),
                    api_key_secret=elastic_config.get('api_key_secret'),
                port=elastic_config.get('port', 9200)
            )
            self.logger.info("Conexión a Elasticsearch establecida")
            
            # Conexión a base de datos (opcional, usa variables de entorno ya cargadas)
            try:
                self.db_connection = get_db_connection()
                self.logger.info("Conexión a base de datos establecida")
            except Exception as db_error:
                self.logger.warning(f"No se pudo conectar a la base de datos (función limitada): {db_error}")
                self.db_connection = None
            
        except Exception as e:
            self.logger.error(f"Error estableciendo conexiones: {e}")
            raise
    
    def _resolve_model_id(self, deployment_id: int, requested_model_id: int = None) -> int:
        """
        Resuelve qué model_id usar según la lógica especificada:
        1. Si se pasa model_id explícitamente, verificar conflictos y usar ese
        2. Si no se pasa, obtener el model_id de la tabla deployment
        3. Si el deployment no tiene model_id, mostrar error
        
        Args:
            deployment_id: ID del deployment
            requested_model_id: ID del modelo solicitado explícitamente (opcional)
            
        Returns:
            ID del modelo a usar
            
        Raises:
            ValueError: Si no se puede resolver el model_id
        """
        try:
            if not self.db_connection:
                raise ValueError("Conexión a base de datos no disponible para resolver model_id")
            
            cursor = self.db_connection.cursor()
            
            # 1. Obtener información del deployment actual
            cursor.execute("""
                SELECT deployment_id, model_id 
                FROM deployment 
                WHERE deployment_id = %s
            """, (deployment_id,))
            
            deployment_info = cursor.fetchone()
            if not deployment_info:
                raise ValueError(f"Deployment {deployment_id} no encontrado en la base de datos")
            
            deployment_model_id = deployment_info[1]  # model_id del deployment
            
            # 2. Si se especificó un model_id explícitamente
            if requested_model_id is not None:
                self.logger.info(f"Usando model_id especificado: {requested_model_id}")
                
                # Verificar si este model_id está usado en otros deployments
                cursor.execute("""
                    SELECT deployment_id 
                    FROM deployment 
                    WHERE model_id = %s AND deployment_id != %s
                """, (requested_model_id, deployment_id))
                
                other_deployments = cursor.fetchall()
                if other_deployments:
                    other_deployment_ids = [row[0] for row in other_deployments]
                    self.logger.warning(f"ATENCIÓN: El model_id {requested_model_id} está siendo usado en otros deployments: {other_deployment_ids}")
                    
                    # Solicitar confirmación del usuario
                    response = input(f"¿Desea continuar usando model_id {requested_model_id}? (s/N): ").strip().lower()
                    if response not in ['s', 'sí', 'si', 'y', 'yes']:
                        raise ValueError("Operación cancelada por el usuario")
                
                return requested_model_id
            
            # 3. Si no se especificó model_id, usar el del deployment
            if deployment_model_id is None:
                raise ValueError(f"El deployment {deployment_id} no tiene model_id asignado. "
                               f"Especifique explícitamente el model_id a usar con --model")
            
            self.logger.info(f"Usando model_id del deployment: {deployment_model_id}")
            return deployment_model_id
            
        except Exception as e:
            self.logger.error(f"Error resolviendo model_id: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def extract_elasticsearch_data(self, deployment_id: int, days_back: int = 30, model_id: int = None) -> pd.DataFrame:
        """
        Extrae datos de inferencia desde Elasticsearch para un deployment y modelo específicos
        Con fallback automático si el modelo especificado no tiene datos en ese deployment
        
        Args:
            deployment_id: ID del deployment a analizar
            days_back: Número de días hacia atrás para la consulta
            model_id: ID del modelo específico a filtrar (con fallback automático)
            
        Returns:
            DataFrame con los datos de Elasticsearch
        """
        try:
            self.logger.info(f"Extrayendo datos de Elasticsearch para deployment_id: {deployment_id}")
            
            # Si se especifica un model_id, intentar primero con él
            if model_id:
                self.logger.info(f"Intentando extraer datos con model_id especificado: {model_id}")
                df_with_model = self._extract_with_model_filter(deployment_id, days_back, model_id)
                
                if len(df_with_model) > 0:
                    self.logger.info(f"Datos encontrados con model_id {model_id}: {len(df_with_model)} registros")
                    return df_with_model
                else:
                    self.logger.warning(f"No se encontraron datos con model_id {model_id} para deployment {deployment_id}")
                    self.logger.info("Aplicando fallback: usando model_id del deployment en la base de datos")
                    
                    # Obtener el model_id del deployment desde la BD
                    fallback_model_id = self._get_deployment_model_id(deployment_id)
                    if fallback_model_id and fallback_model_id != model_id:
                        self.logger.info(f"Extrayendo datos con model_id del deployment: {fallback_model_id}")
                        return self._extract_with_model_filter(deployment_id, days_back, fallback_model_id)
                    else:
                        self.logger.info("Extrayendo todos los datos del deployment sin filtro de modelo")
                        return self._extract_with_model_filter(deployment_id, days_back, None)
            else:
                # Si no se especifica model_id, extraer todos los datos del deployment
                self.logger.info("Extrayendo todos los datos del deployment")
                return self._extract_with_model_filter(deployment_id, days_back, None)
                
        except Exception as e:
            self.logger.error(f"Error extrayendo datos de Elasticsearch: {e}")
            raise
    
    def _extract_with_model_filter(self, deployment_id: int, days_back: int, model_id: int = None) -> pd.DataFrame:
        """
        Función auxiliar para extraer datos con o sin filtro de modelo
        """
        try:
            # Construir query basada en la configuración
            query_must = [
                {
                    "terms": {
                        "transaction_metric.keyword": self.settings.elasticsearch.transaction_metrics
                    }
                },
                {
                    "terms": {
                        "deployment_id": [deployment_id]
                    }
                },
                {
                    "range": {
                        "transaction_start_time": {
                            "gte": f"now-{days_back}d/d",
                            "lt": "now+1d/d"
                        }
                    }
                }
            ]
            
            # Añadir filtro por model_id si se especifica (campo dentro de transaction_data)
            if model_id is not None:
                query_must.append({
                    "terms": {
                        "transaction_data.model_id": [model_id]
                    }
                })
            
            query_body = {
                "size": self.settings.elasticsearch.max_results,
                "query": {
                    "bool": {
                        "must": query_must
                    }
                },
                "sort": [
                    {
                        "transaction_start_time": {
                            "order": "desc"
                        }
                    }
                ]
            }
            
            # Descargar datos usando el archivo de query template
            # Primero necesitamos crear un archivo temporal con nuestro query
            query_template = {
                "query_body": query_body,
                "index": self.settings.elasticsearch.index_pattern
            }
            
            # Escribir query temporal para compatibilidad con download_data
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_query:
                json.dump(query_template, temp_query, ensure_ascii=False, indent=2)
                temp_query_path = temp_query.name
            
            try:
                # Descargar datos
                raw_data = download_data(
                    es=self.es_client,
                    query_file=temp_query_path,
                    scroll=self.settings.elasticsearch.scroll_timeout
                )
            finally:
                # Limpiar archivo temporal
                import os
                if os.path.exists(temp_query_path):
                    os.unlink(temp_query_path)
            
            # Convertir a DataFrame
            df = convert_to_dataframe(raw_data)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error en extracción con filtro de modelo: {e}")
            raise
    
    def _get_deployment_model_id(self, deployment_id: int) -> int:
        """
        Obtiene el model_id asociado al deployment desde la base de datos
        """
        cursor = None
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT model_id FROM deployment WHERE deployment_id = %s"
            cursor.execute(query, (deployment_id,))
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error obteniendo model_id del deployment {deployment_id}: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def extract_references_data(self, deployment_id: int) -> pd.DataFrame:
        """
        Extrae datos de referencias desde la base de datos para un deployment específico
        
        Args:
            deployment_id: ID del deployment
            
        Returns:
            DataFrame con las referencias
        """
        try:
            self.logger.info(f"Extrayendo referencias para deployment_id: {deployment_id}")
            
            # Verificar si hay conexión a base de datos
            if not self.db_connection:
                self.logger.warning("Base de datos no disponible. Retornando DataFrame vacío para referencias.")
                return pd.DataFrame()
            
            query = """
            SELECT 
                reference_id,
                reference_code,
                reference_name,
                label_id,
                deployment_id,
                created_at,
                updated_at
            FROM reference 
            WHERE deployment_id = %s
            ORDER BY reference_id
            """
            
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query, (deployment_id,))
            results = cursor.fetchall()
            cursor.close()
            
            df = pd.DataFrame(results)
            self.logger.info(f"Extraídas {len(df)} referencias de la base de datos")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extrayendo referencias: {e}")
            raise
    
    def extract_labels_data(self) -> pd.DataFrame:
        """
        Extrae todas las etiquetas (labels) desde la base de datos
        
        Returns:
            DataFrame con todas las etiquetas
        """
        try:
            self.logger.info("Extrayendo todas las etiquetas")
            
            # Verificar si hay conexión a base de datos
            if not self.db_connection:
                self.logger.warning("Base de datos no disponible. Retornando DataFrame vacío para etiquetas.")
                return pd.DataFrame()
            
            query = """
            SELECT 
                label_id,
                label_name,
                created_at,
                updated_at
            FROM label 
            ORDER BY label_id
            """
            
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            df = pd.DataFrame(results)
            self.logger.info(f"Extraídas {len(df)} etiquetas de la base de datos")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extrayendo etiquetas: {e}")
            raise
    
    def get_active_model_info(self, elastic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identifica el modelo activo basado en los últimos 1000 registros de Elastic
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            
        Returns:
            Diccionario con información del modelo activo
        """
        try:
            self.logger.info("Identificando modelo activo")
            
            # Tomar los últimos 1000 registros ordenados por fecha
            recent_data = elastic_df.head(1000)
            
            # Contar frecuencia de model_id
            if 'model_id' in recent_data.columns:
                model_counts = recent_data['model_id'].value_counts()
                active_model_id = model_counts.index[0]
                frequency = model_counts.iloc[0]
                
                self.logger.info(f"Modelo activo identificado: {active_model_id} (frecuencia: {frequency})")
                
                return {
                    'model_id': active_model_id,
                    'frequency': frequency,
                    'total_records': len(recent_data),
                    'percentage': (frequency / len(recent_data)) * 100
                }
            else:
                self.logger.warning("Campo 'model_id' no encontrado en datos de Elasticsearch")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error identificando modelo activo: {e}")
            raise
    
    def extract_model_data(self, model_id: int) -> Dict[str, Any]:
        """
        Extrae la información del modelo desde la base de datos
        
        Args:
            model_id: ID del modelo
            
        Returns:
            Diccionario con los datos del modelo
        """
        try:
            self.logger.info(f"Extrayendo datos del modelo: {model_id}")
            
            # Verificar si hay conexión a base de datos
            if not self.db_connection:
                self.logger.warning("Base de datos no disponible. Retornando diccionario vacío para modelo.")
                return {}
            
            # Convertir model_id a int nativo de Python para evitar problemas con numpy int64
            model_id = int(model_id)
            
            query = """
            SELECT 
                model_id,
                model_code,
                model_data,
                created_at,
                updated_at
            FROM model 
            WHERE model_id = %s
            """
            
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute(query, (model_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # Parsear model_data si es JSON string
                if isinstance(result['model_data'], str):
                    try:
                        result['model_data'] = json.loads(result['model_data'])
                    except json.JSONDecodeError:
                        self.logger.warning(f"No se pudo parsear model_data como JSON para modelo {model_id}")
                
                self.logger.info(f"Datos del modelo {model_id} extraídos exitosamente")
                return result
            else:
                self.logger.warning(f"No se encontró modelo con ID: {model_id}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error extrayendo datos del modelo: {e}")
            raise
    
    def extract_all_data(self, deployment_id: int, days_back: int = 30, base_model_id: int = None, model_id: int = None) -> Dict[str, Any]:
        """
        Extrae todos los datos necesarios para el análisis
        
        Args:
            deployment_id: ID del deployment
            days_back: Días hacia atrás para consulta de Elasticsearch
            base_model_id: ID del modelo base específico a usar (opcional)
            model_id: ID del modelo específico a usar. Si no se especifica, usa el del deployment
            
        Returns:
            Diccionario con todos los datos extraídos
        """
        try:
            self.logger.info(f"Iniciando extracción completa de datos para deployment {deployment_id}")
            
            # Configurar conexiones
            self.setup_connections()
            
            # Resolver model_id (solo para indicadores / modelo activo) pero NO filtrar extracción
            resolved_model_id = self._resolve_model_id(deployment_id, model_id)
            
            # Extraer TODOS los datos de Elasticsearch del deployment sin filtrar por modelo
            # (los indicadores %OK se calcularán luego solo sobre las transacciones del modelo)
            elastic_df = self.extract_elasticsearch_data(deployment_id, days_back, None)
            
            # Añadir columnas de información de imágenes a los datos de Elasticsearch
            if not elastic_df.empty and 'transaction_id' in elastic_df.columns:
                self.logger.info("Añadiendo columnas de información de imágenes a los datos de Elasticsearch")
                elastic_df = add_image_columns_to_dataframe(elastic_df, 'transaction_id')
            else:
                self.logger.warning("No se pudieron añadir columnas de imágenes: datos vacíos o falta columna transaction_id")
            
            # Extraer datos de base de datos
            references_df = self.extract_references_data(deployment_id)
            labels_df = self.extract_labels_data()
            
            # Determinar qué modelo usar
            model_to_use = None
            if base_model_id:
                # Usar modelo base especificado
                self.logger.info(f"Usando modelo base especificado: {base_model_id}")
                model_to_use = base_model_id
                active_model_info = {'model_id': base_model_id, 'is_base_model': True}
            else:
                # Identificar modelo activo
                active_model_info = self.get_active_model_info(elastic_df)
                if active_model_info and 'model_id' in active_model_info:
                    model_to_use = active_model_info['model_id']
                    active_model_info['is_base_model'] = False
            
            # Extraer datos del modelo
            model_data = {}
            if model_to_use:
                model_data = self.extract_model_data(model_to_use)
            
            # Cache Elasticsearch data a CSV (para reutilización en creación de dataset sin re-leer Excel)
            try:
                if not elastic_df.empty:
                    # Guardar siempre en la carpeta reports del proceso activo (o reports global)
                    if hasattr(self.settings, 'active_process') and self.settings.active_process:
                        proc_info = self.settings.active_process
                        process_dir = proc_info.get('process_dir') or proc_info.get('directory')
                        if process_dir:
                            reports_dir = os.path.join(process_dir, 'reports')
                        else:
                            reports_dir = self.settings.get_output_directory('reports')
                    else:
                        reports_dir = self.settings.get_output_directory('reports')
                    os.makedirs(reports_dir, exist_ok=True)
                    cache_path = os.path.join(reports_dir, 'elastic_data.csv')
                    write_csv_sc(elastic_df, cache_path, index=False)
                    self.logger.info(f"Datos Elasticsearch cacheados en: {cache_path}")
            except Exception as ce:
                self.logger.warning(f"No se pudo cachear datos Elasticsearch: {ce}")

            # Compilar todos los datos
            extracted_data = {
                'elastic_data': elastic_df,
                'references_data': references_df,
                'labels_data': labels_df,
                'active_model_info': active_model_info,
                'model_data': model_data,
                'extraction_metadata': {
                    'deployment_id': deployment_id,
                    'days_back': days_back,
                    'base_model_id': base_model_id,
                    'model_used': model_to_use,  # model_id usado para indicadores (%OK, etc.)
                    'is_base_model': active_model_info.get('is_base_model', False) if active_model_info else False,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'total_elastic_records': len(elastic_df),
                    'total_references': len(references_df),
                    'total_labels': len(labels_df)
                }
            }
            
            self.logger.info("Extracción completa de datos finalizada exitosamente")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error en extracción completa de datos: {e}")
            raise
        finally:
            # Cerrar conexiones
            self.close_connections()
    
    def close_connections(self):
        """Cierra las conexiones abiertas"""
        try:
            if self.db_connection and self.db_connection.is_connected():
                self.db_connection.close()
                self.logger.info("Conexión a base de datos cerrada")
        except Exception as e:
            self.logger.warning(f"Error cerrando conexiones: {e}")
