"""
Módulo para análisis de referencias y estado de entrenamiento
"""
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
import logging

class ReferenceAnalyzer:
    def __init__(self, settings=None):
        self.logger = logging.getLogger(__name__)
        self.settings = settings
    
    def analyze_complete(self, elastic_df: pd.DataFrame, 
                        references_df: pd.DataFrame,
                        model_data: Dict[str, Any] = None,
                        labels_data: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        DEPRECATED: Método anterior que filtraba primero, mantenido para compatibilidad
        
        Realiza análisis completo de referencias con estado de entrenamiento
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            references_df: DataFrame con referencias
            model_data: Diccionario con datos del modelo
            labels_data: DataFrame con datos de etiquetas
            
        Returns:
            Dict con DataFrames de resultados
        """
        self.logger.warning("DEPRECATED: Se está usando el método antiguo analyze_complete. Considerar migrar a perform_complete_analysis")
        
        try:
            self.logger.info("Iniciando análisis completo de referencias (método deprecated)")
            
            # Paso 1: Filtrar referencias por apariciones
            self.logger.info("Filtrando referencias con mínimo 5 apariciones de cliente")
            filtered_references = self.filter_references_by_appearances(elastic_df, references_df)
            
            # Paso 2: Separar transacciones por tipo
            self.logger.info("Separando por transaction_metric")
            client_transactions = elastic_df[elastic_df['transaction_metric'] == 'client'].copy()
            manual_transactions = elastic_df[elastic_df['transaction_metric'] == 'manual'].copy()
            
            self.logger.info(f"Transacciones de cliente: {len(client_transactions)}")
            self.logger.info(f"Transacciones manuales: {len(manual_transactions)}")
            
            self.logger.info(f"Filtradas {len(filtered_references)} referencias de {len(references_df)} totales")
            self.logger.info("Referencias con transacciones de cliente >= 5")
            
            # Paso 3: Analizar estado de entrenamiento
            self.logger.info("Analizando estado de entrenamiento de referencias")
            training_analysis = self.analyze_training_status(
                elastic_df, filtered_references, model_data, labels_data
            )
            
            results = {
                'filtered_references': filtered_references,
                'client_transactions': client_transactions,
                'manual_transactions': manual_transactions,
                'training_analysis': training_analysis,
                'complete_analysis': training_analysis,  # Alias para compatibilidad con reportes
                'label_suggestions': pd.DataFrame(),  # DataFrame vacío para compatibilidad
                'consistency_analysis': pd.DataFrame(),  # DataFrame vacío para compatibilidad
                'untrained_references': pd.DataFrame()  # DataFrame vacío para compatibilidad
            }
            
            self.logger.info("Análisis completo de referencias finalizado exitosamente")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis completo: {str(e)}")
            raise

    def filter_references_by_appearances(self, elastic_df: pd.DataFrame,
                                       references_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra referencias que aparecen al menos 5 veces en transacciones de cliente
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            references_df: DataFrame con referencias
            
        Returns:
            DataFrame filtrado
        """
        try:
            # Verificar columnas disponibles
            self.logger.info(f"Usando 'selected_reference_code' de Elasticsearch para mapear con 'reference_code' de BD")
            
            # Debug: verificar valores únicos de transaction_metric
            unique_metrics = elastic_df['transaction_metric'].unique()
            self.logger.info(f"Valores únicos de transaction_metric: {unique_metrics}")
            
            # Verificar si existe la columna selected_reference_code en elastic_df
            if 'selected_reference_code' not in elastic_df.columns:
                raise ValueError("Columna 'selected_reference_code' no encontrada en datos de Elasticsearch")
            
            # Filtrar solo transacciones de cliente
            client_transactions = elastic_df[elastic_df['transaction_metric'] == 'client'].copy()
            self.logger.info(f"Transacciones encontradas con transaction_metric='client': {len(client_transactions)}")
            
            if len(client_transactions) == 0:
                # Intentar con otros valores posibles
                self.logger.warning("No se encontraron transacciones con 'client'. Intentando con otros valores...")
                # Probar con valores que contengan 'client' o similares
                possible_client_values = [val for val in unique_metrics if val and ('client' in val.lower() or 'user' in val.lower())]
                self.logger.info(f"Valores posibles para cliente: {possible_client_values}")
                
                if possible_client_values:
                    # Usar el primer valor que contenga 'client' o 'user'
                    client_value = possible_client_values[0]
                    client_transactions = elastic_df[elastic_df['transaction_metric'] == client_value].copy()
                    self.logger.info(f"Usando '{client_value}' como valor de cliente. Transacciones encontradas: {len(client_transactions)}")
            
            # Contar apariciones por selected_reference_code
            reference_counts = client_transactions.groupby('selected_reference_code').size().reset_index(name='count')
            
            # Filtrar referencias con al menos 5 apariciones
            valid_references = reference_counts[reference_counts['count'] >= 5]['selected_reference_code'].tolist()
            
            # Filtrar el DataFrame de referencias usando reference_code
            filtered_references = references_df[references_df['reference_code'].isin(valid_references)].copy()
            
            # Agregar conteos haciendo merge por reference_code
            filtered_references = filtered_references.merge(
                reference_counts.rename(columns={'selected_reference_code': 'reference_code'})[['reference_code', 'count']], 
                on='reference_code', 
                how='left'
            )
            filtered_references.rename(columns={'count': 'cliente_appearances'}, inplace=True)
            
            # Contar transacciones manuales para cada referencia
            manual_transactions = elastic_df[elastic_df['transaction_metric'] == 'manual'].copy()
            if len(manual_transactions) == 0:
                # Intentar encontrar valores para manuales - incluir WSREFERENCEIMAGEAC01 y WSPROCESSCOMPLETED
                possible_manual_values = [val for val in unique_metrics if val and (
                    'manual' in val.lower() or 
                    'WSREFERENCEIMAGEAC' in val or 
                    'image' in val.lower() or
                    'AC01' in val or
                    'WSPROCESSCOMPLETED' in val or
                    'PROCESSCOMPLETED' in val
                )]
                if possible_manual_values:
                    manual_value = possible_manual_values[0]
                    manual_transactions = elastic_df[elastic_df['transaction_metric'] == manual_value].copy()
                    self.logger.info(f"Usando '{manual_value}' como valor manual. Transacciones encontradas: {len(manual_transactions)}")
            
            manual_counts = manual_transactions.groupby('selected_reference_code').size().reset_index(name='manual_count')
            
            filtered_references = filtered_references.merge(
                manual_counts.rename(columns={'selected_reference_code': 'reference_code'})[['reference_code', 'manual_count']], 
                on='reference_code', 
                how='left'
            )
            filtered_references['manual_count'] = filtered_references['manual_count'].fillna(0)
            filtered_references['num_manuales'] = filtered_references['manual_count'].astype(int)
            
            return filtered_references
            
        except Exception as e:
            self.logger.error(f"Error filtrando referencias: {str(e)}")
            raise

    def analyze_training_status(self, elastic_df: pd.DataFrame,
                              filtered_references: pd.DataFrame,
                              model_data: Dict[str, Any] = None,
                              labels_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analiza el estado de entrenamiento de cada referencia
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            filtered_references: DataFrame con referencias filtradas
            model_data: Diccionario con datos del modelo
            labels_data: DataFrame con datos de etiquetas
            
        Returns:
            DataFrame con análisis de entrenamiento
        """
        try:
            analysis_results = []
            
            # Detectar valores dinámicos de transaction_metric igual que en filter_references_by_appearances
            unique_metrics = elastic_df['transaction_metric'].unique()
            self.logger.info(f"Detectando valores de transaction_metric para análisis: {unique_metrics}")
            
            # Determinar valor para transacciones de cliente
            client_metric = 'client'
            if 'client' not in unique_metrics:
                # Buscar valores que contengan 'user' o 'client'
                possible_client_values = [val for val in unique_metrics if val and ('client' in val.lower() or 'user' in val.lower())]
                if possible_client_values:
                    client_metric = possible_client_values[0]
                    self.logger.info(f"Usando '{client_metric}' como valor de cliente para análisis")
                else:
                    # Si no encontramos nada específico, usar el primer valor disponible
                    client_metric = unique_metrics[0] if len(unique_metrics) > 0 else 'client'
                    self.logger.info(f"Usando '{client_metric}' como valor por defecto para análisis")
            
            # Determinar valor para transacciones manuales
            manual_metric = 'manual'
            if 'manual' not in unique_metrics:
                # Buscar valores que contengan 'manual', 'WSREFERENCEIMAGEAC', 'IMAGE' o 'PROCESSCOMPLETED'
                possible_manual_values = [val for val in unique_metrics if val and (
                    'manual' in val.lower() or 
                    'WSREFERENCEIMAGEAC' in val or 
                    'image' in val.lower() or
                    'AC01' in val or
                    'WSPROCESSCOMPLETED' in val or
                    'PROCESSCOMPLETED' in val
                )]
                if possible_manual_values:
                    manual_metric = possible_manual_values[0]
                    self.logger.info(f"Usando '{manual_metric}' como valor manual para análisis")
            
            # Optimización: Convertir label_id en labels_data una sola vez fuera del bucle
            if labels_data is not None and not labels_data.empty and 'label_id' in labels_data.columns:
                labels_data = labels_data.copy()  # Hacer copia para evitar modificar el original
                labels_data['label_id'] = pd.to_numeric(labels_data['label_id'], errors='coerce').astype('Int64')
                self.logger.info(f"Tabla de labels cargada: {len(labels_data)} etiquetas disponibles")
            else:
                self.logger.warning("No hay datos de labels disponibles para el mapeo")
            
            # Mostrar algunas referencias que se van a procesar
            self.logger.info(f"Procesando {len(filtered_references)} referencias filtradas:")
            for i, (_, ref_row) in enumerate(filtered_references.head(5).iterrows()):
                self.logger.info(f"  {i+1}. {ref_row['reference_code']} - {ref_row['reference_name']} (label_id: {ref_row['label_id']})")
            if len(filtered_references) > 5:
                self.logger.info(f"  ... y {len(filtered_references) - 5} más")
            
            for _, ref_row in filtered_references.iterrows():
                reference_id = ref_row['reference_id']
                reference_code = ref_row.get('reference_code', 'N/A')
                
                # Filtrar transacciones para esta referencia usando reference_code
                ref_transactions = elastic_df[elastic_df['selected_reference_code'] == reference_code].copy()
                
                # Separar por tipo de transacción usando los valores detectados
                client_trans = ref_transactions[ref_transactions['transaction_metric'] == client_metric]
                manual_trans = ref_transactions[ref_transactions['transaction_metric'] == manual_metric]
                
                self.logger.debug(f"Referencia {reference_code}: {len(ref_transactions)} total, {len(client_trans)} cliente, {len(manual_trans)} manual")
                
                # Contar transacciones de cliente (consolidado)
                total_client = len(client_trans)
                
                # Contar trained/not_trained solo para determinar estado de entrenamiento
                if 'selected_reference_is_trained' in client_trans.columns:
                    client_trained = len(client_trans[client_trans['selected_reference_is_trained'] == True])
                    client_not_trained = len(client_trans[client_trans['selected_reference_is_trained'] == False])
                else:
                    # Si no hay columna de entrenamiento, asumir todo como no entrenado
                    if len(client_trans) > 0:
                        self.logger.warning(f"Columna 'selected_reference_is_trained' no encontrada para {reference_code}. Asumiendo todo como no entrenado.")
                    client_trained = 0
                    client_not_trained = len(client_trans)
                
                # Determinar estado final de entrenamiento
                if total_client > 0:
                    trained_percentage = client_trained / total_client
                    final_is_trained = trained_percentage >= 0.5  # Mayoría entrenada
                else:
                    final_is_trained = False
                
                # Usar solo la columna consolidada
                num_cliente = total_client
                trained_count = client_trained
                not_trained_count = client_not_trained
                
                # Convertir label_id de la referencia a un entero nullable para la comparación
                label_id_value = ref_row['label_id']
                if pd.isna(label_id_value) or label_id_value is None:
                    ref_label_id = None
                else:
                    try:
                        ref_label_id = int(float(label_id_value))
                    except (ValueError, TypeError):
                        ref_label_id = None

                # Verificar si el producto está en el modelo y obtener class_name
                is_in_dataset = False
                producto_trained = None
                
                if model_data and ref_label_id is not None:
                    
                    self.logger.debug(f"Buscando label_id {ref_label_id} en modelo para referencia {reference_code}")
                    model_structure = model_data.get('model_data')
                    
                    classes_list = None
                    if isinstance(model_structure, str):
                        try:
                            model_structure = json.loads(model_structure)
                            self.logger.debug(f"JSON decodificado exitosamente. Tipo: {type(model_structure)}")
                        except json.JSONDecodeError:
                            self.logger.warning("No se pudo decodificar el JSON de model_data")
                            model_structure = None

                    if isinstance(model_structure, list):
                        classes_list = model_structure
                        self.logger.debug(f"model_data es una lista con {len(classes_list)} elementos")
                    elif isinstance(model_structure, dict) and 'classes' in model_structure:
                        classes_list = model_structure['classes']
                        self.logger.debug(f"model_data es un dict con 'classes' que tiene {len(classes_list)} elementos")
                    else:
                        self.logger.debug(f"model_data no tiene el formato esperado. Tipo: {type(model_structure)}")

                    if classes_list:
                        for class_info in classes_list:
                            # La estructura real tiene 'trained_labels' que es una lista de IDs
                            trained_labels = class_info.get('trained_labels', [])
                            
                            if ref_label_id in trained_labels:
                                is_in_dataset = True
                                producto_trained = class_info.get('index_name', '')
                                self.logger.debug(f"¡Coincidencia encontrada! label_id {ref_label_id} en clase '{producto_trained}'")
                                break
                        
                        if not is_in_dataset:
                            self.logger.debug(f"No se encontró label_id {ref_label_id} en {len(classes_list)} clases del modelo")
                    else:
                        self.logger.debug("No hay lista de clases disponible en model_data")
                else:
                    if model_data is None:
                        self.logger.debug("model_data es None")
                    if ref_label_id is None:
                        self.logger.debug(f"ref_label_id es None para referencia {reference_code}")
                
                # Obtener label_name
                label_name = None
                if labels_data is not None and not labels_data.empty and ref_label_id is not None:
                    # La conversión ya se hizo fuera del bucle
                    label_match = labels_data[labels_data['label_id'] == ref_label_id]
                    if not label_match.empty:
                        label_name = label_match.iloc[0].get('label_name')
                        self.logger.info(f"[OK] Encontrado label_name '{label_name}' para label_id {ref_label_id} (referencia {reference_code})")
                    else:
                        # Label ID no encontrado en la tabla de labels
                        self.logger.warning(f"[X] Label ID {ref_label_id} NO ENCONTRADO en tabla de labels para referencia {reference_code}")
                        # Debug adicional
                        available_ids = sorted(labels_data['label_id'].unique())[:10]  # Primeros 10 para debug
                        self.logger.warning(f"Algunos label_ids disponibles: {available_ids}")
                        label_name = f"LABEL_ID_{ref_label_id}_NOT_FOUND"  # Valor indicativo
                elif ref_label_id is None:
                    self.logger.info(f"[INFO] Referencia {reference_code} no tiene label_id asignado")
                    label_name = "NO_LABEL_ASSIGNED"  # Valor indicativo para referencias sin label_id
                
                # Log de debug para verificar datos
                if num_cliente == 0:
                    self.logger.debug(f"Referencia {reference_code}: Sin transacciones de cliente. Total transacciones encontradas: {len(ref_transactions)}")
                
                analysis_results.append({
                    'reference_id': ref_row['reference_id'],
                    'reference_code': reference_code,
                    'reference_name': ref_row['reference_name'],
                    'label_id': ref_row['label_id'],
                    'label_name': label_name,
                    'num_cliente': num_cliente,  # Columna consolidada
                    'num_manuales': len(manual_trans),  # Usar el conteo actual de transacciones manuales
                    'is_trained_elastic': final_is_trained,
                    'is_in_dataset': is_in_dataset,
                    'producto_trained': producto_trained,
                    'trained_count': trained_count,
                    'not_trained_count': not_trained_count,
                    'cliente_appearances': ref_row.get('cliente_appearances', 0)
                })
            
            result_df = pd.DataFrame(analysis_results)
            self.logger.info(f"Análisis de entrenamiento completado para {len(result_df)} referencias")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error analizando estado de entrenamiento: {str(e)}")
            raise

    def get_reference_stats(self, elastic_df: pd.DataFrame, reference_code: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de una referencia específica
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            reference_code: Código de la referencia
            
        Returns:
            Dict con estadísticas
        """
        try:
            ref_data = elastic_df[elastic_df['selected_reference_code'] == reference_code]
            
            if ref_data.empty:
                return {
                    'reference_code': reference_code,
                    'total_transactions': 0,
                    'client_transactions': 0,
                    'manual_transactions': 0,
                    'trained_transactions': 0,
                    'not_trained_transactions': 0,
                    'training_percentage': 0.0
                }
            
            total_transactions = len(ref_data)
            client_trans = ref_data[ref_data['transaction_metric'] == 'client']
            manual_trans = ref_data[ref_data['transaction_metric'] == 'manual']
            
            client_count = len(client_trans)
            manual_count = len(manual_trans)
            
            # Usar selected_reference_is_trained si está disponible
            if 'selected_reference_is_trained' in ref_data.columns:
                trained_count = len(ref_data[ref_data['selected_reference_is_trained'] == True])
                not_trained_count = len(ref_data[ref_data['selected_reference_is_trained'] == False])
            else:
                trained_count = 0
                not_trained_count = total_transactions
            
            training_percentage = (trained_count / total_transactions * 100) if total_transactions > 0 else 0
            
            return {
                'reference_code': reference_code,
                'total_transactions': total_transactions,
                'client_transactions': client_count,
                'manual_transactions': manual_count,
                'trained_transactions': trained_count,
                'not_trained_transactions': not_trained_count,
                'training_percentage': round(training_percentage, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas de referencia {reference_code}: {str(e)}")
            raise

    def get_summary_stats(self, analysis_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera estadísticas resumen del análisis
        
        Args:
            analysis_results: DataFrame con resultados del análisis
            
        Returns:
            Dict con estadísticas resumen
        """
        try:
            total_references = len(analysis_results)
            
            if total_references == 0:
                return {
                    'total_references': 0,
                    'trained_references': 0,
                    'not_trained_references': 0,
                    'training_percentage': 0.0,
                    'in_model_references': 0,
                    'not_in_model_references': 0,
                    'model_coverage_percentage': 0.0,
                    'total_client_transactions': 0,
                    'total_manual_transactions': 0,
                    'total_trained_transactions': 0
                }
            
            # Verificar si existe la columna is_trained_elastic
            if 'is_trained_elastic' in analysis_results.columns:
                trained_references = len(analysis_results[analysis_results['is_trained_elastic'] == True])
            else:
                trained_references = 0
                
            not_trained_references = total_references - trained_references
            
            # Referencias en modelo
            if 'is_in_dataset' in analysis_results.columns:
                in_model_references = len(analysis_results[analysis_results['is_in_dataset'] == True])
            else:
                in_model_references = 0
                
            not_in_model_references = total_references - in_model_references
            
            # Conteos de transacciones
            total_client_transactions = 0
            total_manual_transactions = 0
            total_trained_transactions = 0
            
            # Usar la nueva columna num_cliente consolidada
            if 'num_cliente' in analysis_results.columns:
                total_client_transactions = analysis_results['num_cliente'].sum()
            
            # Para trained_transactions, usar trained_count si está disponible
            if 'trained_count' in analysis_results.columns:
                total_trained_transactions = analysis_results['trained_count'].sum()
            
            if 'num_manuales' in analysis_results.columns:
                total_manual_transactions = analysis_results['num_manuales'].sum()
            
            training_percentage = (trained_references / total_references * 100) if total_references > 0 else 0
            model_coverage_percentage = (in_model_references / total_references * 100) if total_references > 0 else 0
            
            return {
                'total_references': total_references,
                'total_filtered_references': total_references,  # Alias para compatibilidad
                'trained_references': trained_references,
                'not_trained_references': not_trained_references,
                'untrained_count': not_trained_references,  # Alias para compatibilidad
                'training_percentage': round(training_percentage, 2),
                'in_model_references': in_model_references,
                'not_in_model_references': not_in_model_references,
                'model_coverage_percentage': round(model_coverage_percentage, 2),
                'total_client_transactions': int(total_client_transactions),
                'total_manual_transactions': int(total_manual_transactions),
                'total_trained_transactions': int(total_trained_transactions),
                'unassigned_count': 0,  # Para compatibilidad con reportes
                'high_confidence_suggestions': 0,  # Para compatibilidad con reportes
                'low_confidence_suggestions': 0,  # Para compatibilidad con reportes (campo faltante)
                'total_unassigned': 0,  # Para compatibilidad con reportes
                'consistent_references': 0  # Para compatibilidad con reportes (campo faltante)
            }
            
        except Exception as e:
            self.logger.error(f"Error generando estadísticas resumen: {str(e)}")
            raise

    def enrich_references_with_labels(self, references_df: pd.DataFrame, labels_data: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece TODAS las referencias con información de labels
        
        Args:
            references_df: DataFrame con referencias
            labels_data: DataFrame con datos de etiquetas
            
        Returns:
            DataFrame enriquecido con label_name
        """
        try:
            self.logger.info("Enriqueciendo referencias con información de labels")
            
            # Hacer copia para no modificar el original
            enriched_df = references_df.copy()
            
            if labels_data is not None and not labels_data.empty and 'label_id' in labels_data.columns:
                # Preparar datos de labels
                labels_clean = labels_data.copy()
                labels_clean['label_id'] = pd.to_numeric(labels_clean['label_id'], errors='coerce').astype('Int64')
                
                # Preparar label_id en references para merge
                enriched_df['label_id_numeric'] = pd.to_numeric(enriched_df['label_id'], errors='coerce').astype('Int64')
                
                # Hacer merge para obtener label_name
                enriched_df = enriched_df.merge(
                    labels_clean[['label_id', 'label_name']],
                    left_on='label_id_numeric',
                    right_on='label_id',
                    how='left',
                    suffixes=('', '_from_labels')
                )
                
                # Limpiar columnas temporales
                enriched_df = enriched_df.drop(columns=['label_id_numeric', 'label_id_from_labels'], errors='ignore')
                
                # Contar éxitos
                with_label_name = enriched_df['label_name'].notna().sum()
                total_refs = len(enriched_df)
                
                self.logger.info(f"Referencias enriquecidas con labels: {with_label_name}/{total_refs}")
            else:
                self.logger.warning("No hay datos de labels disponibles")
                enriched_df['label_name'] = None
            
            return enriched_df
            
        except Exception as e:
            self.logger.error(f"Error enriqueciendo referencias con labels: {str(e)}")
            raise

    def enrich_references_with_model(self, references_df: pd.DataFrame, model_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Enriquece TODAS las referencias con información del modelo
        
        Args:
            references_df: DataFrame con referencias (ya enriquecido con labels)
            model_data: Diccionario con datos del modelo
            
        Returns:
            DataFrame enriquecido con is_in_dataset y producto_trained
        """
        try:
            self.logger.info("Enriqueciendo referencias con información del modelo")
            
            # Hacer copia para no modificar el original
            enriched_df = references_df.copy()
            
            # Inicializar columnas del modelo
            enriched_df['is_in_dataset'] = False
            enriched_df['producto_trained'] = None
            
            if model_data:
                # Procesar estructura del modelo
                model_structure = model_data.get('model_data')
                
                classes_list = None
                if isinstance(model_structure, str):
                    try:
                        model_structure = json.loads(model_structure)
                        self.logger.debug(f"JSON del modelo decodificado exitosamente")
                    except json.JSONDecodeError:
                        self.logger.warning("No se pudo decodificar el JSON del modelo")
                        model_structure = None

                if isinstance(model_structure, list):
                    classes_list = model_structure
                elif isinstance(model_structure, dict) and 'classes' in model_structure:
                    classes_list = model_structure['classes']
                
                if classes_list:
                    self.logger.info(f"Procesando {len(classes_list)} clases del modelo")
                    
                    # Crear mapeo de label_id -> clase del modelo
                    label_to_class = {}
                    for class_info in classes_list:
                        trained_labels = class_info.get('trained_labels', [])
                        class_name = class_info.get('index_name', '')
                        
                        for label_id in trained_labels:
                            label_to_class[label_id] = class_name
                    
                    # Aplicar mapeo a todas las referencias
                    for idx, ref_row in enriched_df.iterrows():
                        label_id_value = ref_row['label_id']
                        if pd.notna(label_id_value):
                            try:
                                ref_label_id = int(float(label_id_value))
                                if ref_label_id in label_to_class:
                                    enriched_df.at[idx, 'is_in_dataset'] = True
                                    enriched_df.at[idx, 'producto_trained'] = label_to_class[ref_label_id]
                            except (ValueError, TypeError):
                                continue
                    
                    # Contar éxitos
                    in_model_count = enriched_df['is_in_dataset'].sum()
                    total_refs = len(enriched_df)
                    
                    self.logger.info(f"Referencias encontradas en modelo: {in_model_count}/{total_refs}")
                else:
                    self.logger.warning("No se encontraron clases en los datos del modelo")
            else:
                self.logger.warning("No hay datos del modelo disponibles")
            
            return enriched_df
            
        except Exception as e:
            self.logger.error(f"Error enriqueciendo referencias con modelo: {str(e)}")
            raise

    def enrich_references_with_elastic(self, references_df: pd.DataFrame, elastic_df: pd.DataFrame, labels_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Enriquece TODAS las referencias con estadísticas de Elasticsearch
        Incluye análisis de top 3 result_index más frecuentes
        
        Args:
            references_df: DataFrame con referencias (ya enriquecido con labels y modelo)
            elastic_df: DataFrame con datos de Elasticsearch
            labels_data: DataFrame con datos de etiquetas para obtener nombres de result_index
            
        Returns:
            DataFrame enriquecido con estadísticas de transacciones y top 3 result_index
        """
        try:
            self.logger.info("Enriqueciendo referencias con estadísticas de Elasticsearch")
            
            # Hacer copia para no modificar el original
            enriched_df = references_df.copy()
            
            # Inicializar columnas de estadísticas básicas
            enriched_df['num_cliente'] = 0  # Solo la columna consolidada
            enriched_df['num_manuales'] = 0
            enriched_df['is_trained_elastic'] = False
            enriched_df['pct_ok_sistema'] = 0.0  # Porcentaje de acierto del sistema
            enriched_df['pct_ok_modelo'] = 0.0   # Porcentaje de acierto del modelo
            
            # Inicializar columnas para revisión de imágenes
            enriched_df['revisar_imagenes'] = 'no'
            
            # Inicializar columnas para top 3 result_index
            for rank in range(1, 4):  # Top 1, 2, 3
                enriched_df[f'top{rank}_result_index'] = None
                enriched_df[f'top{rank}_result_label_name'] = None
                enriched_df[f'top{rank}_%pesadas'] = 0.0
                enriched_df[f'top{rank}_pct_ok'] = 0.0
            
            self.logger.info(f"Columnas inicializadas para top 3 result_index. Total columnas: {len(enriched_df.columns)}")
            
            if elastic_df is not None and not elastic_df.empty:
                # Detectar valores de transaction_metric
                unique_metrics = elastic_df['transaction_metric'].unique()
                self.logger.info(f"Valores de transaction_metric en Elasticsearch: {unique_metrics}")
                
                # Determinar valor para transacciones de cliente
                client_metric = 'client'
                if 'client' not in unique_metrics:
                    possible_client_values = [val for val in unique_metrics if val and ('client' in val.lower() or 'user' in val.lower())]
                    if possible_client_values:
                        client_metric = possible_client_values[0]
                        self.logger.info(f"Usando '{client_metric}' como valor de cliente")
                
                # Determinar valor para transacciones manuales
                manual_metric = 'manual'
                if 'manual' not in unique_metrics:
                    # Buscar valores que contengan 'manual', 'WSREFERENCEIMAGEAC', 'IMAGE' o 'PROCESSCOMPLETED'
                    possible_manual_values = [val for val in unique_metrics if val and (
                        'manual' in val.lower() or 
                        'WSREFERENCEIMAGEAC' in val or 
                        'image' in val.lower() or
                        'AC01' in val or
                        'WSPROCESSCOMPLETED' in val or
                        'PROCESSCOMPLETED' in val
                    )]
                    if possible_manual_values:
                        manual_metric = possible_manual_values[0]
                        self.logger.info(f"Usando '{manual_metric}' como valor manual")
                
                # Procesar cada referencia
                refs_with_data = 0
                for idx, ref_row in enriched_df.iterrows():
                    reference_code = ref_row.get('reference_code')
                    reference_name = ref_row.get('reference_name')  # También necesitamos el nombre
                    if pd.isna(reference_code):
                        continue
                    
                    # Filtrar transacciones de esta referencia
                    # Primero por código (para transacciones de cliente principalmente)
                    ref_transactions = elastic_df[elastic_df['selected_reference_code'] == reference_code].copy()
                    
                    # Luego agregar transacciones por nombre (para transacciones manuales)
                    # Solo si la columna selected_reference existe y el nombre de referencia no es nulo
                    if pd.notna(reference_name) and 'selected_reference' in elastic_df.columns:
                        ref_transactions_by_name = elastic_df[elastic_df['selected_reference'] == reference_name].copy()
                        if not ref_transactions_by_name.empty:
                            # Concatenar usando append o concat sin drop_duplicates por ahora
                            ref_transactions = pd.concat([ref_transactions, ref_transactions_by_name], ignore_index=True)
                    
                    if not ref_transactions.empty:
                        refs_with_data += 1
                        
                        # Separar por tipo de transacción
                        client_trans = ref_transactions[ref_transactions['transaction_metric'] == client_metric]
                        manual_trans = ref_transactions[ref_transactions['transaction_metric'] == manual_metric]
                        
                        # Contar trained/not_trained para determinar estado de entrenamiento
                        if 'selected_reference_is_trained' in client_trans.columns:
                            client_trained = len(client_trans[client_trans['selected_reference_is_trained'] == True])
                            client_not_trained = len(client_trans[client_trans['selected_reference_is_trained'] == False])
                        else:
                            client_trained = 0
                            client_not_trained = len(client_trans)
                        
                        # Actualizar valores - solo columna consolidada
                        enriched_df.at[idx, 'num_cliente'] = len(client_trans)  # Total de transacciones de cliente
                        enriched_df.at[idx, 'num_manuales'] = len(manual_trans)
                        
                        # Determinar si está entrenada (mayoría)
                        total_client = len(client_trans)
                        if total_client > 0:
                            trained_percentage = client_trained / total_client
                            enriched_df.at[idx, 'is_trained_elastic'] = trained_percentage >= 0.5
                            
                            # Calcular porcentajes de acierto SOLO con transacciones de cliente
                            # %Ok del sistema (basado en columna 'Ok')
                            if 'Ok' in client_trans.columns:
                                ok_sistema_count = (client_trans['Ok'] == 1).sum()
                                pct_ok_sistema = (ok_sistema_count / total_client * 100)
                                enriched_df.at[idx, 'pct_ok_sistema'] = round(pct_ok_sistema, 2)
                            
                            # %Ok del modelo (basado en columna 'Ok_modelo')
                            if 'Ok_modelo' in client_trans.columns:
                                ok_modelo_count = (client_trans['Ok_modelo'] == 1).sum()
                                pct_ok_modelo = (ok_modelo_count / total_client * 100)
                                enriched_df.at[idx, 'pct_ok_modelo'] = round(pct_ok_modelo, 2)
                            
                            # Calcular top 3 result_index más frecuentes
                            self.logger.debug(f"Calculando top 3 result_index para referencia {reference_code}")
                            top3_stats = self._get_top3_result_indexes(client_trans, labels_data)
                            self.logger.debug(f"Top 3 stats obtenidas: {top3_stats}")
                            
                            # Actualizar las columnas con los datos del top 3
                            for key, value in top3_stats.items():
                                if key in enriched_df.columns:
                                    enriched_df.at[idx, key] = value
                                    self.logger.debug(f"Actualizada columna {key} = {value}")
                                else:
                                    self.logger.warning(f"Columna {key} no existe en DataFrame")
                
                self.logger.info(f"Referencias con datos en Elasticsearch: {refs_with_data}/{len(enriched_df)}")
                
                # Log para verificar datos de top 3
                top1_filled = (enriched_df['top1_result_index'].notna()).sum()
                self.logger.info(f"Referencias con top1_result_index: {top1_filled}/{len(enriched_df)}")
            else:
                self.logger.warning("No hay datos de Elasticsearch disponibles")
            
            return enriched_df
            
        except Exception as e:
            self.logger.error(f"Error enriqueciendo referencias con Elasticsearch: {str(e)}")
            raise

    def _get_top3_result_indexes(self, client_transactions: pd.DataFrame, labels_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Obtiene los 3 result_index más frecuentes agrupando por selected_reference_name, result_index y result_label_name
        
        Args:
            client_transactions: DataFrame con transacciones de cliente para una referencia específica
            labels_data: DataFrame con datos de etiquetas (no se usa, result_label_name viene directamente)
            
        Returns:
            Diccionario con información de los top 3 result_index
        """
        try:
            # Verificar que tenemos las columnas necesarias
            required_cols = ['result_index', 'result_label_name', 'selected_reference_name', 'Ok']
            if client_transactions.empty or not all(col in client_transactions.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in client_transactions.columns]
                if missing_cols:
                    self.logger.debug(f"Columnas faltantes para top3 analysis: {missing_cols}")
                return {}
            
            # Agrupar por selected_reference_name, result_index y result_label_name
            # Calcular número de registros y % OK para cada grupo
            grouped = client_transactions.groupby(['selected_reference_name', 'result_index', 'result_label_name']).agg({
                'transaction_id': 'count',  # Número de registros
                'Ok': ['count', 'sum']       # Total y suma de OK para calcular %
            }).reset_index()
            
            # Aplanar las columnas multi-nivel
            grouped.columns = ['selected_reference_name', 'result_index', 'result_label_name', 'num_registros', 'total_ok_count', 'ok_sum']
            
            # Calcular % OK
            grouped['pct_ok'] = (grouped['ok_sum'] / grouped['total_ok_count'] * 100).fillna(0)
            
            # Ordenar por número de registros (descendente) y tomar los top 3
            top3_groups = grouped.sort_values('num_registros', ascending=False).head(3)
            
            if top3_groups.empty:
                return {}
            
            result_stats = {}
            
            # Llenar la información para cada uno de los top 3
            for rank, (_, row) in enumerate(top3_groups.iterrows(), 1):
                result_stats[f'top{rank}_result_index'] = int(row['result_index'])
                result_stats[f'top{rank}_result_label_name'] = str(row['result_label_name']) if pd.notna(row['result_label_name']) else "N/A"
                result_stats[f'top{rank}_%pesadas'] = round((row['num_registros'] / len(client_transactions) * 100), 2)
                result_stats[f'top{rank}_pct_ok'] = round(row['pct_ok'], 2)
                
                # Log para debug
                self.logger.debug(f"Top{rank}: result_index={row['result_index']}, "
                                f"result_label_name={row['result_label_name']}, "
                                f"num_registros={row['num_registros']}, "
                                f"pct_ok={row['pct_ok']:.2f}%")
            
            return result_stats
            
        except Exception as e:
            self.logger.error(f"Error calculando top 3 result_index: {e}")
            return {}

    def perform_complete_analysis(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza el análisis completo siguiendo la nueva arquitectura:
        1. Enriquecer referencias con labels
        2. Enriquecer referencias con modelo
        3. Enriquecer referencias con estadísticas de Elasticsearch
        
        Args:
            extracted_data: Diccionario con todos los datos extraídos
            
        Returns:
            Diccionario con resultados del análisis
        """
        try:
            self.logger.info("Iniciando análisis completo con nueva arquitectura")
            
            # Extraer componentes de los datos
            elastic_df = extracted_data.get('elastic_data')
            references_df = extracted_data.get('references_data')
            model_data = extracted_data.get('model_data')
            labels_data = extracted_data.get('labels_data')
            
            if references_df is None or references_df.empty:
                raise ValueError("No hay datos de referencias para procesar")
            
            self.logger.info(f"Procesando {len(references_df)} referencias totales")
            
            # Paso 1: Enriquecer con labels
            enriched_refs = self.enrich_references_with_labels(references_df, labels_data)
            
            # Paso 2: Enriquecer con modelo
            enriched_refs = self.enrich_references_with_model(enriched_refs, model_data)
            
            # Paso 3: Enriquecer con estadísticas de Elasticsearch
            enriched_refs = self.enrich_references_with_elastic(enriched_refs, elastic_df, labels_data)
            
            # Filtrar referencias que aparecen en Elasticsearch para compatibilidad
            if elastic_df is not None and not elastic_df.empty:
                filtered_references = enriched_refs[
                    enriched_refs['num_cliente'] >= 5
                ].copy()
                self.logger.info(f"Referencias con >= 5 apariciones de cliente: {len(filtered_references)}")
            else:
                filtered_references = pd.DataFrame()
            
            # Preparar datos de transacciones para compatibilidad
            client_transactions = pd.DataFrame()
            manual_transactions = pd.DataFrame()
            
            if elastic_df is not None and not elastic_df.empty:
                unique_metrics = elastic_df['transaction_metric'].unique()
                client_metric = 'client'
                if 'client' not in unique_metrics:
                    possible_client_values = [val for val in unique_metrics if val and ('client' in val.lower() or 'user' in val.lower())]
                    if possible_client_values:
                        client_metric = possible_client_values[0]
                
                manual_metric = 'manual'
                if 'manual' not in unique_metrics:
                    # Buscar valores que contengan 'manual', 'WSREFERENCEIMAGEAC', 'IMAGE' o 'PROCESSCOMPLETED'
                    possible_manual_values = [val for val in unique_metrics if val and (
                        'manual' in val.lower() or 
                        'WSREFERENCEIMAGEAC' in val or 
                        'image' in val.lower() or
                        'AC01' in val or
                        'WSPROCESSCOMPLETED' in val or
                        'PROCESSCOMPLETED' in val
                    )]
                    if possible_manual_values:
                        manual_metric = possible_manual_values[0]
                        self.logger.info(f"Usando '{manual_metric}' como valor manual")
                
                client_transactions = elastic_df[elastic_df['transaction_metric'] == client_metric].copy()
                manual_transactions = elastic_df[elastic_df['transaction_metric'] == manual_metric].copy()
            
            # Preparar resultados
            results = {
                'filtered_references': filtered_references,
                'client_transactions': client_transactions,
                'manual_transactions': manual_transactions,
                'training_analysis': enriched_refs,  # TODAS las referencias enriquecidas
                'complete_analysis': enriched_refs,  # Alias para compatibilidad
                'label_suggestions': pd.DataFrame(),  # DataFrame vacío para compatibilidad
                'consistency_analysis': pd.DataFrame(),  # DataFrame vacío para compatibilidad
                'untrained_references': pd.DataFrame()  # DataFrame vacío para compatibilidad
            }
            
            # Agregar estadísticas resumen
            summary_stats = self.get_summary_stats(enriched_refs)
            results['summary_stats'] = summary_stats
            
            # Análisis de devices
            devices_analysis = self.analyze_devices(elastic_df, labels_data)
            results['devices_analysis'] = devices_analysis
            
            self.logger.info("Análisis completo finalizado exitosamente")
            return results
            
        except Exception as e:
            self.logger.error(f"Error en análisis completo: {str(e)}")
            raise

    def analyze_devices(self, elastic_df: pd.DataFrame, labels_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Analiza las estadísticas por device_id
        
        Args:
            elastic_df: DataFrame con datos de Elasticsearch
            labels_data: DataFrame con datos de etiquetas para mapping
            
        Returns:
            DataFrame con estadísticas por device
        """
        try:
            if elastic_df is None or elastic_df.empty:
                self.logger.warning("No hay datos de Elasticsearch para análisis de devices")
                return pd.DataFrame()
            
            self.logger.info(f"Analizando devices desde {len(elastic_df)} transacciones")
            
            # Detectar valores de transaction_metric igual que en otras funciones
            unique_metrics = elastic_df['transaction_metric'].unique()
            self.logger.info(f"Valores de transaction_metric disponibles: {unique_metrics}")
            
            # Determinar valor para transacciones de cliente
            client_metric = 'client'
            if 'client' not in unique_metrics:
                possible_client_values = [val for val in unique_metrics if val and ('client' in val.lower() or 'user' in val.lower())]
                if possible_client_values:
                    client_metric = possible_client_values[0]
                    self.logger.info(f"Usando '{client_metric}' como valor de cliente para devices")
            
            # Determinar valor para transacciones manuales
            manual_metric = 'manual'
            if 'manual' not in unique_metrics:
                possible_manual_values = [val for val in unique_metrics if val and (
                    'manual' in val.lower() or 
                    'WSREFERENCEIMAGEAC' in val or 
                    'image' in val.lower() or
                    'AC01' in val or
                    'WSPROCESSCOMPLETED' in val or
                    'PROCESSCOMPLETED' in val
                )]
                if possible_manual_values:
                    manual_metric = possible_manual_values[0]
                    self.logger.info(f"Usando '{manual_metric}' como valor manual para devices")
            
            # Agrupar por device_id y calcular estadísticas
            devices_stats = []
            
            for device_id in elastic_df['device_id'].unique():
                if pd.isna(device_id):
                    continue
                    
                device_data = elastic_df[elastic_df['device_id'] == device_id].copy()
                
                # Separar por tipo de transacción usando valores detectados
                client_data = device_data[device_data['transaction_metric'] == client_metric]
                manual_data = device_data[device_data['transaction_metric'] == manual_metric]
                
                # Calcular estadísticas básicas
                num_cliente = len(client_data)
                num_manual = len(manual_data)
                
                # Calcular %Ok para transacciones de cliente
                pct_ok_sistema = 0.0
                if num_cliente > 0:
                    ok_count = len(client_data[client_data['Ok'] == True])
                    pct_ok_sistema = (ok_count / num_cliente) * 100
                
                # Calcular %Ok_modelo solo para referencias entrenadas
                pct_ok_modelo = 0.0
                if num_cliente > 0:
                    # Para %ok_modelo, usar solo transacciones donde selected_reference_is_trained es True
                    if 'selected_reference_is_trained' in client_data.columns:
                        trained_transactions = client_data[client_data['selected_reference_is_trained'] == True]
                        
                        if len(trained_transactions) > 0 and 'Ok_modelo' in trained_transactions.columns:
                            ok_modelo_count = len(trained_transactions[trained_transactions['Ok_modelo'] == True])
                            pct_ok_modelo = (ok_modelo_count / len(trained_transactions)) * 100
                
                # Obtener nombre del device si está disponible
                device_name = device_data['device_name'].iloc[0] if 'device_name' in device_data.columns else device_id
                
                devices_stats.append({
                    'device_id': device_id,
                    'device_name': device_name,
                    'num_cliente': num_cliente,
                    'num_manual': num_manual,
                    'pct_ok_sistema': round(pct_ok_sistema, 1),
                    'pct_ok_modelo': round(pct_ok_modelo, 1),
                    'revisar_imagenes': 'no'  # Por defecto 'no'
                })
            
            devices_df = pd.DataFrame(devices_stats)
            
            # Ordenar por número de transacciones de cliente (descendente)
            devices_df = devices_df.sort_values('num_cliente', ascending=False)
            
            self.logger.info(f"Análisis de devices completado: {len(devices_df)} devices analizados")
            
            return devices_df
            
        except Exception as e:
            self.logger.error(f"Error en análisis de devices: {e}")
            return pd.DataFrame()
