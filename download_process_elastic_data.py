"""
Download and Process Elasticsearch Data with Enhanced Analytics
Extends the basic download functionality with advanced data processing features
"""
import argparse
import os
import logging
import shutil
import pandas as pd
import numpy as np
from collections import Counter
from utils.elastic_utils import (
    setup_logging, load_config, connect_es, download_data, convert_to_dataframe, print_index_mapping
)
from utils.sharepoint_utils import parse_sharepoint_url, upload_file_to_sharepoint_app
from utils.bbdd_utils import add_image_columns_to_dataframe, test_database_connection

# Verificar que openpyxl esté disponible para Excel
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    print("ADVERTENCIA: openpyxl no está instalado. Instale con: pip install openpyxl")
    EXCEL_AVAILABLE = False

class ElasticDataProcessor:
    """Enhanced data processor for Elasticsearch transaction data"""
    
    def __init__(self, logger=None, parejas_validas=None, todas_las_parejas=None, descripciones_productos=None, encendidos_data=None):
        self.logger = logger or logging.getLogger()
        self.lista_larga_threshold = 6  # Configurable threshold for "lista_larga"
        self.parejas_validas = parejas_validas
        self.todas_las_parejas = todas_las_parejas
        self.descripciones_productos = descripciones_productos
        self.encendidos_data = encendidos_data
        
    def add_predict_codes_analytics(self, df):
        """Add analytics columns related to predicted_codes list"""
        self.logger.info("Adding predicted_codes analytics...")
        
        # Use the correct column name (predicted_codes, not predict_codes)
        codes_column = 'predicted_codes'
        if codes_column not in df.columns:
            self.logger.error(f"Column '{codes_column}' not found. Available columns: {list(df.columns)}")
            return df
        
        # num_candidatos: Number of elements in predicted_codes list
        df['num_candidatos'] = df[codes_column].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        
        # lista_larga: 1 if list has more than threshold elements, 0 otherwise
        df['lista_larga'] = (df['num_candidatos'] > self.lista_larga_threshold).astype(int)
        
        # lista_un_candidato: 1 if list has exactly one element, 0 otherwise
        df['lista_un_candidato'] = (df['num_candidatos'] == 1).astype(int)
        
        self.logger.info(f"Added predicted_codes analytics. Lista larga threshold: {self.lista_larga_threshold}")
        return df
    
    def add_topk_classification(self, df):
        """Add top_k classification columns"""
        self.logger.info("Adding top_k classification...")
        
        # Check if topk column exists (might be 'topk' or 'top_k')
        topk_column = None
        if 'top_k' in df.columns:
            topk_column = 'top_k'
        elif 'topk' in df.columns:
            topk_column = 'topk'
        else:
            self.logger.error(f"Neither 'top_k' nor 'topk' found in columns: {list(df.columns)}")
            # Create dummy columns and return
            df['pos_candidato_elegido'] = ''
            df['top_k_0'] = 0
            df['top_k_medio'] = 0
            df['top_k_alto'] = 0
            return df
        
        # pos_candidato_elegido: Classification based on top_k value
        def classify_topk(top_k):
            if pd.isna(top_k) or top_k == -1:
                return 'NO_CLASIFICADO'  # Use a clear label instead of empty string
            if top_k == 0:
                return 'top_k_0'
            elif 1 <= top_k <= 4:
                return 'top_k_medio'
            else:
                return 'top_k_alto'
        
        df['pos_candidato_elegido'] = df[topk_column].apply(classify_topk)
        
        # Create binary columns for each classification
        df['top_k_0'] = (df['pos_candidato_elegido'] == 'top_k_0').astype(int)
        df['top_k_medio'] = (df['pos_candidato_elegido'] == 'top_k_medio').astype(int)
        df['top_k_alto'] = (df['pos_candidato_elegido'] == 'top_k_alto').astype(int)
        
        self.logger.info(f"Added top_k classification columns using column '{topk_column}'")
        return df
    
    def calculate_expected_model_accuracy(self, df):
        """Calculate expected model accuracy based on deployment/product performance"""
        self.logger.info("Calculating expected model accuracy...")
        
        # Filter data where model is trained using selected_reference_is_trained
        trained_data = df[df['selected_reference_is_trained'] == True].copy()
        
        if len(trained_data) == 0:
            self.logger.warning("No trained model data found for accuracy calculation")
            df['pct_ok_modelo_previsto'] = np.nan
            return df, pd.DataFrame()
        
        # Calculate model accuracy by deployment and product
        # Use Ok_modelo (which is based on Result_modelo == "SUCCESS")
        accuracy_stats = trained_data.groupby(['deployment_id', 'selected_reference_code', 'selected_reference_name']).agg({
            'Ok_modelo': ['count', 'sum']
        }).round(4)
        
        # Flatten column names
        accuracy_stats.columns = ['total_transactions', 'successful_transactions']
        accuracy_stats['pct_ok_modelo'] = (accuracy_stats['successful_transactions'] / accuracy_stats['total_transactions']).round(4)
        accuracy_stats = accuracy_stats.reset_index()
        
        # Filter out products with very few transactions to avoid noise
        min_transactions = 10
        accuracy_stats = accuracy_stats[accuracy_stats['total_transactions'] >= min_transactions]
        
        self.logger.info(f"Generated accuracy stats for {len(accuracy_stats)} deployment/product combinations")
        
        # Merge back to main dataframe
        df = df.merge(
            accuracy_stats[['deployment_id', 'selected_reference_code', 'pct_ok_modelo']],
            on=['deployment_id', 'selected_reference_code'],
            how='left'
        )
        
        # Rename column to match requirement
        df.rename(columns={'pct_ok_modelo': 'pct_ok_modelo_previsto'}, inplace=True)
        
        self.logger.info("Added expected model accuracy column")
        return df, accuracy_stats
    
    def add_is_trained_column(self, df):
        """Add is_trained column as integer based on selected_reference_is_trained"""
        self.logger.info("Adding is_trained column...")

        if 'selected_reference_is_trained' not in df.columns:
            self.logger.warning("Column 'selected_reference_is_trained' not found, setting is_trained to 0")
            df['is_trained'] = 0
        else:
            # Convert boolean/string to integer (True/1 -> 1, False/0 -> 0, others -> 0)
            df['is_trained'] = df['selected_reference_is_trained'].apply(
                lambda x: 1 if x is True or x == 1 or x == "true" or x == "True" else 0
            )
        
        self.logger.info("Added is_trained column")
        return df
    
    def add_num_referencias_no_similares(self, df):
        """Add column counting non-valid references in the predicted list based on validation files"""
        self.logger.info("Adding num_referencias_no_similares column based on validation status...")
        
        def count_non_valid_references(row):
            """Count references in predicted_codes that are marked as false/invalid in validation files"""
            try:
                predicted_codes = row.get('predicted_codes', [])
                result_index = row.get('result_index')
                deployment_id = row.get('deployment_id') if 'deployment_id' in row else None
                
                if not isinstance(predicted_codes, list) or not predicted_codes:
                    """DEPRECATED
                    Reemplazado por el comando oficial:
                        python main.py download_info <deployment_id>

                    Este archivo solo permanece como marcador temporal y será eliminado.
                    """
                    import sys

                    def main():
                        print("[DEPRECATED] Usa: python main.py download_info <deployment_id>")
                        return 1

                    if __name__ == '__main__':
                        sys.exit(main())
                    # Count as non-valid if validation result is "No válida"
                    if validacion == "No válida":
                        non_valid_count += 1
                
                return non_valid_count
                
            except Exception as e:
                self.logger.debug(f"Error counting non-valid references: {e}")
                return 0
        
        df['num_referencias_no_similares'] = df.apply(count_non_valid_references, axis=1)
        self.logger.info("Added num_referencias_no_similares column based on validation status")
        
        # Add num_referencias_similares column (num_candidatos - num_referencias_no_similares)
        df['num_referencias_similares'] = df['num_candidatos'] - df['num_referencias_no_similares']
        self.logger.info("Added num_referencias_similares column")
        
        # Add lista_similar_completa column (1 if all elements are similar, 0 otherwise)
        df['lista_similar_completa'] = ((df['num_referencias_no_similares'] == 0) & (df['num_candidatos'] > 0)).astype(int)
        self.logger.info("Added lista_similar_completa column")
        
        # Add producto_seleccionado_similar column (1 if selected product is valid, 0 otherwise)
        def check_selected_product_validity(row):
            """Check if the selected product is marked as valid in validation files"""
            try:
                result_index = row.get('result_index')
                selected_reference_code = str(row.get('selected_reference_code', '')).strip()
                deployment_id = row.get('deployment_id') if 'deployment_id' in row else None
                
                if not selected_reference_code or selected_reference_code == 'nan':
                    return 0
                
                if not self.parejas_validas or not self.todas_las_parejas:
                    return 0  # No validation data available
                
                # Use the same validation logic as in other functions
                if isinstance(self.parejas_validas, dict) and isinstance(self.todas_las_parejas, dict):
                    if deployment_id is not None:
                        validacion, _ = validar_pareja_completa(result_index, selected_reference_code, deployment_id, 
                                                             self.parejas_validas, self.todas_las_parejas)
                    else:
                        validacion = "No válida"
                else:
                    validacion, _ = validar_pareja_simple(result_index, selected_reference_code, 
                                                        self.parejas_validas, self.todas_las_parejas)
                
                # Return 1 if valid, 0 if not valid
                return 1 if validacion == "Válida" else 0
                
            except Exception as e:
                self.logger.debug(f"Error checking selected product validity: {e}")
                return 0
        
        df['producto_seleccionado_similar'] = df.apply(check_selected_product_validity, axis=1)
        self.logger.info("Added producto_seleccionado_similar column")
        
        return df
    
    def process_dataframe(self, df):
        """Apply all processing steps to the dataframe"""
        self.logger.info(f"Starting data processing for {len(df)} records...")
        
        # Diagnostic: show ALL available columns
        self.logger.info(f"ALL available columns: {list(df.columns)}")
        
        # Apply all processing steps
        df = self.add_predict_codes_analytics(df)
        df = self.add_topk_classification(df)
        df = self.add_is_trained_column(df)
        df, accuracy_stats = self.calculate_expected_model_accuracy(df)
        df = self.add_num_referencias_no_similares(df)
        
        # Add device status column if encendidos data is available
        if self.encendidos_data is not None:
            df = self.add_device_status_column(df, self.encendidos_data)
            df = self.add_fecha_encendido_column(df, self.encendidos_data)
        else:
            self.logger.warning("No encendidos data provided. Skipping device status column.")
            df['estado_dispositivo'] = 'No disponible'
            df['fecha_encendido'] = ''
        
        # Add image information columns from database
        df = self.add_image_info_columns(df)
        
        # Add model_ok_general_fail column
        df = self.add_model_ok_general_fail_column(df)
        
        # Generate parejas productos dataframe
        df_parejas = self.generate_parejas_productos(df)
        
        # Generate undefined pairs report
        df_undefined_pairs = self.generate_undefined_pairs_report(df)
        
        self.logger.info("Data processing completed")
        return df, accuracy_stats, df_parejas, df_undefined_pairs

    def generate_parejas_productos(self, df):
        """Generate parejas productos dataframe similar to analisis_listas.py"""
        self.logger.info("Generating parejas productos dataframe...")
        
        # Verificar si existen las columnas necesarias
        required_columns = ['result_index', 'selected_reference_code', 'selected_reference_name', 'predicted_codes', 'predicted_names']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns for parejas analysis: {missing_columns}")
            return pd.DataFrame()
        
        # Verificar si hay deployment_id
        tiene_deployment_id = 'deployment_id' in df.columns
        
        # Asegurar tipos de datos
        df['selected_reference_code'] = df['selected_reference_code'].astype(str)
        df['selected_reference_name'] = df['selected_reference_name'].astype(str)
        if tiene_deployment_id:
            df['deployment_id'] = df['deployment_id'].astype(str)
        
        todas_parejas = []
        
        # Groupby logic
        if tiene_deployment_id:
            grupos = df.groupby(['result_index', 'deployment_id'])
        else:
            grupos = df.groupby('result_index')
        
        for grupo_key, group in grupos:
            if tiene_deployment_id:
                result_idx, deployment_id = grupo_key
            else:
                result_idx = grupo_key
                deployment_id = None
            
            # Contar compras (seleccionados)
            compras = Counter(zip(group['selected_reference_code'], group['selected_reference_name']))
            
            # Contar apariciones en listas (predicciones)
            lista_counts = Counter()
            
            for idx, row in group.iterrows():
                codes = row['predicted_codes']
                names = row['predicted_names']
                
                # Handle both scalar and list values for NaN checking
                codes_is_empty = (
                    codes is None or 
                    (not isinstance(codes, list) and pd.isna(codes)) or
                    (isinstance(codes, list) and len(codes) == 0)
                )
                names_is_empty = (
                    names is None or 
                    (not isinstance(names, list) and pd.isna(names)) or
                    (isinstance(names, list) and len(names) == 0)
                )
                
                if codes_is_empty or names_is_empty:
                    continue
                
                # Limpiar y procesar códigos y nombres
                if isinstance(codes, list):
                    split_codes = [str(code).strip() for code in codes]
                else:
                    codes_clean = str(codes).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                    split_codes = [code.strip() for code in codes_clean.split(',')]
                
                if isinstance(names, list):
                    split_names = [str(name).strip() for name in names]
                else:
                    names_clean = str(names).replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                    split_names = [name.strip() for name in names_clean.split(',')]
                
                # Asegurar que tengan la misma longitud
                min_length = min(len(split_codes), len(split_names))
                split_codes = split_codes[:min_length]
                split_names = split_names[:min_length]
                
                lista_counts.update(zip(split_codes, split_names))
            
            # Unir claves únicas
            todas_claves = set(compras.keys()) | set(lista_counts.keys())
            
            for code, name in todas_claves:
                # Filtrar selected_reference vacíos o nulos
                if pd.isna(code) or code == '' or code == 'nan' or code == 'None':
                    continue
                if pd.isna(name) or name == '' or name == 'nan' or name == 'None':
                    continue
                
                # Obtener is_trained
                is_trained = group[
                    (group['selected_reference_code'] == code) &
                    (group['selected_reference_name'] == name)
                ]['selected_reference_is_trained'].astype(str).mode().iloc[0] if not group[
                    (group['selected_reference_code'] == code) &
                    (group['selected_reference_name'] == name)
                ].empty else 'False'
                
                # Validar pareja
                validacion, pareja_en_fichero = self._validar_pareja_en_parejas(result_idx, code, deployment_id)
                
                # Obtener descripción del producto
                descripcion_producto = self._obtener_descripcion_producto(result_idx, deployment_id)
                
                # Crear entrada de pareja
                pareja_entry = {
                    'Result_index': result_idx,
                    'producto': descripcion_producto,
                    'selected_reference': code,
                    'reference_name': name,
                    'validacion': validacion,
                    'pareja_en_fichero_ref': pareja_en_fichero,
                    'veces_seleccionado': compras.get((code, name), 0),
                    'veces_en_lista': lista_counts.get((code, name), 0),
                    'is_trained': is_trained
                }
                
                if tiene_deployment_id:
                    pareja_entry['deployment_id'] = deployment_id
                
                todas_parejas.append(pareja_entry)
        
        df_parejas = pd.DataFrame(todas_parejas)
        
        # Ordenar dataframe de parejas
        if not df_parejas.empty:
            if tiene_deployment_id:
                df_parejas = df_parejas.sort_values(['deployment_id', 'Result_index', 'veces_seleccionado'], ascending=[True, True, False])
            else:
                df_parejas = df_parejas.sort_values(['Result_index', 'veces_seleccionado'], ascending=[True, False])
        
        self.logger.info(f"Generated {len(df_parejas)} parejas productos")
        return df_parejas
    
    def _validar_pareja_en_parejas(self, result_idx, code, deployment_id):
        """Helper method to validate pareja for parejas analysis"""
        if not self.parejas_validas or not self.todas_las_parejas:
            return "No válida", False
        
        # Verificar si es formato con deployment_id
        if isinstance(self.parejas_validas, dict) and isinstance(self.todas_las_parejas, dict):
            if deployment_id is not None:
                return validar_pareja_completa(result_idx, code, deployment_id, self.parejas_validas, self.todas_las_parejas)
            else:
                return "No válida", False
        else:
            return validar_pareja_simple(result_idx, code, self.parejas_validas, self.todas_las_parejas)
    
    def _obtener_descripcion_producto(self, result_idx, deployment_id):
        """Helper method to get product description"""
        if not self.descripciones_productos:
            return ""
        
        try:
            result_idx_int = int(result_idx)
            
            # Verificar formato con deployment_id
            if isinstance(self.descripciones_productos, dict) and deployment_id is not None:
                deployment_id_str = str(deployment_id)
                if deployment_id_str in self.descripciones_productos:
                    deployment_descripciones = self.descripciones_productos[deployment_id_str]
                    return deployment_descripciones.get(result_idx_int, "")
            elif isinstance(self.descripciones_productos, dict) and not any(isinstance(v, dict) for v in self.descripciones_productos.values()):
                # Formato anterior sin deployment_id
                return self.descripciones_productos.get(result_idx_int, "")
            
            return ""
        except (ValueError, TypeError):
            return ""

    def generate_undefined_pairs_report(self, df):
        """Generate report of undefined pairs (not in reference files) by deployment_id"""
        self.logger.info("Generating undefined pairs report...")
        
        # Verificar si existen las columnas necesarias
        required_columns = ['result_index', 'selected_reference_code', 'predicted_codes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns for undefined pairs report: {missing_columns}")
            return pd.DataFrame()
        
        # Verificar si hay deployment_id
        tiene_deployment_id = 'deployment_id' in df.columns
        
        if not tiene_deployment_id:
            self.logger.warning("No deployment_id column found. Cannot generate deployment-specific report.")
            return pd.DataFrame()
        
        undefined_pairs_data = []
        
        # Agrupar por deployment_id
        for deployment_id in df['deployment_id'].unique():
            deployment_data = df[df['deployment_id'] == deployment_id]
            
            # Diccionarios para contar parejas no definidas
            undefined_pairs_count = {}  # (result_index, code) -> count
            selection_counts = {}       # (result_index, code) -> veces seleccionado
            list_counts = {}           # (result_index, code) -> veces en lista
            
            # Procesar cada fila del deployment
            for _, row in deployment_data.iterrows():
                result_index = row['result_index']
                selected_code = str(row['selected_reference_code']).strip()
                predicted_codes = row.get('predicted_codes', [])
                
                # Verificar pareja seleccionada
                if selected_code and selected_code != 'nan' and selected_code != 'None':
                    validacion, pareja_en_fichero = self._validar_pareja_en_parejas(
                        result_index, selected_code, deployment_id
                    )
                    
                    if not pareja_en_fichero:  # No está en el archivo
                        pair_key = (result_index, selected_code)
                        selection_counts[pair_key] = selection_counts.get(pair_key, 0) + 1
                        undefined_pairs_count[pair_key] = undefined_pairs_count.get(pair_key, 0) + 1
                
                # Verificar códigos predichos
                if isinstance(predicted_codes, list):
                    for code in predicted_codes:
                        code_str = str(code).strip()
                        if code_str and code_str != 'nan' and code_str != 'None':
                            validacion, pareja_en_fichero = self._validar_pareja_en_parejas(
                                result_index, code_str, deployment_id
                            )
                            
                            if not pareja_en_fichero:  # No está en el archivo
                                pair_key = (result_index, code_str)
                                list_counts[pair_key] = list_counts.get(pair_key, 0) + 1
                                undefined_pairs_count[pair_key] = undefined_pairs_count.get(pair_key, 0) + 1
            
            # Crear entrada para este deployment
            if undefined_pairs_count:
                total_undefined_pairs = len(undefined_pairs_count)
                total_selections = sum(selection_counts.values())
                total_list_appearances = sum(list_counts.values())
                
                undefined_pairs_data.append({
                    'deployment_id': deployment_id,
                    'num_parejas_no_definidas': total_undefined_pairs,
                    'num_veces_seleccionadas': total_selections,
                    'num_veces_listas': total_list_appearances
                })
            else:
                # Agregar entrada con ceros si no hay parejas no definidas
                undefined_pairs_data.append({
                    'deployment_id': deployment_id,
                    'num_parejas_no_definidas': 0,
                    'num_veces_seleccionadas': 0,
                    'num_veces_listas': 0
                })
        
        df_undefined = pd.DataFrame(undefined_pairs_data)
        
        if not df_undefined.empty:
            df_undefined = df_undefined.sort_values('deployment_id')
        
        self.logger.info(f"Generated undefined pairs report for {len(df_undefined)} deployments")
        return df_undefined

    def add_device_status_column(self, df, encendidos_data):
        """Add column indicating if each record is 'Silencioso' or 'Encendido' based on device activation dates"""
        self.logger.info("Adding device status column (Silencioso/Encendido)...")
        
        # Verificar si existe la columna device_id y timestamp
        if 'device_id' not in df.columns:
            self.logger.warning("Column 'device_id' not found. Setting all records as 'Silencioso'")
            df['estado_dispositivo'] = 'Silencioso'
            return df
        
        # Buscar columna de fecha/timestamp
        timestamp_col = None
        for col in ['transaction_start_time', 'transaction_end_time', 'timestamp', 'fecha', 'date', '@timestamp']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            self.logger.warning("No timestamp column found. Setting all records as 'Silencioso'")
            df['estado_dispositivo'] = 'Silencioso'
            return df
        
        # Convertir timestamp a datetime si no lo está ya
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        except Exception as e:
            self.logger.error(f"Error converting timestamp column to datetime: {e}")
            df['estado_dispositivo'] = 'Silencioso'
            return df
        
        # Calcular fecha mínima de los datos menos un mes (para dispositivos sin fecha de encendido)
        min_date_in_data = df[timestamp_col].min()
        if pd.isna(min_date_in_data):
            self.logger.warning("No valid timestamps found in data")
            df['estado_dispositivo'] = 'Silencioso'
            return df
        
        # Fecha por defecto: fecha mínima menos un mes
        default_activation_date = min_date_in_data - pd.DateOffset(months=1)
        self.logger.info(f"Default activation date for devices without date: {default_activation_date}")
        
        def determine_device_status(row):
            """Determine if device is 'Silencioso' or 'Encendido' for this record"""
            try:
                device_id_raw = row['device_id']
                record_timestamp = row[timestamp_col]
                
                # Limpiar y normalizar device_id
                if pd.isna(device_id_raw):
                    return 'Silencioso'
                
                device_id = str(device_id_raw).strip()
                
                # Si es un float que se convirtió a string (ej: "10039.0"), quitar el .0
                if device_id.endswith('.0'):
                    device_id = device_id[:-2]
                
                # Si no hay device_id válido o timestamp válido -> Silencioso
                if not device_id or device_id == 'nan' or device_id == 'None' or pd.isna(record_timestamp):
                    return 'Silencioso'
                
                # Caso 1: device_id no está en el archivo de encendidos -> Silencioso
                if device_id not in encendidos_data:
                    return 'Silencioso'
                
                # Caso 2: device_id está en el archivo pero sin fecha -> usar fecha por defecto
                activation_date = encendidos_data[device_id]
                if activation_date is None:
                    activation_date = default_activation_date
                
                # Caso 3: device_id está en el archivo con fecha -> comparar
                # Convertir ambas fechas a naive (sin timezone) para comparación
                record_timestamp_naive = record_timestamp
                if hasattr(record_timestamp, 'tz_localize'):
                    # Si es timezone-aware, convertir a naive
                    if record_timestamp.tz is not None:
                        record_timestamp_naive = record_timestamp.tz_localize(None)
                elif hasattr(record_timestamp, 'replace'):
                    # Si es datetime con tzinfo, quitarla
                    if getattr(record_timestamp, 'tzinfo', None) is not None:
                        record_timestamp_naive = record_timestamp.replace(tzinfo=None)
                
                activation_date_naive = activation_date
                if hasattr(activation_date, 'tz_localize'):
                    if activation_date.tz is not None:
                        activation_date_naive = activation_date.tz_localize(None)
                elif hasattr(activation_date, 'replace'):
                    if getattr(activation_date, 'tzinfo', None) is not None:
                        activation_date_naive = activation_date.replace(tzinfo=None)
                
                if record_timestamp_naive >= activation_date_naive:
                    return 'Encendido'
                else:
                    return 'Silencioso'
                    
            except Exception as e:
                self.logger.debug(f"Error determining device status: {e}")
                return 'Silencioso'
        
        df['estado_dispositivo'] = df.apply(determine_device_status, axis=1)
        
        # Log estadísticas
        status_counts = df['estado_dispositivo'].value_counts()
        self.logger.info(f"Device status distribution: {status_counts.to_dict()}")
        
        return df
    
    def add_fecha_encendido_column(self, df, encendidos_data):
        """Add column with activation date for each device"""
        self.logger.info("Adding fecha_encendido column...")
        
        # Verificar si existe la columna device_id
        if 'device_id' not in df.columns:
            self.logger.warning("Column 'device_id' not found. Setting fecha_encendido as empty")
            df['fecha_encendido'] = ''
            return df
        
        def get_device_activation_date(row):
            """Get activation date for device"""
            try:
                device_id_raw = row['device_id']
                
                # Limpiar y normalizar device_id
                if pd.isna(device_id_raw):
                    return ''
                
                device_id = str(device_id_raw).strip()
                
                # Si es un float que se convirtió a string (ej: "10039.0"), quitar el .0
                if device_id.endswith('.0'):
                    device_id = device_id[:-2]
                
                if not device_id or device_id == 'nan' or device_id == 'None':
                    return ''
                
                # Buscar en datos de encendidos
                if device_id in encendidos_data:
                    activation_date = encendidos_data[device_id]
                    if activation_date is not None:
                        # Convertir a string en formato YYYY-MM-DD
                        return activation_date.strftime('%Y-%m-%d')
                    else:
                        return 'Sin fecha'
                else:
                    return 'No en archivo'
                    
            except Exception as e:
                self.logger.debug(f"Error getting activation date: {e}")
                return ''
        
        df['fecha_encendido'] = df.apply(get_device_activation_date, axis=1)
        
        # Log estadísticas
        fecha_counts = df['fecha_encendido'].value_counts()
        self.logger.info(f"Fecha encendido distribution: {fecha_counts.head(10).to_dict()}")
        
        return df
    
    def add_image_info_columns(self, df):
        """Add image information columns from database"""
        self.logger.info("Adding image information columns from database...")
        
        try:
            # Verificar si existe la columna transaction_id
            if 'transaction_id' not in df.columns:
                self.logger.warning("Column 'transaction_id' not found. Adding empty image columns")
                df['has_image'] = 0
                df['num_images'] = 0
                df['image_link'] = ""
                return df
            
            # Probar conexión a la base de datos
            self.logger.info("Testing database connection...")
            if not test_database_connection():
                self.logger.error("Database connection failed. Adding empty image columns")
                df['has_image'] = 0
                df['num_images'] = 0
                df['image_link'] = ""
                return df
            
            # Usar la función del módulo bbdd_utils para añadir las columnas
            df = add_image_columns_to_dataframe(df, 'transaction_id')
            
            self.logger.info("[OK] Image information columns added successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error adding image information: {e}")
            # En caso de error, añadir columnas vacías
            df['has_image'] = 0
            df['num_images'] = 0
            df['image_link'] = ""
            return df

    def add_model_ok_general_fail_column(self, df):
        """Add column that is 1 when model is OK but general result is Error (not no_result)"""
        self.logger.info("Adding model_ok_general_fail column...")
        
        def check_model_ok_general_fail(row):
            """Check if model is OK (Ok_modelo=1) but general result is Error"""
            try:
                # Check if model is OK - looking for Ok_modelo column (with capital O)
                model_ok = False
                if 'Ok_modelo' in df.columns:
                    model_ok = row.get('Ok_modelo', 0) == 1
                elif 'ok_modelo' in df.columns:
                    # Fallback to lowercase version if exists
                    model_ok = row.get('ok_modelo', 0) == 1
                else:
                    self.logger.warning("Column 'Ok_modelo' or 'ok_modelo' not found")
                    return 0
                
                # Check if general result is Error (not No_result)
                general_error = False
                if 'Result' in df.columns:
                    result_value = row.get('Result', '')
                    # Only consider as error if Result is specifically "Error", not "No_result"
                    general_error = (result_value == "Error")
                else:
                    self.logger.warning("Column 'Result' not found")
                    return 0
                
                # Return 1 if model OK (Ok_modelo=1) AND general result is Error, 0 otherwise
                return 1 if (model_ok and general_error) else 0
                
            except Exception as e:
                self.logger.debug(f"Error checking model_ok_general_fail: {e}")
                return 0
        
        df['model_ok_general_fail'] = df.apply(check_model_ok_general_fail, axis=1)
        
        # Log statistics
        model_ok_general_fail_count = df['model_ok_general_fail'].sum()
        total_records = len(df)
        percentage = (model_ok_general_fail_count / total_records * 100) if total_records > 0 else 0
        
        # Log detailed breakdown for debugging
        if ('Ok_modelo' in df.columns or 'ok_modelo' in df.columns) and 'Result' in df.columns:
            # Use the correct column name
            ok_modelo_col = 'Ok_modelo' if 'Ok_modelo' in df.columns else 'ok_modelo'
            ok_modelo_count = (df[ok_modelo_col] == 1).sum()
            result_error_count = (df['Result'] == "Error").sum()
            result_no_result_count = (df['Result'] == "No_result").sum()
            
            self.logger.info(f"Column statistics:")
            self.logger.info(f"  - {ok_modelo_col} = 1: {ok_modelo_count} records")
            self.logger.info(f"  - Result = 'Error': {result_error_count} records")
            self.logger.info(f"  - Result = 'No_result': {result_no_result_count} records (ignored)")
            self.logger.info(f"  - Both conditions met: {model_ok_general_fail_count} records ({percentage:.1f}%)")
        else:
            self.logger.info(f"Model OK but General Fail cases: {model_ok_general_fail_count} of {total_records} ({percentage:.1f}%)")
        
        return df

def cargar_referencias_validas(filepath):
    """Carga el archivo de referencias válidas y devuelve un conjunto de parejas válidas y todas las parejas por deployment_id."""
    if not os.path.exists(filepath):
        print(f"Archivo de referencias válidas no encontrado: {filepath}")
        return {}, {}
    
    try:
        print(f"Cargando referencias válidas desde: {filepath}")
        
        # Manejo de diferentes codificaciones
        df_validas = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                # Primero intentar con tabulador
                df_validas = pd.read_csv(filepath, sep=';', encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                try:
                    # Si falla, intentar con punto y coma
                    df_validas = pd.read_csv(filepath, sep=';', encoding=encoding)
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        
        if df_validas is None:
            print(f"ERROR: No se pudo cargar el archivo con ninguna codificación")
            return {}, {}
        
        # Verificar si tiene deployment_id (nuevo formato) o no (formato anterior)
        tiene_deployment_id = 'deployment_id' in df_validas.columns
        
        if tiene_deployment_id:
            # Nuevo formato con deployment_id
            required_cols = ['deployment_id', 'result_index', 'selected_reference', 'valida']
            missing_cols = [col for col in required_cols if col not in df_validas.columns]
            if missing_cols:
                print(f"ERROR: Columnas faltantes en archivo de referencias: {missing_cols}")
                return {}, {}
            
            parejas_validas_por_deployment = {}
            todas_las_parejas_por_deployment = {}
            
            for deployment_id in df_validas['deployment_id'].unique():
                deployment_id_str = str(deployment_id)
                deployment_data = df_validas[df_validas['deployment_id'] == deployment_id]
                
                parejas_validas = set()
                todas_las_parejas = set()
                
                for _, row in deployment_data.iterrows():
                    try:
                        result_idx = int(row['result_index'])
                        selected_ref = str(row['selected_reference']).strip()
                        
                        # Convertir selected_reference a integer para normalizar
                        try:
                            selected_ref_int = int(selected_ref)
                            pareja = (result_idx, selected_ref_int)
                        except ValueError:
                            # Si no se puede convertir a int, usar string original
                            pareja = (result_idx, selected_ref)
                        
                        todas_las_parejas.add(pareja)
                        
                        valida_str = str(row['valida']).lower().strip()
                        if valida_str in ['true', '1', 'yes', 'sí', 'si']:
                            parejas_validas.add(pareja)
                    except (ValueError, TypeError) as e:
                        print(f"Error procesando fila para deployment {deployment_id}: {row} - {e}")
                        continue
                
                parejas_validas_por_deployment[deployment_id_str] = parejas_validas
                todas_las_parejas_por_deployment[deployment_id_str] = todas_las_parejas
            
            print(f"Cargados {len(parejas_validas_por_deployment)} deployments desde {filepath}")
            
            for dep_id, parejas in parejas_validas_por_deployment.items():
                print(f"  Deployment {dep_id}: {len(parejas)} parejas válidas de {len(todas_las_parejas_por_deployment[dep_id])} totales")
            
            return parejas_validas_por_deployment, todas_las_parejas_por_deployment
            
        else:
            # Formato anterior sin deployment_id
            required_cols = ['result_index', 'selected_reference', 'valida']
            missing_cols = [col for col in required_cols if col not in df_validas.columns]
            if missing_cols:
                print(f"ERROR: Columnas faltantes en archivo de referencias: {missing_cols}")
                return set(), set()
            
            parejas_validas = set()
            todas_las_parejas = set()
            
            for _, row in df_validas.iterrows():
                try:
                    result_idx = int(row['result_index'])
                    selected_ref = str(row['selected_reference']).strip()
                    
                    # Convertir selected_reference a integer para normalizar
                    try:
                        selected_ref_int = int(selected_ref)
                        pareja = (result_idx, selected_ref_int)
                    except ValueError:
                        # Si no se puede convertir a int, usar string original
                        pareja = (result_idx, selected_ref)
                    
                    todas_las_parejas.add(pareja)
                    
                    valida_str = str(row['valida']).lower().strip()
                    if valida_str in ['true', '1', 'yes', 'sí', 'si']:
                        parejas_validas.add(pareja)
                except (ValueError, TypeError) as e:
                    print(f"Error procesando fila: {row} - {e}")
                    continue
            
            print(f"Cargadas {len(parejas_validas)} descripciones de productos desde {filepath}")
            return parejas_validas, todas_las_parejas
    except Exception as e:
        print(f"Error cargando referencias válidas: {e}")
        return {}, {}

def cargar_descripciones_productos(filepath):
    """Carga las descripciones de productos desde el archivo de etiquetas por deployment_id."""
    if not os.path.exists(filepath):
        print(f"Archivo de descripciones de productos no encontrado: {filepath}")
        return {}
    
    try:
        print(f"Cargando descripciones de productos desde: {filepath}")
        
        # Manejo de diferentes codificaciones
        df_productos = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df_productos = pd.read_csv(filepath, sep=';', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df_productos is None:
            print(f"ERROR: No se pudo cargar el archivo con ninguna codificación")
            return {}
        
        df_productos.columns = df_productos.columns.str.strip()
        
        # Verificar si existe la columna deployment_id
        tiene_deployment_id = 'deployment_id' in df_productos.columns
        
        if tiene_deployment_id:
            # Nuevo formato con deployment_id
            descripciones_por_deployment = {}
            
            for deployment_id in df_productos['deployment_id'].unique():
                deployment_id_str = str(deployment_id)
                deployment_data = df_productos[df_productos['deployment_id'] == deployment_id]
                
                deployment_data = deployment_data.copy()
                deployment_data['label_index'] = deployment_data['label_index'].astype(str).str.replace('\xa0', '').str.strip()
                deployment_data['label_name'] = deployment_data['label_name'].astype(str).str.replace('\xa0', '').str.strip()
                
                deployment_data['label_index'] = pd.to_numeric(deployment_data['label_index'], errors='coerce')
                deployment_data = deployment_data.dropna(subset=['label_index'])
                deployment_data['label_index'] = deployment_data['label_index'].astype(int)
                
                descripciones = dict(zip(deployment_data['label_index'], deployment_data['label_name']))
                descripciones_por_deployment[deployment_id_str] = descripciones
            
            print(f"Cargadas descripciones para {len(descripciones_por_deployment)} deployments desde {filepath}")
            return descripciones_por_deployment
            
        else:
            # Formato anterior sin deployment_id
            df_productos['label_index'] = df_productos['label_index'].astype(str).str.replace('\xa0', '').str.strip()
            df_productos['label_name'] = df_productos['label_name'].astype(str).str.replace('\xa0', '').str.strip()
            
            df_productos['label_index'] = pd.to_numeric(df_productos['label_index'], errors='coerce')
            df_productos = df_productos.dropna(subset=['label_index'])
            df_productos['label_index'] = df_productos['label_index'].astype(int)
            
            descripciones = dict(zip(df_productos['label_index'], df_productos['label_name']))
            print(f"Cargadas {len(descripciones)} descripciones de productos desde {filepath}")
            return descripciones
        
    except Exception as e:
        print(f"Error cargando descripciones de productos: {e}")
        return {}

def validar_pareja_completa(result_index, selected_reference, deployment_id, parejas_validas_por_deployment, todas_las_parejas_por_deployment):
    """Valida si una pareja result_index/selected_reference es válida y si está en el archivo para un deployment específico."""
    deployment_id_str = str(deployment_id)
    
        
    if not todas_las_parejas_por_deployment or deployment_id_str not in todas_las_parejas_por_deployment:
        return "No válida", False
    
    try:
        result_idx = int(result_index)
        selected_ref = str(selected_reference).strip()
        
        # Convertir selected_reference a integer para normalizar comparación
        # Los datos de Elasticsearch pueden venir como '00014' y en CSV como 14
        try:
            selected_ref_int = int(selected_ref)
            pareja = (result_idx, selected_ref_int)
        except ValueError:
            # Si no se puede convertir a int, usar string original
            pareja = (result_idx, selected_ref)
        
        parejas_validas = parejas_validas_por_deployment.get(deployment_id_str, set())
        todas_las_parejas = todas_las_parejas_por_deployment.get(deployment_id_str, set())
        
        pareja_en_fichero = pareja in todas_las_parejas
        
        
        if pareja_en_fichero:
            if pareja in parejas_validas:
                return "Válida", True
            else:
                return "No válida", True
        else:
            return "No válida", False
        
    except (ValueError, TypeError) as e:
        print(f"Error validando pareja ({result_index}, {selected_reference}) para deployment {deployment_id_str}: {e}")
        return "No válida", False

def validar_pareja_simple(result_index, selected_reference, parejas_validas, todas_las_parejas):
    """Valida si una pareja result_index/selected_reference es válida (formato sin deployment_id)."""
    if not todas_las_parejas:
        return "No válida", False
    
    try:
        result_idx = int(result_index)
        selected_ref = str(selected_reference).strip()
        
        # Convertir selected_reference a integer para normalizar comparación
        try:
            selected_ref_int = int(selected_ref)
            pareja = (result_idx, selected_ref_int)
        except ValueError:
            # Si no se puede convertir a int, usar string original
            pareja = (result_idx, selected_ref)
        
        pareja_en_fichero = pareja in todas_las_parejas
        
        if pareja_en_fichero:
            if pareja in parejas_validas:
                return "Válida", True
            else:
                return "No válida", True
        else:
            return "No válida", False
        
    except (ValueError, TypeError) as e:
        print(f"Error validando pareja ({result_index}, {selected_reference}): {e}")
        return "No válida", False

def clean_dataframe_for_excel(df, logger):
    """Clean dataframe to avoid Excel corruption issues"""
    df_clean = df.copy()
    
    # Clean problematic data types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Convert lists/dicts to strings and handle special characters
            df_clean[col] = df_clean[col].apply(lambda x: 
                str(x).replace('\x00', '').replace('\ufffd', '') if x is not None else ''
            )
            
            # Limit string length to avoid Excel issues
            df_clean[col] = df_clean[col].apply(lambda x: 
                x[:32767] if isinstance(x, str) and len(x) > 32767 else x
            )
    
    # Replace inf and -inf with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with appropriate values
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna('')
    
    logger.info(f"Cleaned dataframe for Excel export: {len(df_clean)} rows, {len(df_clean.columns)} columns")
    return df_clean

def save_excel_file(df_processed, df_parejas, accuracy_stats, df_undefined_pairs, output_file, logger):
    """Save data to Excel file with error handling and data cleaning"""
    try:
        logger.info(f"Preparando datos para Excel...")
        
        # Clean dataframes
        df_processed_clean = clean_dataframe_for_excel(df_processed, logger)
        df_parejas_clean = clean_dataframe_for_excel(df_parejas, logger)
        df_undefined_clean = clean_dataframe_for_excel(df_undefined_pairs, logger)
        
        # Save as Excel with multiple sheets
        logger.info(f"Guardando archivo Excel en: {output_file}")
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            logger.info("  - Guardando hoja 'Datos_Procesados'...")
            df_processed_clean.to_excel(writer, sheet_name='Datos_Procesados', index=False)
            
            logger.info("  - Guardando hoja 'Parejas_Productos'...")
            df_parejas_clean.to_excel(writer, sheet_name='Parejas_Productos', index=False)
            
            if len(accuracy_stats) > 0:
                logger.info("  - Guardando hoja 'Accuracy_Stats'...")
                accuracy_stats_clean = clean_dataframe_for_excel(accuracy_stats, logger)
                accuracy_stats_clean.to_excel(writer, sheet_name='Accuracy_Stats', index=False)
            
            if len(df_undefined_pairs) > 0:
                logger.info("  - Guardando hoja 'Parejas_No_Definidas'...")
                df_undefined_clean.to_excel(writer, sheet_name='Parejas_No_Definidas', index=False)
        
        logger.info(f"[OK] Archivo Excel guardado exitosamente: {output_file}")
        
    except Exception as e:
        logger.error(f"[ERROR] Error guardando Excel: {e}")
        logger.error(f"Detalles del error: {type(e).__name__}: {str(e)}")
        
        # Fallback to CSV
        try:
            csv_output = output_file.replace('.xlsx', '.csv')
            df_processed.to_csv(csv_output, index=False, sep=';', decimal=',', encoding='utf-8-sig')
            logger.info(f"[WARNING] Guardado como CSV: {csv_output}")
            
            # Save parejas as separate CSV
            parejas_csv = output_file.replace('.xlsx', '_parejas.csv')
            df_parejas.to_csv(parejas_csv, index=False, sep=';', decimal=',', encoding='utf-8-sig')
            logger.info(f"[WARNING] Parejas guardadas como CSV: {parejas_csv}")
            
            if len(accuracy_stats) > 0:
                accuracy_csv = output_file.replace('.xlsx', '_accuracy_stats.csv')
                accuracy_stats.to_csv(accuracy_csv, index=False, sep=';', decimal=',', encoding='utf-8-sig')
                logger.info(f"[WARNING] Accuracy stats guardadas como CSV: {accuracy_csv}")
            
            if len(df_undefined_pairs) > 0:
                undefined_csv = output_file.replace('.xlsx', '_parejas_no_definidas.csv')
                df_undefined_pairs.to_csv(undefined_csv, index=False, sep=';', decimal=',', encoding='utf-8-sig')
                logger.info(f"[WARNING] Parejas no definidas guardadas como CSV: {undefined_csv}")
                
        except Exception as csv_error:
            logger.error(f"[ERROR] Error también en fallback CSV: {csv_error}")
            raise

def cargar_encendidos(filepath):
    """Carga las fechas de encendido por device_id desde el archivo CSV."""
    if not os.path.exists(filepath):
        print(f"Archivo de encendidos no encontrado: {filepath}")
        return {}
    
    try:
        print(f"Cargando fechas de encendido desde: {filepath}")
        
        # Manejo de diferentes codificaciones
        df_encendidos = None
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df_encendidos = pd.read_csv(filepath, sep=';', encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df_encendidos is None:
            print(f"ERROR: No se pudo cargar el archivo con ninguna codificación")
            return {}
        
        # Limpiar nombres de columnas
        df_encendidos.columns = df_encendidos.columns.str.strip()
        
        # Verificar columnas requeridas
        required_cols = ['device_id']
        missing_cols = [col for col in required_cols if col not in df_encendidos.columns]
        if missing_cols:
            print(f"ERROR: Columnas faltantes en archivo de encendidos: {missing_cols}")
            return {}
        
        # Procesar fechas de encendido
        encendidos = {}
        
        for _, row in df_encendidos.iterrows():
            try:
                device_id_raw = row['device_id']
                
                # Limpiar y normalizar device_id
                if pd.isna(device_id_raw):
                    continue
                
                # Convertir a string y limpiar
                device_id = str(device_id_raw).strip()
                
                # Si es un float que se convirtió a string (ej: "10039.0"), quitar el .0
                if device_id.endswith('.0'):
                    device_id = device_id[:-2]
                
                if not device_id or device_id == 'nan' or device_id == 'None':
                    continue
                
                # Verificar si hay columna fecha_encendido
                fecha_col = None
                for col_name in ['fecha_encendido', 'Fecha encendido', 'fecha_encendido', 'Fecha_encendido']:
                    if col_name in df_encendidos.columns:
                        fecha_col = col_name
                        break
                
                if fecha_col is not None:
                    fecha_encendido = row[fecha_col]
                    
                    # Si la fecha está vacía o es NaN, se marcará como None
                    if pd.isna(fecha_encendido) or str(fecha_encendido).strip() == '' or str(fecha_encendido).strip().lower() == 'nan':
                        encendidos[device_id] = None
                    else:
                        # Convertir a datetime
                        try:
                            # Intentar diferentes formatos de fecha
                            fecha_str = str(fecha_encendido).strip()
                            for date_format in ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S']:
                                try:
                                    fecha_dt = pd.to_datetime(fecha_str, format=date_format)
                                    encendidos[device_id] = fecha_dt
                                    break
                                except ValueError:
                                    continue
                            else:
                                # Si ningún formato funciona, usar pandas parser automático
                                encendidos[device_id] = pd.to_datetime(fecha_str, errors='coerce')
                                if pd.isna(encendidos[device_id]):
                                    print(f"ADVERTENCIA: No se pudo parsear fecha '{fecha_str}' para device_id {device_id}")
                                    encendidos[device_id] = None
                        except Exception as e:
                            print(f"Error procesando fecha para device_id {device_id}: {e}")
                            encendidos[device_id] = None
                else:
                    # Si no hay columna fecha_encendido, marcar como None (se calculará fecha mínima - 1 mes)
                    encendidos[device_id] = None
                    
            except Exception as e:
                print(f"Error procesando fila de encendidos: {row} - {e}")
                continue
        
        print(f"Cargados {len(encendidos)} dispositivos desde {filepath}")
        
        device_count_with_date = sum(1 for fecha in encendidos.values() if fecha is not None)
        device_count_without_date = len(encendidos) - device_count_with_date
        
        print(f"  - Dispositivos con fecha: {device_count_with_date}")
        print(f"  - Dispositivos sin fecha: {device_count_without_date}")
        
        return encendidos
        
    except Exception as e:
        print(f"Error cargando archivo de encendidos: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(
        description="Download and process data from Elasticsearch with enhanced analytics."
    )
    parser.add_argument('--config', type=str, default='config/credentials/credentials_elastic_prod.json',
                        help='Path to the JSON config file')
    parser.add_argument('--output', type=str, default='results/processed_device_transactions.xlsx',
                        help='Path to the output XLSX file')
    parser.add_argument('--query', type=str, default='query/consulta_elastic_dataset_training.json',
                        help='Path to the query JSON file')
    parser.add_argument('--log', type=str, default=None,
                        help='Optional log file path')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with limited data')
    parser.add_argument('--referencias_validas', type=str, 
                       default='config/referencias/referencias_validas.csv', 
                       help='Ruta al fichero CSV de parejas válidas result_index/selected_reference')
    parser.add_argument('--productos_labels', type=str, 
                       default='config/referencias/productos_visuales.csv', 
                       help='Ruta al fichero CSV con descripciones de productos por label_index')
    parser.add_argument('--encendidos', type=str, 
                       default='config/status/encendidos.csv', 
                       help='Ruta al fichero CSV con fechas de encendido por device_id')
    args = parser.parse_args()

    setup_logging(args.log)
    logger = logging.getLogger()

    config_file = args.config
    output_file = args.output
    query_file = args.query
    
    # Create output directory if needed
    if not (output_file.startswith('http://') or output_file.startswith('https://')):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        # Cargar archivos auxiliares
        print("=== Cargando archivos auxiliares ===")
        parejas_validas, todas_las_parejas = cargar_referencias_validas(args.referencias_validas)
        descripciones_productos = cargar_descripciones_productos(args.productos_labels)
        encendidos_data = cargar_encendidos(args.encendidos)
        
        # Initialize processor with reference data
        processor = ElasticDataProcessor(logger, parejas_validas, todas_las_parejas, descripciones_productos, encendidos_data)
        
        # Connect to Elasticsearch
        config = load_config(config_file)
        es_client = connect_es(
            host=config.get('host', 'localhost'),
            username=config.get('username', ''),
            password=config.get('password', ''),
            verify_certs=config.get('verify_certs', True),
            timeout=config.get('timeout', 60),
            auth_method=config.get('auth_method'),
            api_key_id=config.get('api_key_id'),
            api_key_secret=config.get('api_key_secret'),
            ca_certs=config.get('ca_certs'),
            port=config.get('port', 9200)
        )
        
        if es_client.ping():
            logger.info("Successfully connected to Elasticsearch.")
            
            # Download data
            docs = download_data(es_client, query_file)
            logger.info(f"Total documents retrieved: {len(docs)}")
            
            # Convert to dataframe
            df = convert_to_dataframe(docs)
            
            # Remove duplicates (same logic as original script)
            initial_count = len(df)
            logger.info(f"Checking for unhashable columns that prevent duplicate removal...")
            
            # Handle problematic columns for duplicate removal
            problematic_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample_values = df[col].dropna().head(5)
                    if not sample_values.empty:
                        for val in sample_values:
                            if isinstance(val, (list, dict)):
                                problematic_cols.append(col)
                                break
            
            if problematic_cols:
                logger.info(f"Found {len(problematic_cols)} columns with unhashable types: {problematic_cols}")
                df_for_dedup = df.copy()
                for col in problematic_cols:
                    df_for_dedup[col] = df_for_dedup[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
                
                duplicated_mask = df_for_dedup.duplicated()
                df = df[~duplicated_mask]
                final_count = len(df)
            else:
                df = df.drop_duplicates()
                final_count = len(df)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} duplicate records. Final count: {final_count}")
            
            # Process the data with enhanced analytics
            df_processed, accuracy_stats, df_parejas, df_undefined_pairs = processor.process_dataframe(df)
            
            # Asegurar que la salida sea .xlsx
            if not output_file.endswith('.xlsx'):
                if output_file.endswith('.csv'):
                    output_file = output_file.replace('.csv', '.xlsx')
                else:
                    output_file += '.xlsx'
                logger.info(f"Cambiando extensión de salida a .xlsx: {output_file}")
            
            # Save processed data
            if output_file.startswith('http://') or output_file.startswith('https://'):
                # SharePoint upload logic - same as download_elastic_data.py
                import tempfile
                from utils.sharepoint_utils import upload_file_to_sharepoint_app_v2, parse_sharepoint_url
                
                logger.info("Detectada URL de SharePoint. Guardando y subiendo archivo...")
                
                try:
                    _, _, real_filename = parse_sharepoint_url(output_file)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_xlsx:
                        temp_path = tmp_xlsx.name
                    
                    # Save as Excel with multiple sheets to temporary file
                    logger.info(f"Guardando archivo Excel temporalmente en: {temp_path}")
                    save_excel_file(df_processed, df_parejas, accuracy_stats, df_undefined_pairs, temp_path, logger)
                    
                    # Upload to SharePoint
                    logger.info(f"Subiendo archivo a SharePoint: {output_file}")
                    cred_path = 'config/credentials/credentials_sharepoint.json'
                    upload_file_to_sharepoint_app_v2(
                        local_path=temp_path,
                        output_url=output_file,
                        cred_path=cred_path
                    )
                    
                    # Clean up temporary file
                    os.remove(temp_path)
                    logger.info(f"[OK] Archivo subido exitosamente a SharePoint: {output_file}")
                    
                except Exception as e:
                    logger.error(f"Error subiendo a SharePoint: {e}")
                    logger.error("Guardando archivo localmente como respaldo...")
                    local_output = output_file.replace('http://', '').replace('https://', '').replace('/', '_')
                    local_output = f"results/{local_output}"
                    save_excel_file(df_processed, df_parejas, accuracy_stats, df_undefined_pairs, local_output, logger)
                    logger.info(f"Archivo guardado localmente en: {local_output}")
                    
            else:
                # Local file save as Excel with multiple sheets
                logger.info(f"Guardando archivo Excel en: {output_file}")
                save_excel_file(df_processed, df_parejas, accuracy_stats, df_undefined_pairs, output_file, logger)
                
                # Summary statistics
                logger.info("=== PROCESSING SUMMARY ===")
                logger.info(f"Total processed records: {len(df_processed)}")
                logger.info(f"Records with lista_larga: {df_processed['lista_larga'].sum()}")
                logger.info(f"Records with lista_un_candidato: {df_processed['lista_un_candidato'].sum()}")
                logger.info(f"Top_k_0 records: {df_processed['top_k_0'].sum()}")
                logger.info(f"Top_k_medio records: {df_processed['top_k_medio'].sum()}")
                logger.info(f"Top_k_alto records: {df_processed['top_k_alto'].sum()}")
                logger.info(f"No clasificado records (topk=-1): {(df_processed['pos_candidato_elegido'] == 'NO_CLASIFICADO').sum()}")
                logger.info(f"Records with is_trained=1: {df_processed['is_trained'].sum()}")
                logger.info(f"Records with expected accuracy: {df_processed['pct_ok_modelo_previsto'].notna().sum()}")
                logger.info(f"Records with num_referencias_no_similares>0: {(df_processed['num_referencias_no_similares'] > 0).sum()}")
                logger.info(f"Records with num_referencias_similares>0: {(df_processed['num_referencias_similares'] > 0).sum()}")
                logger.info(f"Records with lista_similar_completa=1: {df_processed['lista_similar_completa'].sum()}")
                logger.info(f"Records with producto_seleccionado_similar=1: {df_processed['producto_seleccionado_similar'].sum()}")
                logger.info(f"Parejas productos records: {len(df_parejas)}")
                
                # Device status summary
                if 'estado_dispositivo' in df_processed.columns:
                    device_status_counts = df_processed['estado_dispositivo'].value_counts()
                    logger.info("=== DEVICE STATUS SUMMARY ===")
                    for status, count in device_status_counts.items():
                        logger.info(f"Records with status '{status}': {count}")
                else:
                    logger.info("No device status information available")
                
                # Fecha encendido summary
                if 'fecha_encendido' in df_processed.columns:
                    fecha_encendido_counts = df_processed['fecha_encendido'].value_counts()
                    logger.info("=== FECHA ENCENDIDO SUMMARY ===")
                    for fecha, count in fecha_encendido_counts.head(10).items():
                        logger.info(f"Records with fecha_encendido '{fecha}': {count}")
                else:
                    logger.info("No fecha encendido information available")
                
                # Summary de parejas no definidas
                if len(df_undefined_pairs) > 0:
                    total_undefined = df_undefined_pairs['num_parejas_no_definidas'].sum()
                    total_selections_undefined = df_undefined_pairs['num_veces_seleccionadas'].sum()
                    total_lists_undefined = df_undefined_pairs['num_veces_listas'].sum()
                    
                    logger.info("=== PAREJAS NO DEFINIDAS ===")
                    logger.info(f"Total deployments with undefined pairs: {len(df_undefined_pairs[df_undefined_pairs['num_parejas_no_definidas'] > 0])}")
                    logger.info(f"Total undefined pairs across all deployments: {total_undefined}")
                    logger.info(f"Total selections of undefined pairs: {total_selections_undefined}")
                    logger.info(f"Total list appearances of undefined pairs: {total_lists_undefined}")
                    
                    # Mostrar detalle por deployment
                    logger.info("Undefined pairs by deployment:")
                    for _, row in df_undefined_pairs.iterrows():
                        if row['num_parejas_no_definidas'] > 0:
                            logger.info(f"  Deployment {row['deployment_id']}: {row['num_parejas_no_definidas']} parejas no definidas, "
                                      f"{row['num_veces_seleccionadas']} selecciones, {row['num_veces_listas']} apariciones en listas")
                else:
                    logger.info("No undefined pairs report generated")
                
                logger.info("[OK] All data saved to Excel file with multiple sheets")
                
        else:
            logger.error("Elasticsearch did not respond to ping.")
            
    except Exception as e:
        logger.exception(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
