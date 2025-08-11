"""
Utilidades para conexión y consultas a la base de datos.
Maneja la conexión a MySQL y consultas relacionadas con imágenes de transacciones.
"""

import os
import ast
import logging
import mysql.connector
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Configurar logging
logger = logging.getLogger(__name__)

def load_db_config():
    """Carga la configuración de la base de datos desde variables de entorno."""
    load_dotenv()
    
    config = {
        'host': os.getenv("DB_PROD_RO_HOST"),
        'user': os.getenv("DB_PROD_RO_USER"), 
        'password': os.getenv("DB_PROD_RO_PASSWORD"),
        'database': os.getenv("DB_PROD_RO_DATABASE"),
        'port': int(os.getenv("DB_PROD_RO_PORT", 3306)),
        'use_pure': True
    }
    
    # Verificar que todas las variables estén configuradas
    missing_vars = [key for key, value in config.items() if value is None and key != 'port']
    if missing_vars:
        raise ValueError(f"Variables de entorno faltantes: {missing_vars}")
    
    return config

def get_db_connection():
    """
    Establece conexión con la base de datos MySQL.
    
    Returns:
        mysql.connector.connection: Conexión a la base de datos
        
    Raises:
        mysql.connector.Error: Si hay error en la conexión
        ValueError: Si faltan variables de entorno
    """
    try:
        config = load_db_config()
        conn = mysql.connector.connect(**config)
        logger.info("Conexión a la base de datos exitosa")
        return conn
    except mysql.connector.Error as err:
        logger.error("Error al conectar a la base de datos:")
        logger.error(f"Host: {os.getenv('DB_PROD_RO_HOST')}")
        logger.error(f"User: {os.getenv('DB_PROD_RO_USER')}")
        logger.error(f"Database: {os.getenv('DB_PROD_RO_DATABASE')}")
        logger.error(f"Mensaje de error: {err}")
        raise
    except Exception as e:
        logger.error(f"Error general de configuración: {e}")
        raise

def fetch_folder_data(transaction_ids: List[str], db_conn) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Obtiene información de carpetas e imágenes para una lista de transaction_ids.
    
    Args:
        transaction_ids: Lista de IDs de transacciones
        db_conn: Conexión a la base de datos
        
    Returns:
        Dict con transaction_id como clave y tuple (folder_url, folder_files) como valor
    """
    if not transaction_ids:
        return {}
    
    placeholders = ",".join(["%s"] * len(transaction_ids))
    query = f"""
        SELECT transaction_id, folder_url, folder_files
        FROM folder
        WHERE transaction_id IN ({placeholders})
    """
    
    try:
        cursor = db_conn.cursor()
        cursor.execute(query, transaction_ids)
        result = {str(row[0]): (row[1], row[2]) for row in cursor.fetchall()}
        cursor.close()
        
        logger.info(f"[STATS] Consultadas {len(transaction_ids)} transacciones, encontradas {len(result)} con datos de carpeta")
        return result
        
    except mysql.connector.Error as err:
        logger.error(f"[ERROR] Error en consulta SQL: {err}")
        raise

def parse_folder_files(folder_files_str: Optional[str]) -> List[str]:
    """
    Parsea la cadena folder_files que contiene una lista en formato string.
    
    Args:
        folder_files_str: String que contiene la lista de archivos
        
    Returns:
        Lista de nombres de archivos, vacía si hay error o es None
    """
    if not folder_files_str:
        return []
    
    try:
        files = ast.literal_eval(folder_files_str)
        return files if isinstance(files, list) else []
    except (ValueError, SyntaxError) as e:
        logger.warning(f"[WARNING] Error parseando folder_files: {folder_files_str}, error: {e}")
        return []

def get_image_info(transaction_id: str, folder_data: Dict[str, Tuple[Optional[str], Optional[str]]]) -> Tuple[int, int, str]:
    """
    Obtiene información de imágenes para un transaction_id específico.
    
    Args:
        transaction_id: ID de la transacción
        folder_data: Diccionario con datos de carpetas obtenido de fetch_folder_data
        
    Returns:
        Tuple con (has_image, num_images, image_link)
        - has_image: 1 si tiene imágenes, 0 si no
        - num_images: número de imágenes
        - image_link: URL de la carpeta o string vacío
    """
    data = folder_data.get(str(transaction_id))
    
    if not data:
        return 0, 0, ""
    
    folder_url, folder_files_str = data
    files = parse_folder_files(folder_files_str)
    num_images = len(files)
    has_image = 1 if num_images > 0 else 0
    image_link = folder_url if folder_url and has_image else ""
    
    return has_image, num_images, image_link

def add_image_columns_to_dataframe(df, transaction_id_column='transaction_id'):
    """
    Añade columnas de información de imágenes a un DataFrame.
    
    Args:
        df: DataFrame de pandas con columna de transaction_id
        transaction_id_column: Nombre de la columna que contiene los transaction_ids
        
    Returns:
        DataFrame con columnas añadidas: has_image, num_images, image_link
    """
    if transaction_id_column not in df.columns:
        logger.error(f"[ERROR] Columna '{transaction_id_column}' no encontrada en el DataFrame")
        # Añadir columnas vacías
        df['has_image'] = 0
        df['num_images'] = 0
        df['image_link'] = ""
        return df
    
    try:
        # Obtener lista única de transaction_ids
        transaction_ids = df[transaction_id_column].astype(str).unique().tolist()
        logger.info(f"[SEARCH] Consultando información de imágenes para {len(transaction_ids)} transacciones únicas...")
        
        # Conectar a la base de datos y obtener datos
        with get_db_connection() as db_conn:
            folder_data = fetch_folder_data(transaction_ids, db_conn)
        
        # Aplicar función para obtener información de imágenes
        def get_image_info_for_row(tid):
            return get_image_info(str(tid), folder_data)
        
        # Crear las columnas
        image_info = df[transaction_id_column].astype(str).map(get_image_info_for_row)
        df['has_image'] = image_info.map(lambda x: x[0])
        df['num_images'] = image_info.map(lambda x: x[1])
        df['image_link'] = image_info.map(lambda x: x[2])
        
        # Estadísticas
        total_with_images = (df['has_image'] == 1).sum()
        total_images = df['num_images'].sum()
        
        logger.info(f"[STATS] Estadísticas de imágenes:")
        logger.info(f"  - Transacciones con imágenes: {total_with_images}/{len(df)} ({total_with_images/len(df)*100:.1f}%)")
        logger.info(f"  - Total de imágenes: {total_images}")
        logger.info(f"  - Promedio de imágenes por transacción con imágenes: {total_images/max(total_with_images, 1):.1f}")
        
        return df
        
    except Exception as e:
        logger.error(f"[ERROR] Error añadiendo columnas de imágenes: {e}")
        # En caso de error, añadir columnas vacías
        df['has_image'] = 0
        df['num_images'] = 0
        df['image_link'] = ""
        return df

def test_database_connection():
    """
    Función de prueba para verificar la conexión a la base de datos.
    
    Returns:
        bool: True si la conexión es exitosa, False en caso contrario
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0] == 1:
                logger.info("[OK] Test de conexión exitoso")
                return True
            else:
                logger.error("[ERROR] Test de conexión falló - resultado inesperado")
                return False
                
    except Exception as e:
        logger.error(f"[ERROR] Test de conexión falló: {e}")
        return False

if __name__ == "__main__":
    # Test básico cuando se ejecuta directamente
    logging.basicConfig(level=logging.INFO)
    
    print("[TEST] Probando conexión a la base de datos...")
    if test_database_connection():
        print("[OK] Conexión exitosa")
    else:
        print("[ERROR] Error en la conexión")
