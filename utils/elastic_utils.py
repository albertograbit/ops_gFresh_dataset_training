import logging
import json
from elasticsearch import Elasticsearch, exceptions as es_exceptions
import pandas as pd
from typing import Any, Dict, Optional
import tempfile
import requests
import shutil
import os


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure logging for the script."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(filename=log_file, level=level, format=log_format)
    else:
        logging.basicConfig(level=level, format=log_format)


def download_file_from_url(url: str) -> str:
    """Download a file from a URL (e.g., SharePoint) to a temporary file and return its path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        suffix = os.path.splitext(url)[-1] or ''
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(response.raw, tmp_file)
            tmp_path = tmp_file.name
        logging.info(f"Downloaded file from {url} to {tmp_path}")
        return tmp_path
    except Exception as e:
        logging.error(f"Failed to download file from {url}: {e}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file or URL."""
    try:
        if config_path.startswith('http://') or config_path.startswith('https://'):
            config_path = download_file_from_url(config_path)
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config file '{config_path}': {e}")
        raise


def connect_es(host: str, username: str = '', password: str = '', verify_certs: bool = True, timeout: int = 60, auth_method: str = None, api_key_id: str = None, api_key_secret: str = None, ca_certs: str = None, port: int = 9200) -> Elasticsearch:
    """Establish a connection to Elasticsearch supporting both API key and basic auth."""
    try:
        protocol = "https" if verify_certs or ca_certs else "http"
        es_url = f"{protocol}://{host}:{port}"
        if auth_method == "api_key":
            if not api_key_secret:
                raise ValueError("API key secret is required for API key authentication.")
            es = Elasticsearch(
                es_url,
                api_key=api_key_secret,
                verify_certs=verify_certs,
                ca_certs=ca_certs
            )
        elif auth_method == "basic_auth" or (username and password):
            es = Elasticsearch(
                es_url,
                basic_auth=(username, password),
                verify_certs=verify_certs,
                ca_certs=ca_certs
            )
        else:
            raise ValueError("auth_method must be either 'api_key' or 'basic_auth', or username/password must be provided.")
        if not es.ping():
            raise es_exceptions.ConnectionError("Could not connect to Elasticsearch.")
        logging.info("Connected to Elasticsearch.")
        return es
    except Exception as e:
        logging.error("Failed to connect to Elasticsearch.")
        raise


def download_data(es: Elasticsearch, query_file: str, scroll: str = "2m") -> list:
    """Download data from Elasticsearch using a scroll query and a query file path (local or URL)."""
    try:
        if query_file.startswith('http://') or query_file.startswith('https://'):
            query_file = download_file_from_url(query_file)
        with open(query_file, "r", encoding="utf-8") as f:
            fichero_query = json.load(f)
        query_body = fichero_query["query_body"]
        index_pattern = fichero_query["index"]
        results = []
        page = es.search(index=index_pattern, body=query_body, scroll=scroll)
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
        results.extend(page['hits']['hits'])
        logging.info(f"Initial batch: {scroll_size} records.")
        while scroll_size > 0:
            page = es.scroll(scroll_id=sid, scroll=scroll)
            sid = page['_scroll_id']
            scroll_size = len(page['hits']['hits'])
            if scroll_size == 0:
                break
            results.extend(page['hits']['hits'])
            logging.info(f"Fetched {scroll_size} more records (total: {len(results)}).")
        try:
            es.clear_scroll(scroll_id=sid)
        except es_exceptions.NotFoundError as nf_err:
            logging.warning(f"Scroll ID not found or already cleared: {nf_err}")
        return results
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        raise


def convert_to_dataframe(hits: list) -> pd.DataFrame:
    """Convert Elasticsearch hits to a pandas DataFrame and add post-processing columns as expected by downstream scripts."""
    try:
        if not hits:
            return pd.DataFrame()
        records = [h['_source'] for h in hits if '_source' in h]
        df = pd.DataFrame(records)
        # Post-processing: fechas
        if 'transaction_start_time' in df.columns:
            df['transaction_start_time'] = pd.to_datetime(df['transaction_start_time'], errors='coerce')
            df['transaction_start_time'] = df['transaction_start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'transaction_end_time' in df.columns:
            df['transaction_end_time'] = pd.to_datetime(df['transaction_end_time'], errors='coerce')
            df['transaction_end_time'] = df['transaction_end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'transaction_data' in df.columns:
            df_transaction_data = df['transaction_data'].apply(pd.Series)
            df = pd.concat([df.drop(columns=['transaction_data']), df_transaction_data], axis=1)
        elif 'metric_data' in df.columns:
            df_metric_data = df['metric_data'].apply(pd.Series)
            df = pd.concat([df.drop(columns=['metric_data']), df_metric_data], axis=1)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['time'] = df['time'].dt.tz_localize(None).dt.date
        # Ordenar por transaction_start_time (si existe)
        if 'transaction_start_time' in df.columns:
            df.sort_values(by='transaction_start_time', inplace=True)
        # Reordenar columnas
        if 'transaction_start_time' in df.columns and 'transaction_end_time' in df.columns:
            columnas = ['transaction_start_time', 'transaction_end_time'] + [col for col in df.columns if col not in ('transaction_start_time', 'transaction_end_time')]
            df = df[columnas]
        # Añadir columna Date
        if 'transaction_start_time' in df.columns:
            df['transaction_start_time'] = pd.to_datetime(df['transaction_start_time'], errors='coerce')
            df['Date'] = df['transaction_start_time'].dt.date
            df['transaction_start_time'] = df['transaction_start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Añadir columna Result y columnas binarias
        def calcular_result(row):
            if 'confidence' in row and row['confidence'] == "LOW":
                return "No_result"
            elif 'topk' in row and row['topk'] == -1:
                return "Error"
            else:
                return "Ok"
        df['Result'] = df.apply(calcular_result, axis=1)
        df['Ok'] = (df['Result'] == 'Ok').astype(int)
        df['No_result'] = (df['Result'] == 'No_result').astype(int)
        df['Error'] = (df['Result'] == 'Error').astype(int)

        # --- NUEVO: calcular_result_modelo y columnas asociadas ---
        def calcular_result_modelo(row):
            # Verifica si existen las claves necesarias y cumplen condiciones
            metric = row.get('transaction_metric') or row.get('transaction_metric.keyword')
            if metric in ["WSPROCESSCOMPLETED03", "GFRSUSERPROCESSCOM03"]:
                # Acceso seguro a campos anidados
                selected_trained = None
                result_index = None
                selected_index = None
                # Buscar en transaction_data si existe
                if 'transaction_data.selected_reference_is_trained' in row:
                    selected_trained = row['transaction_data.selected_reference_is_trained']
                elif 'selected_reference_is_trained' in row:
                    selected_trained = row['selected_reference_is_trained']
                if 'transaction_data.result_index' in row:
                    result_index = row['transaction_data.result_index']
                elif 'result_index' in row:
                    result_index = row['result_index']
                if 'transaction_data.selected_reference_index' in row:
                    selected_index = row['transaction_data.selected_reference_index']
                elif 'selected_reference_index' in row:
                    selected_index = row['selected_reference_index']
                # Lógica de resultado
                if selected_trained is False or selected_trained == 0:
                    return "NOT_TRAINED"
                elif selected_trained is True or selected_trained == 1:
                    if result_index == selected_index:
                        return "SUCCESS"
                    else:
                        return "FAIL"
            return ""

        df['Result_modelo'] = df.apply(calcular_result_modelo, axis=1)
        df['Ok_modelo'] = (df['Result_modelo'] == 'SUCCESS').astype(int)

        logging.info(f"Converted {len(df)} records to DataFrame with post-processing columns.")
        return df
    except Exception as e:
        logging.error(f"Error converting hits to DataFrame: {e}")
        raise


def print_index_mapping(es: Elasticsearch, index: str) -> None:
    """Print the mapping of an Elasticsearch index."""
    try:
        mapping = es.indices.get_mapping(index=index)
        logging.info(f"Index mapping for '{index}': {json.dumps(mapping, indent=2)}")
    except Exception as e:
        logging.error(f"Error fetching index mapping: {e}")
        raise


def upload_file_to_url(local_path: str, url: str, method: str = "PUT", headers: dict = None) -> None:
    """Upload a file to a URL (e.g., SharePoint) using PUT or POST. Default is PUT. Raises on failure."""
    try:
        with open(local_path, 'rb') as f:
            if method.upper() == "PUT":
                response = requests.put(url, data=f, headers=headers)
            else:
                response = requests.post(url, files={'file': f}, headers=headers)
        response.raise_for_status()
        logging.info(f"Uploaded file {local_path} to {url} (status {response.status_code})")
    except Exception as e:
        logging.error(f"Failed to upload file to {url}: {e}")
        raise


def upload_file_to_sharepoint(local_path: str, sharepoint_url: str, site_url: str, folder_path: str, username: str = None, password: str = None) -> None:
    """
    Upload a file to SharePoint using Office365-REST-Python-Client.
    Requiere usuario y contraseña (sin MFA) por argumentos o variables de entorno SP_USER y SP_PASS.
    """
    try:
        from office365.sharepoint.client_context import ClientContext
        from office365.runtime.auth.user_credential import UserCredential
        import os
        if username is None:
            username = os.environ.get("SP_USER")
        if password is None:
            password = os.environ.get("SP_PASS")
        if not username or not password:
            raise ValueError("Debes definir usuario y contraseña de SharePoint por argumentos o variables de entorno SP_USER y SP_PASS.")
        filename = os.path.basename(local_path)
        ctx = ClientContext(site_url).with_credentials(UserCredential(username, password))
        target_folder = ctx.web.get_folder_by_server_relative_url(folder_path)
        with open(local_path, 'rb') as content_file:
            file_content = content_file.read()
        upload_file = target_folder.upload_file(filename, file_content).execute_query()
        logging.info(f"Uploaded file to SharePoint: {sharepoint_url}")
    except Exception as e:
        logging.error(f"Failed to upload file to SharePoint: {e}")
        raise


def upload_file_to_sharepoint_app(local_path: str, site_url: str, folder_path: str, config: dict = None):
    """
    Sube un archivo a SharePoint usando autenticación de aplicación (client_id/client_secret) leídos de un diccionario de configuración.
    El diccionario debe tener las claves: client_id, client_secret, tenant_id
    """
    try:
        from office365.sharepoint.client_context import ClientContext
        from office365.runtime.auth.client_credential import ClientCredential
        import os
        if config is None:
            raise ValueError("Se requiere un diccionario de configuración con client_id, client_secret y tenant_id.")
        client_id = config.get('client_id')
        client_secret = config.get('client_secret')
        tenant_id = config.get('tenant_id')
        if not all([client_id, client_secret, tenant_id]):
            raise ValueError("El fichero de configuración debe contener client_id, client_secret y tenant_id.")
        filename = os.path.basename(local_path)
        credentials = ClientCredential(client_id, client_secret)
        ctx = ClientContext(site_url).with_credentials(credentials)
        target_folder = ctx.web.get_folder_by_server_relative_url(folder_path)
        with open(local_path, 'rb') as content_file:
            file_content = content_file.read()
        print(f"Uploading {filename} to {site_url}{folder_path}")
        #quiero ver el tamaño del archivo
        print (f"Tamaño del archivo: {len(file_content)} bytes")
         
        upload_file = target_folder.upload_file(filename, file_content).execute_query()
        logging.info(f"Uploaded file to SharePoint: {site_url}{folder_path}/{filename}")
    except Exception as e:
        logging.error(f"Failed to upload file to SharePoint with app credentials: {e}")
        raise
