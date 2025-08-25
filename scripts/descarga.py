import argparse
import os
import pandas as pd
import boto3
from botocore.config import Config
import mysql.connector
import re
from pathlib import Path
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urlparse
from dotenv import load_dotenv

# 1) Cargar el .env indicado por el usuario (por defecto ../env)
def load_env_config(env_path: str):
    """Carga exclusivamente el .env indicado en env_path (ruta relativa al script) con pequeñas tolerancias.

    Fallbacks si no existe la ruta proporcionada:
    - Intenta ../.env y ../.env_walmart relative al script
    - Intenta ./.env y ./.env_walmart relative al script
    """
    if not env_path:
        raise ValueError("Debe indicarse la ruta del .env")

    base_dir = Path(__file__).resolve().parent
    p = Path(env_path)
    print(base_dir, p)
    if not p.is_absolute():
        p = (base_dir / env_path).resolve()

    candidates = [p]
    # Si no existe, probar alternativas habituales
    if not p.exists():
        candidates.extend([
            (base_dir.parent / ".env").resolve(),
            (base_dir.parent / ".env_walmart").resolve(),
            (base_dir / ".env").resolve(),
            (base_dir / ".env_walmart").resolve(),
        ])

    chosen = None
    for c in candidates:
        if c.exists():
            chosen = c
            break

    if not chosen:
        raise FileNotFoundError(f"No se encontró el archivo .env. Probado: {[str(c) for c in candidates]}")
    print(chosen)
    print(f"Cargando configuración desde: {chosen}")
    load_dotenv(str(chosen), override=True)
    return str(chosen)

# 2) Función para obtener configuración según el tipo de almacenamiento
def _parse_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().strip('"').strip("'").lower() in {"1", "true", "t", "yes", "y"}


def get_config():
    """Obtiene configuración desde el .env ya cargado."""
    return {
        "DB_PROD_RO_HOST": os.getenv("DB_PROD_RO_HOST"),
        "DB_PROD_RO_USER": os.getenv("DB_PROD_RO_USER"),
        "DB_PROD_RO_PASSWORD": os.getenv("DB_PROD_RO_PASSWORD"),
        "DB_PROD_RO_DATABASE": os.getenv("DB_PROD_RO_DATABASE"),
    # Credenciales: preferir REMOTE_*, luego MINIO_ACCESS_KEY/SECRET_KEY, luego MINIO_USER/PASSWORD
    "REMOTE_STORAGE_ACCESS_KEY": os.getenv("REMOTE_STORAGE_ACCESS_KEY")
    or os.getenv("MINIO_ACCESS_KEY")
    or os.getenv("MINIO_USER"),
    "REMOTE_STORAGE_SECRET_KEY": os.getenv("REMOTE_STORAGE_SECRET_KEY")
    or os.getenv("MINIO_SECRET_KEY")
    or os.getenv("MINIO_PASSWORD"),
    "REMOTE_STORAGE_REGION": os.getenv("REMOTE_STORAGE_REGION") or os.getenv("AWS_REGION") or "us-east-1",
        "S3_BUCKET": os.getenv("S3_BUCKET"),
        "MINIO_BUCKET": os.getenv("MINIO_BUCKET"),
        "MINIO_ENDPOINT": os.getenv("MINIO_ENDPOINT"),  # host:port si aplica
    "MINIO_SECURE": _parse_bool(os.getenv("MINIO_SECURE"), default=False),
    }


def detect_storage_type(cfg) -> str:
    """Detecta el tipo de almacenamiento a partir del .env cargado."""
    if cfg.get("MINIO_ENDPOINT") or os.getenv("MINIO_USER") or os.getenv("MINIO_ACCESS_KEY"):
        return "minio"
    return "s3"


def get_storage_client(cfg, storage_type="s3"):
    """Devuelve un cliente S3 (boto3) configurado.

    - Para S3: usa región y credenciales REMOTE_*
    - Para MinIO: usa endpoint http(s)://MINIO_ENDPOINT con credenciales REMOTE_* (o MINIO_* si es lo que hay en el .env)
    """
    endpoint_url = None
    verify_ssl = True
    if storage_type.lower() == "minio":
        if not cfg["MINIO_ENDPOINT"]:
            raise ValueError("Define MINIO_ENDPOINT en tu .env para usar MinIO (ej: host:9000)")
        try:
            host, port = str(cfg["MINIO_ENDPOINT"]).split(":")
        except ValueError:
            host, port = str(cfg["MINIO_ENDPOINT"]).strip(), None

        # MINIO_VERIFY_SSL optional override for self-signed certs
        verify_ssl = _parse_bool(os.getenv("MINIO_VERIFY_SSL"), default=True)
        # Respetar endpoint tal cual; solo avisar si es 9001
        if port == "9001":
            print("Aviso: MINIO_ENDPOINT usa 9001. Asegúrate de que MINIO_SECURE concuerda con el protocolo real del servicio (http vs https).")
        scheme = "https" if cfg["MINIO_SECURE"] else "http"
        endpoint_url = f"{scheme}://{cfg['MINIO_ENDPOINT']}"
    # Config común; MinIO funciona mejor con path-style y v4
    boto_cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    return boto3.client(
        "s3",
        aws_access_key_id=cfg["REMOTE_STORAGE_ACCESS_KEY"],
        aws_secret_access_key=cfg["REMOTE_STORAGE_SECRET_KEY"],
        region_name=cfg["REMOTE_STORAGE_REGION"],
        endpoint_url=endpoint_url,
        config=boto_cfg,
        verify=verify_ssl,
    )


# Conexión a la base de datos
def get_db_connection():
    try:
        # No cargar .env aquí ya que se carga en main()
        # Evitar imprimir credenciales en logs
        print("Conectando a la base de datos (lectura)...")

        conn = mysql.connector.connect(
            host=os.getenv("DB_PROD_RO_HOST"),
            user=os.getenv("DB_PROD_RO_USER"),
            password=os.getenv("DB_PROD_RO_PASSWORD"),
            database=os.getenv("DB_PROD_RO_DATABASE"),
            port=int(os.getenv("DB_PROD_RO_PORT", "3306")),
            use_pure=True,
        )
        print("Conexión a la base de datos exitosa.")
        return conn
    except Exception as err:
        print("Error al conectar a la base de datos:", err)
        raise


# Recupera los folder_url relativos
def fetch_folder_urls(transaction_ids, db_conn):
    placeholders = ",".join(["%s"] * len(transaction_ids))
    q = f"""
        SELECT t.transaction_id, f.folder_url
        FROM transaction t
        LEFT JOIN folder f USING (transaction_id)
        WHERE t.transaction_id IN ({placeholders})
    """
    cur = db_conn.cursor()
    cur.execute(q, transaction_ids)
    return dict(cur.fetchall())


def listar_buckets(cfg, storage_type="s3"):
    s3 = get_storage_client(cfg, storage_type)
    response = s3.list_buckets()
    print("Buckets disponibles:")
    for b in response.get("Buckets", []):
        print("-", b["Name"])


# Llama a esta función antes de tu flujo principal para comprobar acceso


# Descarga imágenes de S3/Minio
def download_images_from_s3_folder(
    storage_client, storage_url, local_dir, transaction_id, max_images=None
):
    parsed = urlparse(storage_url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"URL de almacenamiento inválida, bucket vacío en: {storage_url}")

    paginator = storage_client.get_paginator("list_objects_v2")
    counter = 1
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Si se especifica max_images y ya hemos descargado ese número, parar
            if max_images is not None and counter > max_images:
                print(f"    Límite alcanzado: {max_images} imágenes descargadas para ID '{transaction_id}'")
                return
                
            key = obj["Key"]
            ext = os.path.splitext(key)[1]
            if ext.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Si es la primera imagen, llamarla transaction_id_1.jpg, luego _2, etc.
            dest_filename = f"{transaction_id}_{counter}.jpg"
            dest_path = os.path.join(local_dir, dest_filename)

            storage_client.download_file(bucket, key, dest_path)
            counter += 1


def preflight_minio_api_check(storage_client):
    try:
        storage_client.list_buckets()
    except ClientError as e:
        msg = str(e)
        if "API port" in msg or "must be made to API port" in msg:
            print("Error: Estás conectando al puerto de la consola de MinIO. Debes apuntar al puerto del API S3 en MINIO_ENDPOINT o ajustar tu proxy para exponer el API en este puerto.")
        raise


# Flujo principal
def main(csv_path, base_output_dir, max_images=None, env_path: str = "../.env"):
    # 1) Cargar exclusivamente el .env indicado
    env_file_used = load_env_config(env_path)
    config = get_config()
    storage_type = detect_storage_type(config)
    
    # Obtener buckets según el tipo de almacenamiento
    BUCKET = config.get("S3_BUCKET")
    MINIO_BUCKET = config.get("MINIO_BUCKET")
    
    # 2) Carga CSV
    try:
        df = pd.read_csv(csv_path, sep=";") 
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {csv_path}")
        return
    except Exception as e:
        print(f"Error al cargar el CSV '{csv_path}': {e}")
        return

    if df.empty:
        print(f"El archivo CSV '{csv_path}' está vacío.")
        return

    if len(df.columns) == 0:
        print(f"El CSV '{csv_path}' no tiene columnas.")
        return
        
    transaction_id_col_name = df.columns[0]
    directory_cols_names = df.columns[1:].tolist()
    print(f"Columna de ID/Transacción: '{transaction_id_col_name}'")
    print(f"Columnas para directorios: {directory_cols_names}")

    # 3) Prepara clientes y conexión a BD
    try:
        # Usar la función get_storage_client que soporta S3 y Minio
        storage_client = get_storage_client(config, storage_type)
        db_conn = get_db_connection()
        
        # Determinar qué bucket usar según el tipo de almacenamiento
        if storage_type.lower() == "minio":
            bucket_name = MINIO_BUCKET if MINIO_BUCKET else BUCKET
            if not bucket_name:
                raise ValueError("Define MINIO_BUCKET en tu .env para usar Minio")
        else:
            bucket_name = BUCKET
            if not bucket_name:
                raise ValueError("Define S3_BUCKET en tu .env para usar S3")
                
        print(f"Usando almacenamiento: {storage_type.upper()}")
        print(f"Bucket: {bucket_name}")
        if storage_type.lower() == "minio":
            # Verificar que el endpoint apunta al API y no a la consola
            try:
                preflight_minio_api_check(storage_client)
            except ClientError:
                return
        
    except Exception as e:
        print(f"Error al inicializar clientes de almacenamiento/BD: {e}")
        return

    # 4) Obtén URLs de S3 (carpetas) desde la BD
    s3_relative_paths_map = {}
    try:
        transaction_ids_list = df[transaction_id_col_name].astype(str).unique().tolist()
        if transaction_ids_list:
            # fetch_folder_urls devuelve rutas relativas como 'folders/cliente/campana/id/'
            s3_relative_paths_map = fetch_folder_urls(transaction_ids_list, db_conn)
        else:
            print("No hay IDs de transacción en el CSV para procesar.")
            if 'db_conn' in locals() and db_conn.is_connected(): # type: ignore
                db_conn.close() # type: ignore
            return
    except Exception as e:
        print(f"Error al obtener URLs de S3 desde la BD: {e}")
    finally:
        if 'db_conn' in locals() and db_conn.is_connected(): # type: ignore
            db_conn.close() # type: ignore

    # 5) Itera sobre el CSV, construye rutas y descarga
    max_images_msg = f" (máximo {max_images} imágenes por carpeta)" if max_images else ""
    print(f"\\nIniciando proceso de descarga hacia el directorio base: '{base_output_dir}'{max_images_msg}")
    for index, row in df.iterrows():
        transaction_id = str(row[transaction_id_col_name])
        relative_s3_path = s3_relative_paths_map.get(transaction_id)

        if not relative_s3_path:
            print(f"Advertencia: No se encontró ruta S3 para ID '{transaction_id}'. Saltando.")
            continue

        # Construye la URL para el cliente y una URL amigable para logging
        full_storage_url = f"s3://{bucket_name}/{relative_s3_path.lstrip('/')}"
        if storage_type == "minio" and config.get("MINIO_ENDPOINT"):
            # Reflejar la URL final de endpoint usada por el cliente
            try:
                host, port = str(config["MINIO_ENDPOINT"]).split(":")
            except ValueError:
                host, port = str(config["MINIO_ENDPOINT"]).strip(), None
            scheme = "https" if config.get("MINIO_SECURE") else "http"
            log_url = f"minio://{scheme}://{config['MINIO_ENDPOINT']}/{bucket_name}/{relative_s3_path.lstrip('/')}"
        else:
            log_url = full_storage_url
        print(f"full_storage_url: {log_url}")

        path_components = []
        if not directory_cols_names:
            path_components.append(transaction_id)
        else:
            for col_name in directory_cols_names:
                if col_name in row and pd.notna(row[col_name]):
                    component = str(row[col_name]).strip()
                    # Limpiar componentes para que sean nombres de directorio válidos
                    component = re.sub(r"[^a-zA-Z0-9_.-]", "_", component)
                    if component:
                        path_components.append(component)

        current_download_dir = base_output_dir
        if path_components:
            current_download_dir = os.path.join(base_output_dir, *path_components)

        try:
            os.makedirs(current_download_dir, exist_ok=True)
        except OSError as e:
            print(f"Error al crear directorio '{current_download_dir}' para ID '{transaction_id}': {e}. Saltando.")
            continue

        print(f"  Procesando ID: '{transaction_id}'")
        print(f"    URL de almacenamiento (completa): {full_storage_url}")
        print(f"    Directorio local: {current_download_dir}")

        try:
            download_images_from_s3_folder(storage_client, full_storage_url, current_download_dir, transaction_id, max_images)
        except ValueError as ve:
            print(f"    Error de valor para ID '{transaction_id}' (URL: {full_storage_url}): {ve}")
        except NoCredentialsError:
            print(f"    Error: No se encontraron credenciales de almacenamiento. Asegúrate de que están configuradas.")
            return  # Detener si fallan las credenciales
        except ClientError as e:
            msg = str(e)
            if "API port" in msg or "must be made to API port" in msg:
                print("    Error: El endpoint de MinIO apunta a la consola. Cambia MINIO_ENDPOINT al puerto del API S3 o reajusta el proxy.")
                return
            print(f"    Error del cliente S3/MinIO para ID '{transaction_id}': {e}")
        except Exception as e:
            print(f"    Error al descargar imágenes para ID '{transaction_id}' (URL: {full_storage_url}): {e}")

    print(f"\\nProceso de descarga finalizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga imágenes de S3/Minio usando un CSV de transacciones (carga .env indicado)."
    )
    parser.add_argument(
        "-c",
        "--csv",
        default="./csv/CSV_LIMPIO.csv",
        help="Ruta al CSV de entrada",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="descargas",
        help="Directorio donde guardar las imágenes",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=1,
        help="Número máximo de imágenes a descargar por carpeta (si no se especifica, descarga todas)",
    )
    parser.add_argument(
        "--env",
    default="../.env",
    help="Ruta al archivo .env a usar (default: ../.env). Ej: ../.env_walmart",
    )
    args = parser.parse_args()
    main(args.csv, args.output, args.max_images, args.env)
