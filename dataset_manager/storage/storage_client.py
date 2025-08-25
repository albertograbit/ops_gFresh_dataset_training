"""Shared storage (S3 / MinIO) client utilities.

Centralises logic to:
 - Detect storage type (AWS S3 vs MinIO) from environment variables
 - Sanitize environment variables (strip quotes/extra spaces that cause auth errors)
 - Create a boto3 client with proper configuration (endpoint/path style for MinIO)
 - Perform a light validation of credentials/bucket access to fail fast with clearer diagnostics

Usage:
    from dataset_manager.storage.storage_client import get_storage_client
    s3_client, env = get_storage_client(logger)
    # env.bucket, env.storage_type, env.region available

Environment variable precedence (mirrors previous bespoke logic):
 - MINIO_* variables imply MinIO unless explicitly missing endpoint AND explicit REMOTE_STORAGE_* present.
 - REMOTE_STORAGE_ACCESS_KEY / SECRET_KEY or AWS default credentials imply S3.

Security: We DO NOT copy .env files; only reference them externally.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError


SANITIZE_PREFIXES = (
    'MINIO_', 'REMOTE_STORAGE_', 'S3_BUCKET', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'
)


@dataclass
class StorageEnv:
    storage_type: str  # 's3' | 'minio'
    bucket: str
    region: str
    access_key: Optional[str]
    secret_key: Optional[str]
    endpoint_url: Optional[str] = None
    secure: bool = True
    verify_ssl: bool = True


def _strip_quotes(value: Optional[str]) -> Optional[str]:
    if value is None:
        return value
    v = value.strip()
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1].strip()
    return v


def sanitize_environment(logger: Optional[logging.Logger] = None) -> None:
    """Strip surrounding quotes and spaces from relevant environment variables in-place."""
    for key, val in list(os.environ.items()):
        if any(key.startswith(pfx) for pfx in SANITIZE_PREFIXES):
            new_val = _strip_quotes(val)
            if new_val != val:
                os.environ[key] = new_val
                if logger:
                    logger.debug(f"Sanitized env var {key}")


def detect_storage_env(logger: Optional[logging.Logger] = None) -> StorageEnv:
    sanitize_environment(logger)

    # Detect MinIO first if explicit endpoint or credentials present
    minio_endpoint = os.getenv('MINIO_ENDPOINT')
    minio_access = os.getenv('MINIO_ACCESS_KEY') or os.getenv('MINIO_USER')
    minio_secret = os.getenv('MINIO_SECRET_KEY') or os.getenv('MINIO_PASSWORD')
    minio_bucket = os.getenv('MINIO_BUCKET')
    minio_secure = str(os.getenv('MINIO_SECURE', 'false')).lower() in ('1', 'true', 'yes', 'y')
    minio_verify = str(os.getenv('MINIO_VERIFY_SSL', 'true')).lower() in ('1','true','yes','y')

    remote_access = os.getenv('REMOTE_STORAGE_ACCESS_KEY')
    remote_secret = os.getenv('REMOTE_STORAGE_SECRET_KEY')
    s3_bucket_env = os.getenv('S3_BUCKET')
    region = (
        os.getenv('REMOTE_STORAGE_REGION') or
        os.getenv('AWS_REGION') or
        os.getenv('AWS_DEFAULT_REGION') or
        'eu-west-2'
    )

    if (minio_endpoint or minio_access) and minio_bucket:
        storage_type = 'minio'
        bucket = minio_bucket
        access = minio_access or remote_access  # allow fallback
        secret = minio_secret or remote_secret
        scheme = 'https' if minio_secure else 'http'
        endpoint_url = f"{scheme}://{minio_endpoint}" if minio_endpoint else None
        return StorageEnv(
            storage_type=storage_type,
            bucket=bucket,
            region=region,
            access_key=access,
            secret_key=secret,
            endpoint_url=endpoint_url,
            secure=minio_secure,
            verify_ssl=minio_verify,
        )

    # Default to S3
    bucket = s3_bucket_env or 'grabit-data'
    access = remote_access or os.getenv('AWS_ACCESS_KEY_ID')
    secret = remote_secret or os.getenv('AWS_SECRET_ACCESS_KEY')
    return StorageEnv(
        storage_type='s3',
        bucket=bucket,
        region=region,
        access_key=access,
        secret_key=secret,
    )


def create_boto3_client(env: StorageEnv, logger: Optional[logging.Logger] = None):
    try:
        if env.storage_type == 'minio':
            from botocore.config import Config as BotoConfig
            cfg = BotoConfig(signature_version='s3v4', s3={'addressing_style': 'path'})
            client = boto3.client(
                's3',
                region_name=env.region,
                aws_access_key_id=env.access_key,
                aws_secret_access_key=env.secret_key,
                endpoint_url=env.endpoint_url,
                config=cfg,
                verify=env.verify_ssl,
            )
        else:
            if env.access_key and env.secret_key:
                client = boto3.client(
                    's3',
                    region_name=env.region,
                    aws_access_key_id=env.access_key,
                    aws_secret_access_key=env.secret_key,
                )
            else:
                client = boto3.client('s3', region_name=env.region)
        return client
    except NoCredentialsError:
        if logger:
            logger.error('Credenciales no encontradas para crear cliente S3/MinIO')
        return None
    except Exception as e:
        if logger:
            logger.error(f'Error creando cliente de almacenamiento: {e}')
        return None


def validate_client(client, env: StorageEnv, logger: Optional[logging.Logger] = None) -> Tuple[bool, Optional[str]]:
    """Attempt a light validation to surface credential errors early.

    Returns (ok, error_message)
    """
    if client is None:
        return False, 'Cliente no inicializado'
    try:
        # head_bucket is cheaper; list objects might require permissions
        client.head_bucket(Bucket=env.bucket)
        return True, None
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code')
        msg = e.response.get('Error', {}).get('Message')
        if logger:
            logger.error(f'Validación de bucket falló ({code}): {msg}')
            logger.error('Verifique claves, bucket y tipo de almacenamiento (S3 vs MinIO).')
        return False, f'{code}: {msg}'
    except Exception as e:
        if logger:
            logger.error(f'Error validando bucket: {e}')
        return False, str(e)


def get_storage_client(logger: Optional[logging.Logger] = None, validate: bool = True):
    """High level helper returning (client, env).

    If validation fails returns (client, env) but logs error so caller can decide to proceed / abort.
    """
    env = detect_storage_env(logger)
    client = create_boto3_client(env, logger)
    if validate:
        ok, err = validate_client(client, env, logger)
        if not ok and logger:
            logger.warning(f'La validación del cliente de almacenamiento falló: {err}')
    if logger:
        logger.info(f"Almacenamiento detectado: type={env.storage_type} bucket={env.bucket} region={env.region} endpoint={env.endpoint_url or 'aws'}")
    return client, env


__all__ = [
    'StorageEnv',
    'get_storage_client',
    'detect_storage_env',
    'sanitize_environment',
]
