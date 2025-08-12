"""Utility helpers for safe file access (CSV/Excel) with SharePoint & Windows lock awareness.

Features:
 - Semicolon enforced CSV read/write (read_csv_sc / write_csv_sc)
 - File lock detection on Windows (is_file_locked)
 - Safe overwrite with backup (safe_overwrite)
 - Simple URL/SharePoint awareness (is_remote_path)

NOTE: SharePoint specific uploads/downloads are handled in existing sharepoint_utils.
This module just centralizes local file operations to avoid duplication.
"""

from __future__ import annotations

import os
import time
import shutil
import logging
from contextlib import contextmanager
import pandas as pd
from typing import Callable

LOGGER = logging.getLogger(__name__)
SEMICOLON = ';'

def is_remote_path(path: str) -> bool:
    """Return True if path looks like an HTTP(S) resource."""
    if not path:
        return False
    return path.lower().startswith(('http://', 'https://'))

def is_file_locked(path: str) -> bool:
    """Best‑effort detection if a file is locked (e.g. open in Excel on Windows)."""
    if not os.path.exists(path):
        return False
    try:
        if os.name == 'nt':  # Windows specific improved check
            import msvcrt  # type: ignore
            with open(path, 'a+b') as fh:  # binary append
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    return True
        else:
            with open(path, 'a'):
                pass
        return False
    except (PermissionError, OSError):
        return True

def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def read_csv_sc(path: str, **pandas_kwargs) -> pd.DataFrame:
    """Read a semicolon separated CSV enforcing delimiter consistency."""
    return pd.read_csv(path, sep=SEMICOLON, **pandas_kwargs)

def write_csv_sc(df: pd.DataFrame, path: str, index: bool = False, mode: str = 'w') -> None:
    ensure_parent_dir(path)
    df.to_csv(path, sep=SEMICOLON, index=index, mode=mode, encoding='utf-8')
    LOGGER.info(f"CSV escrito (delimitador ';'): {path} ({len(df)} filas)")

@contextmanager
def wait_for_writable(path: str, retries: int = 3, delay: float = 2.0):
    """Context manager that waits for a file to become writable (Excel closed)."""
    attempt = 0
    while attempt <= retries and is_file_locked(path):
        if attempt == retries:
            raise TimeoutError(f"Archivo bloqueado tras {retries} reintentos: {path}")
        LOGGER.warning(f"Archivo bloqueado, reintentando en {delay}s ({attempt+1}/{retries}) -> {path}")
        time.sleep(delay)
        attempt += 1
    yield

def safe_overwrite(path: str, write_fn: Callable[[str], None], backup: bool = True) -> None:
    """Safely overwrite a file using a temporary file then atomic replace."""
    ensure_parent_dir(path)
    directory = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    temp_path = os.path.join(directory, f".__tmp__{base}")
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass
    write_fn(temp_path)
    if backup and os.path.exists(path):
        bak = path + '.bak'
        try:
            shutil.copy2(path, bak)
        except Exception as e:
            LOGGER.warning(f"No se pudo crear backup {bak}: {e}")
    os.replace(temp_path, path)
    LOGGER.info(f"Archivo sobrescrito de forma segura: {path}")

def append_csv_sc(df: pd.DataFrame, path: str) -> None:
    """Append DataFrame to CSV (semicolon). Creates header only if file absent."""
    ensure_parent_dir(path)
    header = not os.path.exists(path)
    df.to_csv(path, sep=SEMICOLON, mode='a', header=header, index=False, encoding='utf-8')
    LOGGER.debug(f"Append {len(df)} filas -> {path}")

__all__ = [
    'is_remote_path', 'is_file_locked', 'read_csv_sc', 'write_csv_sc', 'append_csv_sc',
    'wait_for_writable', 'safe_overwrite'
]
