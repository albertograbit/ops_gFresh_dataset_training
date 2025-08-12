import os
import time
import logging
from typing import Optional, Callable
from contextlib import contextmanager
from utils.excel_validations import ensure_dropdown_validations

try:
    import openpyxl
except ImportError:  # graceful degrade
    openpyxl = None

LOGGER = logging.getLogger(__name__)

class ExcelLockedError(Exception):
    pass

class ReportExcelManager:
    """Gestión centralizada del archivo Excel del dataset (report).

    Responsabilidades:
     - Verificar bloqueo (archivo abierto) antes de operar.
     - Ofrecer retry interactivo (cerrar y pulsar Enter) opcional.
     - Guardar modificaciones preservando validaciones (reaplica ensure_dropdown_validations).
     - Evitar duplicar lógica de comprobación dispersa.
    """

    def __init__(self, excel_path: str, interactive: bool = True, max_retries: int = 3, retry_wait: float = 2.0):
        self.excel_path = excel_path
        self.interactive = interactive
        self.max_retries = max_retries
        self.retry_wait = retry_wait

    def exists(self) -> bool:
        return os.path.exists(self.excel_path)

    def _is_locked(self) -> bool:
        if not self.exists():
            return False
        temp = self.excel_path + ".locktest"
        try:
            os.rename(self.excel_path, temp)
            os.rename(temp, self.excel_path)
            return False
        except PermissionError:
            return True
        except OSError:
            # Lo tratamos como bloqueo probable
            return True
        except Exception:
            return True

    def wait_until_unlocked(self) -> None:
        """Intentar esperar/solicitar cierre si está bloqueado."""
        attempts = 0
        while self._is_locked():
            attempts += 1
            if not self.interactive or attempts > self.max_retries:
                raise ExcelLockedError(f"El archivo sigue bloqueado tras {attempts-1} reintentos")
            input(f"⚠ El Excel '{os.path.basename(self.excel_path)}' está abierto. Ciérralo y pulsa Enter para continuar...")
            time.sleep(self.retry_wait)

    @contextmanager
    def open_workbook(self):
        """Context manager que asegura que el archivo no esté bloqueado antes de abrir.
        Devuelve el workbook cargado con openpyxl (si disponible)."""
        self.wait_until_unlocked()
        if openpyxl is None:
            raise RuntimeError("openpyxl no instalado; no se puede abrir el workbook")
        wb = openpyxl.load_workbook(self.excel_path)
        try:
            yield wb
        finally:
            # Nada especial aquí
            pass

    def save_with_validations(self, modify_fn: Optional[Callable[[object], None]] = None) -> None:
        """Abrir, opcionalmente modificar (callback) y guardar reforzando validaciones.

        modify_fn: función que recibe el workbook para aplicar cambios (crear/actualizar hojas, etc.)
        """
        self.wait_until_unlocked()
        if openpyxl is None:
            raise RuntimeError("openpyxl no instalado; no se puede guardar el workbook")
        wb = openpyxl.load_workbook(self.excel_path)
        if modify_fn:
            modify_fn(wb)
        wb.save(self.excel_path)
        try:
            ensure_dropdown_validations(self.excel_path)
        except Exception as e:
            LOGGER.warning(f"No se pudieron reforzar validaciones tras guardar: {e}")

__all__ = ["ReportExcelManager", "ExcelLockedError"]
