import json
import logging
import os
import shutil
from urllib.parse import urlparse, unquote
from pathlib import Path

def parse_sharepoint_url(output_url):
    """Extrae site_url, folder_path y filename de una URL de SharePoint."""
    parsed = urlparse(output_url)
    path_parts = parsed.path.split('/')
    try:
        site_idx = path_parts.index('sites')
        site_name = path_parts[site_idx + 1]
        site_url = f"{parsed.scheme}://{parsed.netloc}/sites/{site_name}/"
        folder_path = '/' + '/'.join(path_parts[1:-1])
        filename = path_parts[-1]
    except (ValueError, IndexError):
        raise ValueError("No se pudo parsear correctamente la URL de SharePoint. Revisa el formato.")
    return site_url, unquote(folder_path), unquote(filename)


def upload_file_to_sharepoint_app(local_path, site_url, folder_path, config):
    """
    Sube un archivo a SharePoint usando autenticación de aplicación (client_id/client_secret) leídos de un diccionario de configuración.
    El diccionario debe tener las claves: client_id, client_secret, tenant_id
    """
    try:
        from office365.sharepoint.client_context import ClientContext
        from office365.runtime.auth.client_credential import ClientCredential
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
        upload_file = target_folder.upload_file(filename, file_content).execute_query()
        logging.info(f"Uploaded file to SharePoint: {site_url}{folder_path}/{filename}")
    except Exception as e:
        logging.error(f"Failed to upload file to SharePoint with app credentials: {e}")
        raise

def save_excel_to_sharepoint(escribir_excel_func, args, deployments, devices, products, output_url, cred_path):
    """
    Genera un Excel temporal y lo sube a SharePoint usando la configuración y utilidades del proyecto.
    """
    import tempfile
    site_url, folder_path, real_filename = parse_sharepoint_url(output_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_xlsx:
        escribir_excel_func(Path(tmp_xlsx.name), args, deployments=deployments, devices=devices, products=products)
        temp_named_path = os.path.join(os.path.dirname(tmp_xlsx.name), real_filename)
        shutil.copyfile(tmp_xlsx.name, temp_named_path)
        with open(cred_path, encoding='utf-8') as credf:
            sp_config = json.load(credf)
        upload_file_to_sharepoint_app(
            local_path=temp_named_path,
            site_url=site_url,
            folder_path=folder_path,
            config=sp_config
        )
        os.remove(temp_named_path)
    logging.info(f"[OK] Informe subido a SharePoint en '{output_url}'.")

def save_file_to_sharepoint(write_func, output_url, cred_path, *args, **kwargs):
    """
    Guarda un archivo temporal usando write_func y lo sube a SharePoint.
    - write_func: función que escribe el archivo (ej: escribir_excel, escribir_csv, etc.)
    - output_url: URL de SharePoint destino (incluye nombre de archivo y extensión)
    - cred_path: ruta al JSON de credenciales
    - *args, **kwargs: argumentos para la función de escritura
    """
    import tempfile
    site_url, folder_path, real_filename = parse_sharepoint_url(output_url)
    suffix = Path(real_filename).suffix or '.tmp'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        write_func(tmp_path, *args, **kwargs)
        temp_named_path = tmp_path.parent / real_filename
        shutil.copyfile(tmp_path, temp_named_path)
        with open(cred_path, encoding='utf-8') as credf:
            sp_config = json.load(credf)
        upload_file_to_sharepoint_app(
            local_path=str(temp_named_path),
            site_url=site_url,
            folder_path=folder_path,
            config=sp_config
        )
        os.remove(temp_named_path)
    logging.info(f"[OK] Archivo subido a SharePoint en '{output_url}'.")

# =====================
# UNIVERSAL SHAREPOINT UPLOAD UTILITY
# =====================
def upload_file_to_sharepoint_app_v2(local_path, output_url, cred_path):
    """
    Upload any local file to SharePoint using an Azure AD app (client_id/client_secret).

    Parameters:
    - local_path (str or Path): Path to the local file to upload (absolute or relative).
    - output_url (str): Full SharePoint destination URL (including filename and extension).
    - cred_path (str or Path): Path to the JSON file with SharePoint app credentials (must contain client_id, client_secret, tenant_id).

    This function is portable and does not depend on project structure. It will rename the file temporarily if needed to match the SharePoint filename.
    """
    import os, shutil, tempfile, json
    from pathlib import Path
    local_path = os.path.abspath(str(local_path))
    cred_path = os.path.abspath(str(cred_path))
    site_url, folder_path, real_filename = parse_sharepoint_url(output_url)
    with open(cred_path, encoding='utf-8') as credf:
        sp_config = json.load(credf)
    src_path = local_path
    if os.path.basename(local_path) != real_filename:
        tmp_dir = tempfile.gettempdir()
        temp_named_path = os.path.join(tmp_dir, real_filename)
        shutil.copyfile(local_path, temp_named_path)
        src_path = temp_named_path
    upload_file_to_sharepoint_app(
        local_path=src_path,
        site_url=site_url,
        folder_path=folder_path,
        config=sp_config
    )
    if src_path != local_path:
        os.remove(src_path)
    logging.info(f"[OK] Archivo subido a SharePoint en '{output_url}'.")
