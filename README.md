# Dataset Manager gFresh
* Python 3.8+
* Acceso a Elasticsearch (lectura)
* Acceso de lectura a MySQL
* Credenciales de AWS S3 (opcional para imágenes / registro)
* Espacio en disco suficiente (imágenes)
* Conexión a internet estable

## 🔄 Cambios Clave

* Renombrado `crear_dataset` → `create_dataset`.
* Integrado `review_devices` en `review_images`.
* Nuevos comandos: `folder_structure`, `merge_datasets`, `summary_dataset`, `register_dataset`.
* Multi‑entorno vía `--env` y archivos `.env_<nombre>` (posición flexible en CLI).
* Migración completa a `.env` (eliminado JSON Elastic). Soporte API Key única (`ELASTIC_API_KEY`).
* Limpieza automática de variables sensibles al cambiar de entorno.
* Centralización guardado Excel (`ReportExcelManager`) preservando dropdowns y gestionando bloqueo.
* Cache de datos Elastic en `reports/elastic_data.csv` dentro del proceso activo.
* Dataset filtering: métricas/validaciones restringidas a referencias incluidas (`incluir_dataset=si/sí`).
* Eliminado modo obsoleto `analyze`.
* Simplificación autenticación Elastic (api_key / basic_auth) + flags `ELASTIC_VERIFY_CERTS` / `ELASTIC_USE_SSL`.
* Eliminada creación de archivos Excel alternativos “pendientes”.

## 🛠️ Desarrollo & Contribución

```bash
# Clonar repositorio
git clone <repository_url>
cd ops_gFresh_dataset_training

# Crear entorno
python -m venv venv
./venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Copiar y editar variables
cp .env.example .env

# Ejecutar comando de prueba
python main.py status

# Instalar en editable (opcional)
# Dataset Manager gFresh

CLI para descargar datos de inferencia, generar informes y gestionar el ciclo de vida de datasets (creación, fusión, resumen y registro en S3).

## 🎯 Objetivo

Automatizar la selección y preparación de imágenes de producción para entrenar y versionar modelos de clasificación retail.

## 🏗️ Estructura

`dataset_manager/` contiene los módulos internos (extracción, análisis, creación de datasets, descarga de imágenes y reporting) expuestos vía `main.py`.

## 🚀 Principales Capacidades

1. Extracción y consolidación de datos (Elasticsearch + MySQL) en un Excel multi‑pestaña.
2. Marcado y descarga selectiva de imágenes (referencias y devices) con modo dry‑run.
3. Creación estructurada de datasets (clase/label_id) y utilidades: merge, summary, registro en S3.
4. Gestión de procesos aislados (cada ejecución crea un directorio autocontenido con logs, datos, imágenes y reports).
5. Multi‑entorno flexible mediante `--env` y archivos `.env_<nombre>`.

## 📦 Instalación Rápida

```bash
python -m venv venv
./venv/Scripts/activate  # Windows
pip install -r requirements.txt
cp .env.example .env
```

## 🔐 Credenciales y Multi‑entorno

Todas las credenciales se gestionan EXCLUSIVAMENTE vía archivos `.env`. Eliminado el uso de JSON de credenciales Elastic.

Archivos soportados:

```text
.env           # Por defecto
.env_walmart   # Cliente / entorno específico
.env_dev       # Desarrollo
.env_prod      # Producción
```

Ejemplos de uso (el parámetro `--env` puede ir en cualquier posición):

```bash
python main.py download_info 101 --env walmart --days-back 7
python main.py --env dev status
```

Si `.env_<env>` no existe se usa `.env` con aviso.

### Variables Base MySQL (lectura)

```dotenv
DB_PROD_RO_HOST=your_mysql_host
DB_PROD_RO_USER=readonly_user
DB_PROD_RO_PASSWORD=readonly_password
DB_PROD_RO_DATABASE=your_database
DB_PROD_RO_PORT=3306
```

### Elasticsearch (dos modos)

API Key (recomendado):

```dotenv
ELASTIC_HOST=cluster_id.region.aws.elastic-cloud.com
ELASTIC_PORT=9243
ELASTIC_AUTH_METHOD=api_key
ELASTIC_API_KEY=base64_api_key_unico
ELASTIC_VERIFY_CERTS=true
ELASTIC_USE_SSL=true
```

Basic Auth:

```dotenv
ELASTIC_HOST=cluster_id.region.aws.elastic-cloud.com
ELASTIC_PORT=9243
ELASTIC_AUTH_METHOD=basic_auth
ELASTIC_USERNAME=elastic_user
ELASTIC_PASSWORD=elastic_password
ELASTIC_VERIFY_CERTS=true
ELASTIC_USE_SSL=true
```

Notas:

* `ELASTIC_API_KEY` basta (no requiere id/secret separados).
* Si defines `ELASTIC_API_KEY` y no `ELASTIC_AUTH_METHOD`, se infiere `api_key`.
* `ELASTIC_VERIFY_CERTS=false` para entornos de prueba.
* Cambiar de entorno limpia variables `ELASTIC_*` y `DB_PROD_RO_*` previas.

### S3 (opcional para imágenes / registro)

```dotenv
REMOTE_STORAGE_ACCESS_KEY=...
REMOTE_STORAGE_SECRET_KEY=...
REMOTE_STORAGE_REGION=eu-west-2
S3_BUCKET=grabit-data
```

### Ejemplo mínimo `.env.example`

```dotenv
DB_PROD_RO_HOST=your_mysql_host
DB_PROD_RO_USER=readonly_user
DB_PROD_RO_PASSWORD=readonly_password
DB_PROD_RO_DATABASE=your_db
DB_PROD_RO_PORT=3306

ELASTIC_HOST=your-cluster.eu-west-3.aws.elastic-cloud.com
ELASTIC_PORT=9243
ELASTIC_AUTH_METHOD=api_key
ELASTIC_API_KEY=base64_api_key_value
ELASTIC_VERIFY_CERTS=true
ELASTIC_USE_SSL=true

S3_BUCKET=grabit-data
LOG_LEVEL=INFO
```

## 🖥️ Uso Rápido

```bash
./venv/Scripts/activate  # Windows
source venv/bin/activate # Linux/Mac

# Crear proceso y Excel (30 días por defecto) en entorno walmart
python main.py download_info 130 --days-back 30 --env walmart

# Crear dataset desde el proceso activo
python main.py create_dataset

# Resumen del dataset
python main.py summary_dataset -d dataset_130_010825_v1

# Registrar dataset (ZIP + S3 + CSV global)
python main.py register_dataset -d dataset_130_010825_v1

# Fusionar datasets
python main.py merge_datasets

# Estructura de carpetas hoja
python main.py folder_structure
```

## 📜 Comandos Principales

| Comando | Descripción |
|---------|-------------|
| `download_info <deployment_id>` | Extrae datos y genera Excel multi‑pestaña |
| `active-process` | Gestiona proceso activo (listar, set, clear) |
| `status` | Muestra configuración y paths |
| `review_images` | Descarga imágenes marcadas (productos + devices) |
| `create_dataset` | Crea estructura de dataset desde Excel |
| `folder_structure` | CSV con carpetas hoja e imágenes |
| `merge_datasets` | Fusiona varios datasets existentes |
| `summary_dataset` | CSV resumen cruzando imágenes con datos |
| `register_dataset` | Empaqueta, sube a S3 y registra dataset |

Parámetros útiles: `--days-back`, `--confidence`, `--min-appearances`, `--process-name`, `--fast`, `--limit-refs`, `--images-per-ref`, `--dry-run`, `--no-filter-used`.

## 📊 Outputs

### Estructura de procesos

```text
output/
 └─ processes/
     └─ dataset_{deployment}_{fecha}_vX/
         ├─ reports/
         ├─ images/
         │   ├─ productos/
         │   └─ devices/
         ├─ data/
         ├─ logs/
         └─ process_metadata.json
```

### Excel principal `dataset_{deployment_id}_{timestamp}.xlsx`

Pestañas: `Resumen`, `Datos_Elasticsearch`, `References`, `Devices`, `Labels`, `Model_Data`, `Consistencia`, `revisar_imagenes_bajadas`, `devices_imagenes_bajadas`.

**References (ejemplo columnas):** `reference_name`, `label_name`, `total_transactions`, `ok_percentage`, `model_ok_percentage`, `revisar_imagenes`, métricas top1/top2/top3.

**Devices (ejemplo columnas):** `device_id`, `device_name`, `total_transactions`, `ok_percentage`, `revisar_imagenes`, métricas de precisión y errores.

## ⚙️ Configuración Rápida (.env)

```bash
DB_PROD_RO_HOST=...
DB_PROD_RO_USER=...
DB_PROD_RO_PASSWORD=...
DB_PROD_RO_DATABASE=...
DB_PROD_RO_PORT=3306
```

### Descarga de Imágenes (settings.yaml)

```yaml
image_review:
  num_imagenes_revision: 5
  tipo_transacciones: "ambas"
  clear_output_folder: true
  tipo_imagenes_bajar: "clase_y_similares"
  s3_bucket: "grabit-data"
  s3_region: "eu-west-2"
```

## 🔧 Flujo Resumido

1. `download_info` → crea proceso + Excel.
2. Editar Excel: marcar `revisar_imagenes=si`.
3. `review_images` (opcional `--dry-run`).
4. `create_dataset` (opcional `--fast`).
5. `summary_dataset` y/o `merge_datasets`.
6. `register_dataset` (ZIP + S3 + registro).

## 🔒 Seguridad

**Principios:** uso exclusivo de `.env`, sólo lectura en BD, aislamiento por proceso, `.gitignore` protege artefactos y credenciales.

**Archivos ignorados clave:**

```text
.env*
output/
logs/
venv/
config/active_process.json
```

### Gestión de Excel Segura

`ReportExcelManager` controla:

* Reintentos si el archivo está abierto.
* Reaplicación de validaciones (dropdowns).
* Sin archivos alternativos “pendientes”.

## 🚨 Problemas Comunes

**Credenciales inválidas:** `python main.py status`.

**Proceso activo ausente:** `python main.py active-process`.

**Imágenes no descargadas:** ver marca `revisar_imagenes`, credenciales S3 y usar `--dry-run`.

**Excel bloqueado:** cerrar el archivo; el sistema reintenta automáticamente.

### Logs

Ubicación: `output/processes/<nombre>/logs/` + consola.

## 📋 Requisitos

* Python 3.8+
* Acceso lectura Elasticsearch
* Acceso lectura MySQL
* (Opcional) Credenciales AWS S3
* Espacio en disco suficiente
* Conexión a internet estable

## 🔄 Cambios Clave (Resumen)

* Renombrado `crear_dataset` → `create_dataset`.
* Integración de `review_devices` en `review_images`.
* Nuevos comandos: `folder_structure`, `merge_datasets`, `summary_dataset`, `register_dataset`.
* Multi‑entorno flexible (`--env`).
* Migración a `.env` + soporte `ELASTIC_API_KEY` único.
* Limpieza automática de variables sensibles al cambiar de entorno.
* Excel centralizado con preservación de validaciones.
* Cache Elastic en `reports/elastic_data.csv`.
* Dataset filtering (sólo referencias incluidas) para métricas.
* Eliminado modo obsoleto `analyze`.
* Simplificación autenticación Elastic (api_key / basic_auth).
* Eliminados archivos Excel alternativos temporales.

## 🛠️ Desarrollo

```bash
git clone <repository_url>
cd ops_gFresh_dataset_training
python -m venv venv
./venv/Scripts/activate  # Windows
pip install -r requirements.txt
cp .env.example .env
python main.py status
pip install -e .  # opcional editable
python -m pytest -v  # si hay tests
```

### Extensión

* Usar `ProcessAwareSettings` para configuración contextual.
* Logging unificado con `ProcessLoggerManager`.
* Respetar estructura por proceso.
* Documentar nuevos comandos / parámetros en este README.

---

Autor: Alberto Gómez · Versión 2.1.1 · Agosto 2025
