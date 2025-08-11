# Dataset Manager gFresh

CLI para analizar datos operativos, generar informes y gestionar el ciclo de vida de datasets (creación, fusión, resumen y registro en S3).

## 🎯 Objetivo

Automatizar selección y preparación de imágenes de producción para entrenar y versionar modelos de clasificación retail.

## 🏗️ Estructura

`dataset_manager/` contiene los módulos internos (extracción, análisis, creación de datasets, descarga de imágenes y reporting) expuestos vía `main.py`.

## 🚀 Funcionalidades

### 🔍 Análisis

- **Detección inteligente** de clases nuevas o no entrenadas
- **Análisis de devices** con métricas de rendimiento individuales
- **Identificación automática** de referencias sin `label_id` asignado
- **Cálculo de métricas** de % Ok del sistema y % Ok del modelo
- **Verificación de consistencia** con alertas automáticas
- **Análisis de dispositivos** con seguimiento de performance por device

### 🛠️ Procesamiento

- **Descarga automática** de datos de inferencia desde Elasticsearch
- **Extracción completa** de tablas `reference`, `label` y `model` desde BD
- **Identificación automática** del modelo activo en producción
- **Filtrado inteligente** por número mínimo de apariciones
- **Soporte para múltiples deployments** con gestión independiente

### 📈 Informes

**Excel principal** con múltiples pestañas especializadas:

- **`Resumen`**: Dashboard ejecutivo con métricas clave
- **`Datos_Elasticsearch`**: Datos raw de Elastic con toda la información
- **`References`**: Análisis completo de referencias con porcentajes Ok
- **`Devices`**: Análisis detallado por device con métricas individuales
- **`Labels`**: Información de etiquetas y clasificaciones
- **`Model_Data`**: Información del modelo expandida y configuración
- **`Consistencia`**: Verificaciones de integridad y alertas

### 🖼️ Revisión de Imágenes

**Nueva funcionalidad avanzada para descarga selectiva de imágenes:**

- **Descarga automática** de imágenes para referencias marcadas para revisión
- **Descarga de imágenes de devices** para análisis de rendimiento por dispositivo
- **Organización inteligente** por carpetas:
  - `productos/` → Imágenes de referencias de productos
  - `devices/{device_id}/` → Imágenes específicas por device
- **Logs detallados** en hojas Excel separadas:
  - `revisar_imagenes_bajadas` → Log detallado de referencias
  - `devices_imagenes_bajadas` → Log detallado de devices
- **Modo dry-run** para simulación sin descarga real
- **Selección inteligente** de transacciones representativas

### 🔄 Procesos

- **Sistema de procesos independientes** con metadata completa
- **Tracking automático** de configuraciones y resultados
- **Logs detallados** por proceso con timestamps
- **Reutilización** de configuraciones y datos entre sesiones

## 📦 Instalación

```bash
python -m venv venv
./venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env  # Rellenar credenciales
```

### Credenciales

1. **Variables de entorno (.env):**

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar con credenciales reales
DB_PROD_RO_HOST=your_mysql_host
DB_PROD_RO_USER=your_readonly_user
DB_PROD_RO_PASSWORD=your_readonly_password
DB_PROD_RO_DATABASE=your_database_name
```

1. **Elasticsearch:**

Crear `config/credentials/credentials_elastic_prod.json`:

```json
{
    "host": "tu-cluster.elasticsearch.com",
    "port": 9243,
    "verify_certs": true,
    "username": "tu_usuario",
    "password": "tu_password",
    "use_ssl": true
}
```

1. **AWS S3:**

Las credenciales de S3 se configuran automáticamente desde el archivo de Elasticsearch.

## 🖥️ Uso Rápido

```bash
# Activar entorno virtual
./venv/Scripts/activate  # Windows
source venv/bin/activate # Linux/Mac

# Descarga y genera Excel del deployment 130 (últimos 30 días)
python main.py download_info 130 --days-back 30

# Crear dataset (estructura de carpetas) a partir del proceso activo
python main.py create_dataset

# Resumen del dataset
python main.py summary_dataset -d dataset_130_010825_v1

# Registrar (ZIP + S3 + CSV global)
python main.py register_dataset -d dataset_130_010825_v1

# Fusionar varios datasets
python main.py merge_datasets

# Estructura de carpetas hoja
python main.py folder_structure
```

## 📜 Comandos

| Comando | Descripción | Parámetros clave |
|---------|-------------|------------------|
| `analyze <deployment_id>` | Extrae y analiza (sin Excel, sin imágenes). | `--days-back`, `--confidence`, `--min-appearances` |
| `download_info <deployment_id>` | Pipeline completo (datos + Excel). | `--days-back`, `--process-name`, filtros |
| `active-process` | Ver / listar / set / limpiar proceso activo. | `--list`, `--clear`, `--set` |
| `status` | Estado y configuración. |  |
| `review_images` | Descarga imágenes (referencias + devices) marcadas. | `--dry-run`, `-e <excel>` |
| `create_dataset` | Construye dataset (clase/label_id) desde Excel del proceso. | `--fast`, límites, `--no-filter-used` |
| `folder_structure` | CSV con carpetas hoja y conteo de imágenes. | `--input`, `--output` |
| `merge_datasets` | Une varios datasets en uno nuevo. | interactivo |
| `summary_dataset` | Genera `_summary_<dataset>.csv` cruzando imágenes con Excel. | `-d <dataset>` |
| `register_dataset` | ZIP + S3 + registro CSV + opcional subir datasets.csv a S3. | `-d`, `--s3-base`, `--registry-csv` |

Comandos eliminados: `crear_dataset`, `review_devices`, `folder`, `validate-config`, `process`.

### Ejemplos

```bash
# Ajustando confianza y nombre de proceso
python main.py download_info 130 --days-back 45 --confidence 0.78 --process-name dataset_130_aug_v2

# Dataset modo rápido
python main.py create_dataset --fast --limit-refs 25 --images-per-ref 12

# No filtrar transacciones ya usadas
python main.py create_dataset --no-filter-used

# Resumen y registro sin regenerar ZIP ni reemplazar S3
python main.py summary_dataset -d dataset_130_aug_v2
python main.py register_dataset -d dataset_130_aug_v2
```

## 📊 Outputs

### Directorios del Sistema

```text
output/
├── processes/                 # Procesos independientes
│   └── dataset_{id}_{date}_v1/
│       ├── reports/          # Archivos Excel generados
│       ├── images/           # Imágenes descargadas
│       │   ├── productos/    # Imágenes de referencias
│       │   └── devices/      # Imágenes por device
│       ├── data/            # Datos procesados
│       ├── logs/            # Logs detallados
│       └── process_metadata.json
└── legacy/                   # Archivos antiguos (compatibilidad)
```

### Archivos Excel Generados

#### Archivo Principal: `dataset_{deployment_id}_{timestamp}.xlsx`

**Pestañas principales:**

- **`Resumen`**: Métricas ejecutivas y KPIs principales
- **`Datos_Elasticsearch`**: Dataset completo con 26K+ registros
- **`References`**: 74 referencias analizadas con métricas
- **`Devices`**: 9 devices con análisis individual de rendimiento
- **`Labels`**: Clasificaciones y etiquetas del modelo
- **`Model_Data`**: Configuración del modelo activo
- **`Consistencia`**: Validaciones e inconsistencias detectadas

**Pestañas de seguimiento de imágenes:**

- **`revisar_imagenes_bajadas`**: Log detallado de imágenes de productos
- **`devices_imagenes_bajadas`**: Log detallado de imágenes de devices

#### Estructura de Datos en Pestañas

**References (ejemplo):**

- `reference_name`: Código de la referencia
- `label_name`: Nombre de la etiqueta asociada
- `total_transactions`: Total de transacciones
- `ok_percentage`: % de aciertos del sistema
- `model_ok_percentage`: % de aciertos del modelo
- `revisar_imagenes`: Marca para descarga de imágenes
- Métricas adicionales por top1, top2, top3

**Devices (ejemplo):**

- `device_id`: ID único del dispositivo
- `device_name`: Nombre del dispositivo
- `total_transactions`: Transacciones procesadas
- `ok_percentage`: Rendimiento del dispositivo
- `revisar_imagenes`: Marca para descarga de imágenes
- Métricas de precisión y errores

## ⚙️ Configuración (extracto)

### Variables de Entorno (.env)

```bash
# Base de datos MySQL (Solo lectura) - REQUERIDO
DB_PROD_RO_HOST=your_mysql_host
DB_PROD_RO_USER=your_readonly_user
DB_PROD_RO_PASSWORD=your_readonly_password
DB_PROD_RO_DATABASE=your_database_name
DB_PROD_RO_PORT=3306

# Configuración opcional
LOG_LEVEL=INFO
OUTPUT_BASE_DIR=./output
```

### Configuración de Descarga de Imágenes

La configuración se maneja automáticamente, pero se puede personalizar:

```yaml
image_review:
  num_imagenes_revision: 5              # Imágenes por referencia/device
  tipo_transacciones: "ambas"           # correctas|incorrectas|ambas
  clear_output_folder: true             # Limpiar carpeta antes de descargar
  tipo_imagenes_bajar: "clase_y_similares"  # Tipos de productos a descargar
  s3_bucket: "grabit-data"              # Bucket de S3
  s3_region: "eu-west-2"                # Región de S3
```

## 🔧 Flujo RecomENDADO

### 1. Análisis Inicial

```bash
# Generar análisis completo
python main.py download_info 125 --days-back 7

# Revisar archivo Excel generado
# Marcar referencias/devices para revisión
```

### 2. Revisión de Imágenes

```bash
# Simular descarga primero
python main.py review_images --dry-run

# Descargar imágenes reales
python main.py review_images

# Revisar imágenes descargadas en:
# output/processes/dataset_125_*/images/productos/
# output/processes/dataset_125_*/images/devices/
```

### 3. Análisis de Resultados

- Revisar logs detallados en las pestañas de Excel
- Analizar métricas de rendimiento por device
- Identificar patrones en las imágenes descargadas
- Tomar decisiones sobre inclusión en dataset

## 🔒 Seguridad

### Credenciales Seguras

- **Variables de entorno** para DB (archivo `.env`)
- **Archivos JSON separados** para servicios complejos
- **Exclusión completa** del control de versiones
- **Conexiones de solo lectura** para todas las fuentes

### Archivos Protegidos (en .gitignore)

```text
.env                          # Variables de entorno
config/credentials/           # Directorio de credenciales
logs/                         # Archivos de log
output/                       # Archivos de salida
venv/                         # Entorno virtual
```

## 🚨 Problemas Comunes

### Problemas Comunes

**Error de credenciales:**

```bash
python main.py status  # Verificar configuración
```

**Proceso activo no encontrado:**

```bash
python main.py active-process  # Ver estado actual
```

**Imágenes no se descargan:**

- Verificar credenciales de S3
- Comprobar que hay referencias/devices marcados con `revisar_imagenes=si`
- Usar `--dry-run` para simular primero

**Excel está abierto:**

- Cerrar archivo Excel antes de ejecutar comandos
- El sistema intentará 3 veces y creará backup si es necesario

### Logs y Diagnóstico

Los logs detallados se guardan en:

- `output/processes/{process_name}/logs/`
- Console output con timestamps
- Logs específicos por comando ejecutado

## 📋 Requisitos

- **Python 3.8+** (verificado automáticamente)
- **Acceso a Elasticsearch** con datos de inferencia
- **Acceso de lectura a MySQL** con tablas de referencia
- **Credenciales de AWS S3** para descarga de imágenes
- **Espacio en disco** para imágenes (aprox. 10MB por device/referencia)
- **Conexión a internet** estable

## 🔄 Cambios Clave

| Función | Estado |
|---------|--------|
| Renombrado `crear_dataset` → `create_dataset` | ✅ |
| `review_devices` integrado en `review_images` | ✅ |
| Nuevos: `folder_structure`, `merge_datasets`, `summary_dataset`, `register_dataset` | ✅ |
| Progreso ZIP y subida S3 en registro | ✅ |
| Subida opcional de `datasets.csv` a S3 | ✅ |
| Manejo de Excel corrupto / permisos (logs y merges) | ✅ |
| Columnas enriquecidas (origen, dataset_original, relative_folder, filename) | ✅ |

## 🛠️ Desarrollo

Para desarrollo:

```bash
# Clonar repositorio
git clone <repository_url>
cd ops_gFresh_dataset_training

# Configurar entorno de desarrollo
python setup.py
.\venv\Scripts\activate

# Instalar en modo desarrollo
pip install -e .

# Ejecutar pruebas
python -m pytest tests/ -v
```

### Extensión

- Usar `ProcessAwareSettings` para configuración
- Implementar logging con `ProcessLoggerManager`
- Seguir patrón de carpetas por proceso
- Documentar cambios en README

---

---
Autor: Alberto Gómez · Versión 2.1.0 · Agosto 2025

## 🗂️ Publicación en GitHub

```bash
git init
git add .
git commit -m "Initial cleaned version"
git branch -M main
git remote add origin git@github.com:TU_ORG/ops_gfresh_dataset_manager.git
git push -u origin main
```

Listo.
