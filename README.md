# Whisper Diarizer en Docker

Este proyecto permite realizar transcripción de audio y diarización de hablantes (identificación de quién habla en cada momento) utilizando OpenAI Whisper y PyAnnote en un contenedor Docker con soporte para GPU.

## Funcionalidades

- Transcripción de audio con OpenAI Whisper
- Diarización de hablantes con PyAnnote
- Procesamiento acelerado por GPU
- Soporte para múltiples formatos de audio (.wav, .mp3, .flac, .ogg, .m4a)
- Salida en varios formatos (texto, SRT, JSON)
- Configuración flexible

## Requisitos previos

- Docker instalado en el sistema anfitrión
- Controladores NVIDIA y NVIDIA Container Toolkit para soporte de GPU
- Tarjeta gráfica compatible con CUDA

## Estructura del proyecto

```
.
├── Dockerfile              # Configuración para construir la imagen Docker
├── requirements.txt        # Dependencias de Python
├── whisper_diarizer.py     # Script original de Colab
├── whisper_diarizer_container.py  # Script adaptado para contenedor
├── config.yaml             # Configuración del sistema
└── run_docker.sh           # Script para facilitar la ejecución del contenedor
```

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuusuario/whisper-diarizer-docker.git
   cd whisper-diarizer-docker
   ```

2. Otorga permisos de ejecución al script de ayuda:
   ```bash
   chmod +x run_docker.sh
   ```

3. Construye la imagen Docker:
   ```bash
   ./run_docker.sh --build
   ```
   
   Alternativamente, puedes construirla directamente con Docker:
   ```bash
   docker build -t whisper_diarizer .
   ```

## Uso

### Utilizando el script de ayuda

El script `run_docker.sh` facilita la construcción y ejecución del contenedor:

```bash
# Ver ayuda y opciones disponibles
./run_docker.sh --help

# Procesar un solo archivo
./run_docker.sh --run --input ./mi_audio.mp3 --output ./resultados

# Procesar un directorio completo
./run_docker.sh --run --input ./mis_audios --output ./resultados

# Especificar idioma y modelo
./run_docker.sh --run --input ./mis_audios --output ./resultados --language es --model medium

# Especificar número de hablantes (si lo conoces)
./run_docker.sh --run --input ./mis_audios --output ./resultados --num-speakers 3
```

### Utilizando Docker directamente

También puedes ejecutar el contenedor directamente con Docker:

```bash
# Procesar un archivo
docker run --gpus all \
  -v "$(pwd)/mis_audios:/app/input" \
  -v "$(pwd)/resultados:/app/output" \
  whisper_diarizer \
  --input "/app/input/mi_audio.mp3" \
  --output "/app/output" \
  --language es \
  --model_size large-v2

# Procesar un directorio
docker run --gpus all \
  -v "$(pwd)/mis_audios:/app/input" \
  -v "$(pwd)/resultados:/app/output" \
  whisper_diarizer \
  --input "/app/input" \
  --output "/app/output" \
  --language es \
  --model_size large-v2
```

## Configuración

Puedes personalizar el comportamiento del sistema editando el archivo `config.yaml`:

```yaml
# Modelo de Whisper a utilizar
model_size: "large-v2"  # Opciones: tiny, base, small, medium, large, large-v2

# Idioma del audio
language: "es"  # Opciones: es (español), en (inglés), auto (detección automática)

# Número de hablantes (null para estimación automática)
num_speakers: null 

# Carpetas dentro del contenedor
input_folder: "/app/input"
output_folder: "/app/output"

# Formatos de salida
output_formats:
  - txt
  - json
  - srt
```

## Solución de problemas

### Verificar soporte de GPU

Para verificar que Docker pueda acceder a la GPU:

```bash
docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Deberías ver información sobre tu GPU si está configurada correctamente.

### Error "could not select device driver"

Si ves un error como:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```

Necesitas instalar NVIDIA Container Toolkit:

```bash
# Para Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Problemas con HuggingFace

Si ves errores relacionados con la descarga de modelos de HuggingFace, puedes:

1. Crear una cuenta en HuggingFace
2. Generar un token de acceso en https://huggingface.co/settings/tokens
3. Usar el token al ejecutar el contenedor:

```bash
docker run --gpus all \
  -v "$(pwd)/mis_audios:/app/input" \
  -v "$(pwd)/resultados:/app/output" \
  -e HUGGINGFACE_TOKEN="tu_token_aquí" \
  whisper_diarizer \
  --input "/app/input" \
  --output "/app/output"
```

## Notas adicionales

- La primera ejecución puede ser lenta mientras se descargan los modelos
- Los modelos más grandes (como "large-v2") ofrecen mejor precisión pero requieren más memoria GPU
- Si tienes poca memoria GPU, usa modelos más pequeños como "base" o "small"

## Licencia

Este proyecto utiliza varias bibliotecas de código abierto:
- [OpenAI Whisper](https://github.com/openai/whisper) - Licencia MIT
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio) - Licencia MIT

## Atribuciones

Este proyecto fue adaptado del script original para Google Colab y mejorado para funcionar en contenedores Docker.