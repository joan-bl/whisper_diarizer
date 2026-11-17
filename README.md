# Whisper Diarizer en Docker

Transcribe audio con [OpenAI Whisper](https://github.com/openai/whisper) e identifica **quién habla en cada fragmento** (diarización). Funciona en un contenedor Docker con GPU NVIDIA.

## Cómo funciona

1. Si el audio no es WAV, se convierte a WAV mono de 16 kHz con ffmpeg.
2. Whisper transcribe el audio en segmentos.
3. Para cada segmento se calcula una "huella de voz" con el modelo `speechbrain/spkrec-ecapa-voxceleb`, cargado desde pyannote.audio. No necesita token de Hugging Face.
4. Los segmentos se agrupan por hablante con clustering aglomerativo. Si no indicas cuántos hablantes hay, se estima uno por cada minuto de audio, entre `min_speakers` y `max_speakers`. Esta estimación es muy aproximada: **si sabes cuántos hablantes hay, indícalo** con `-n`.
5. Se guardan tres archivos por audio: `_transcript.txt` (agrupado por hablante), `.json` y `.srt` (subtítulos).

Formatos admitidos: `.wav`, `.mp3`, `.flac`, `.ogg` y `.m4a`.

## Requisitos

- Docker.
- Una GPU NVIDIA con sus controladores y el [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). También funciona sin GPU, pero mucho más lento.

La imagen ocupa unos 16 GB. La primera ejecución descarga además los modelos.

## Uso

```bash
git clone https://github.com/joan-bl/whisper_diarizer.git
cd whisper_diarizer
docker build -t whisper_diarizer .
```

**Procesar todos los audios de una carpeta**, con los valores de `config.yaml`:

```bash
docker run --gpus all \
  -v "$(pwd)/mis_audios:/app/input" \
  -v "$(pwd)/resultados:/app/output" \
  whisper_diarizer
```

**Con opciones:**

```bash
docker run --gpus all \
  -v "$(pwd)/mis_audios:/app/input" \
  -v "$(pwd)/resultados:/app/output" \
  whisper_diarizer --input /app/input/entrevista.mp3 -n 2 -l es -m medium
```

| Opción | Qué hace | Por defecto |
|---|---|---|
| `-i`, `--input` | Archivo o carpeta de audio | `input_folder` de `config.yaml` |
| `-o`, `--output` | Carpeta de resultados | `output_folder` de `config.yaml` |
| `-n`, `--num_speakers` | Número de hablantes | Estimación automática |
| `-l`, `--language` | Idioma (`es`, `en`…; `any` para detección automática) | `language` de `config.yaml` |
| `-m`, `--model_size` | `tiny`, `base`, `small`, `medium`, `large`, `large-v2` o `large-v3` | `model_size` de `config.yaml` |
| `-c`, `--config` | Otro archivo de configuración | `config.yaml` |

Las opciones de la línea de comandos tienen prioridad sobre `config.yaml`.

Para probarlo hay un audio de ejemplo en `input/VocesVarias.m4a`.

## Sin Docker

Con Python 3.10, ffmpeg y PyTorch instalados:

```bash
pip install -r requirements.txt
python whisper_diarizer_container.py -i input -o output -n 2
```

## Estructura

```
whisper_diarizer_container.py   # Script principal
config.yaml                     # Configuración por defecto
Dockerfile, requirements.txt
input/                          # Audio de ejemplo
docs/                           # Script original de Colab, plan de integración y notas de la imagen
```

## Solución de problemas

- **Comprobar que Docker ve la GPU:** `docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`
- **`could not select device driver "" with capabilities: [[gpu]]`:** falta el NVIDIA Container Toolkit (ver el enlace de Requisitos).
- **Falta memoria en la GPU:** usa un modelo más pequeño (`-m small` o `-m medium`).

## Licencias de terceros

[OpenAI Whisper](https://github.com/openai/whisper) (MIT), [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT) y [SpeechBrain](https://github.com/speechbrain/speechbrain) (Apache 2.0).
