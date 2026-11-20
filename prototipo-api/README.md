# Prototipo: API + interfaz gráfica (sin terminar)

Siguiente fase del proyecto, que quedó a medias. La idea era dividir el programa en dos partes:

- **backend/**: un servidor con [FastAPI](https://fastapi.tiangolo.com/) que recibe un audio y devuelve la transcripción (`POST /api/v1/transcribe`) o la transcripción con hablantes (`POST /api/v1/diarize`).
- **frontend/**: una ventana de escritorio con Tkinter para elegir el audio, enviarlo al servidor y guardar el resultado.
- **docker/**: un contenedor para cada parte y un `docker-compose.yml` que los levanta juntos.

El código se generó con ayuda de Claude (la conversación está en `docs/prompt-contexto-claude.docx`) y se iba a subir a otro repositorio, que ya no existe. Esta carpeta es la única copia que queda. **No se ha llegado a probar.**

## Estado

- `backend/app/services/transcription.py` es una **versión simulada**: no carga Whisper, solo devuelve un texto de ejemplo. Servía para probar la API sin descargar modelos de varios GB.
- `backend/app/services/transcription_produccion.py` es la versión real con Whisper. Para usarla, hay que sustituir con ella el archivo anterior y descomentar Whisper en `backend/requirements.txt` y en `docker/Dockerfile.backend`.
- `backend/app/services/diarization.py` usa el pipeline `pyannote/speaker-diarization-3.0`. Necesita un token de Hugging Face en la variable de entorno `HF_TOKEN`.
- `frontend/src/gui.py` solo llama al endpoint `/transcribe`, todavía no a `/diarize`.
- `frontend/src/gui_v1_simulada.py` es la primera versión de la ventana, sin servidor: simula el proceso con una espera de 3 segundos.
- Faltan los `__init__.py`, los tests y `app/core/logging.py` que aparecen en `docs/estructura-prevista.txt`.

`README.original.md` es el README en inglés que se escribió para aquel repositorio.
