#!/usr/bin/env python3
"""
Whisper Diarizer para uso en contenedor Docker (o en local).
Transcribe audio con OpenAI Whisper e identifica quién habla en cada fragmento
(embeddings de SpeechBrain + clustering aglomerativo).

Adaptado del script original de Google Colab (whisper_diarizer.py).
"""

import argparse
import contextlib
import datetime
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import whisper
import yaml
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MODEL_SIZES = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
EMBEDDING_SIZE = 192  # dimensión de los embeddings de speechbrain/spkrec-ecapa-voxceleb


def parse_arguments():
    """Argumentos de línea de comandos. Lo que no se indique se toma de config.yaml."""
    parser = argparse.ArgumentParser(description="Whisper Diarizer: transcribe y separa hablantes")
    parser.add_argument("--input", "-i", help="Archivo de audio o carpeta (por defecto: input_folder de config.yaml)")
    parser.add_argument("--output", "-o", help="Carpeta de resultados (por defecto: output_folder de config.yaml)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Archivo de configuración YAML")
    parser.add_argument("--num_speakers", "-n", type=int, help="Número de hablantes (por defecto: estimación automática)")
    parser.add_argument("--language", "-l", help="Código de idioma, p. ej. es, en; 'any' para detección automática")
    parser.add_argument("--model_size", "-m", choices=MODEL_SIZES, help="Tamaño del modelo Whisper")
    return parser.parse_args()


def load_config(config_path):
    """Carga la configuración desde un archivo YAML (o {} si no existe o falla)."""
    if not os.path.exists(config_path):
        print(f"Archivo de configuración no encontrado: {config_path}. Se usan valores predeterminados.")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        print(f"Configuración cargada desde {config_path}")
        return config
    except Exception as e:
        print(f"Error al cargar la configuración: {e}")
        return {}


def get_input_files(input_path):
    """Lista de archivos de audio (como str) a partir de un archivo o una carpeta."""
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() in AUDIO_EXTENSIONS:
            return [str(input_path)]
        print(f"El archivo {input_path} no parece un audio compatible.")
        return []
    if input_path.is_dir():
        files = sorted(str(f) for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
        print(f"Se encontraron {len(files)} archivos de audio en {input_path}")
        return files
    print(f"La ruta de entrada no existe: {input_path}")
    return []


def is_wav_file(filepath):
    """True si el archivo es un WAV legible. wave.open necesita una ruta str, no Path."""
    try:
        with wave.open(str(filepath), "rb"):
            return True
    except (wave.Error, EOFError, OSError):
        return False


def convert_to_wav(input_path, output_path):
    """Convierte cualquier audio a WAV mono 16 kHz con ffmpeg."""
    print("🔄 Convirtiendo a WAV mono 16 kHz...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and is_wav_file(output_path):
        return True
    print(f"❌ La conversión falló: {result.stderr.strip()}")
    return False


def srt_time(seconds):
    """Formato de tiempo SRT: HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def short_time(seconds):
    return str(datetime.timedelta(seconds=round(seconds)))


def estimate_speakers(duration, min_speakers, max_speakers):
    """Estimación sencilla: un hablante más por cada minuto de audio, dentro de [min, max]."""
    return max(min_speakers, min(max_speakers, int(duration / 60) + 1))


def process_audio_file(file_path, output_dir, settings, whisper_model, embedding_model):
    """Transcribe y diariza un archivo. Guarda .txt, .json y .srt en output_dir."""
    print(f"\n{'=' * 50}\nProcesando: {file_path}\n{'=' * 50}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(file_path).stem

    # Se trabaja siempre sobre un WAV mono de 16 kHz
    temp_wav_path = os.path.join(output_dir, f"{base_name}_temp.wav")
    if is_wav_file(file_path):
        wav_path = file_path
    elif convert_to_wav(file_path, temp_wav_path):
        wav_path = temp_wav_path
    else:
        print(f"No se pudo procesar {file_path}.")
        return

    try:
        print("⏳ Transcribiendo audio...")
        language = settings["language"]
        result = whisper_model.transcribe(wav_path, language=None if language in (None, "any", "auto") else language)
        segments = result.get("segments", [])
        if not segments:
            print("🚨 No se detectó voz en el archivo.")
            return
        print(f"✅ Transcripción completada: {len(segments)} segmentos.")

        with contextlib.closing(wave.open(str(wav_path), "rb")) as f:
            duration = f.getnframes() / float(f.getframerate())
        print(f"📏 Duración: {duration:.2f} s")

        # Embedding de voz de cada segmento
        audio = Audio()
        embeddings = np.zeros((len(segments), EMBEDDING_SIZE))
        for i, segment in enumerate(segments):
            start, end = segment["start"], min(duration, segment["end"])
            try:
                waveform, _ = audio.crop(wav_path, Segment(start, end))
                waveform = torch.as_tensor(waveform, dtype=torch.float32)
                if waveform.ndim == 1:      # [muestras] -> [1, 1, muestras]
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.ndim == 2:    # [canales, muestras] -> [1, canales, muestras]
                    waveform = waveform.unsqueeze(0)
                if waveform.shape[1] > 1:   # a mono
                    waveform = waveform.mean(dim=1, keepdim=True)
                embeddings[i] = embedding_model(waveform)
            except Exception as e:
                print(f"🚨 Error en el segmento {start:.1f}-{end:.1f} s: {e}")
        embeddings = np.nan_to_num(embeddings)

        # Agrupar segmentos por hablante
        num_speakers = settings["num_speakers"]
        if not num_speakers:
            num_speakers = estimate_speakers(duration, settings["min_speakers"], settings["max_speakers"])
            print(f"Número de hablantes no indicado: se estiman {num_speakers}.")
        num_speakers = min(num_speakers, len(segments))  # no puede haber más grupos que segmentos

        if num_speakers > 1:
            labels = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings).labels_
        else:
            labels = np.zeros(len(segments), dtype=int)
        for segment, label in zip(segments, labels):
            segment["speaker"] = f"SPEAKER {int(label) + 1}"

        # Guardar resultados
        output_txt = os.path.join(output_dir, f"{base_name}_transcript.txt")
        output_json = os.path.join(output_dir, f"{base_name}_transcript.json")
        output_srt = os.path.join(output_dir, f"{base_name}_transcript.srt")

        with open(output_txt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    f.write(f"\n{segment['speaker']} [{short_time(segment['start'])}]\n")
                f.write(segment["text"].strip() + " ")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)

        with open(output_srt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                f.write(f"{i}\n{srt_time(segment['start'])} --> {srt_time(segment['end'])}\n")
                f.write(f"[{segment['speaker']}] {segment['text'].strip()}\n\n")

        print("✅ Resultados guardados en:")
        for path in (output_txt, output_json, output_srt):
            print(f"   - {path}")
    finally:
        if wav_path == temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def main():
    args = parse_arguments()
    config = load_config(args.config)

    # Prioridad: argumentos de línea de comandos > config.yaml > valores por defecto
    def pick(arg_value, key, default):
        return arg_value if arg_value is not None else config.get(key, default)

    input_path = pick(args.input, "input_folder", "./input")
    output_dir = pick(args.output, "output_folder", "./output")
    model_size = pick(args.model_size, "model_size", "large-v2")
    settings = {
        "language": pick(args.language, "language", "any"),
        "num_speakers": pick(args.num_speakers, "num_speakers", None),
        "min_speakers": config.get("min_speakers", 2),
        "max_speakers": config.get("max_speakers", 10),
    }

    input_files = get_input_files(input_path)
    if not input_files:
        print("No se encontraron archivos de audio para procesar.")
        return

    # Los modelos se cargan una sola vez para todos los archivos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏳ Cargando Whisper {model_size} en {device}...")
    whisper_model = whisper.load_model(model_size, device=device)
    print("⏳ Cargando el modelo de embeddings de voz...")
    embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)

    for file_path in input_files:
        process_audio_file(file_path, output_dir, settings, whisper_model, embedding_model)

    print("\n✅ Procesamiento completado para todos los archivos.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProceso interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        sys.exit(1)
