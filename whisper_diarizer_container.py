#!/usr/bin/env python3
"""
Whisper Diarizer para uso en contenedor Docker.
Adaptado del script original de Google Colab.
"""

import os
import sys
import subprocess
import datetime
import torch
import numpy as np
import whisper
import wave
import contextlib
import argparse
from pathlib import Path
import yaml
import json
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering

def install_if_missing(package, install_name=None):
    """
    Verifica si un paquete está instalado, y lo instala si falta.
    
    Parámetros:
    - package: Nombre del módulo en Python (ej. "torch")
    - install_name: Nombre usado para instalación (ej. "torch" o "git+https://github.com/openai/whisper.git")
    """
    import importlib.util
    if importlib.util.find_spec(package) is None:
        install_name = install_name or package
        print(f"Instalando {install_name}...")
        subprocess.call(["pip", "install", install_name])
    else:
        print(f"{package} ya está instalado.")

def check_dependencies():
    """Verifica si todas las dependencias están instaladas."""
    print("Verificando dependencias...")
    install_if_missing("whisper", "git+https://github.com/openai/whisper.git")
    install_if_missing("speechbrain")
    install_if_missing("torch")
    install_if_missing("torchvision")
    install_if_missing("torchaudio")
    install_if_missing("numpy")
    install_if_missing("ffmpeg")
    install_if_missing("pyannote", "git+https://github.com/pyannote/pyannote-audio")
    print("Todas las dependencias están instaladas correctamente.")

def parse_arguments():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Whisper Diarizer - Transcribe y diarize archivos de audio")
    parser.add_argument("--input", "-i", type=str, required=True, 
                      help="Ruta al archivo de audio de entrada o directorio con archivos de audio")
    parser.add_argument("--output", "-o", type=str, default="./output",
                      help="Directorio para guardar los resultados")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                      help="Ruta al archivo de configuración YAML")
    parser.add_argument("--num_speakers", "-n", type=int, default=None,
                      help="Número de hablantes (predeterminado: estimación automática)")
    parser.add_argument("--language", "-l", type=str, default="any",
                      help="Código del idioma (default: 'any' para detección automática)")
    parser.add_argument("--model_size", "-m", type=str, default="large-v2",
                      choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                      help="Tamaño del modelo Whisper (default: large-v2)")
    
    return parser.parse_args()

def load_config(config_path):
    """Carga la configuración desde un archivo YAML."""
    if not os.path.exists(config_path):
        print(f"Archivo de configuración no encontrado: {config_path}")
        print("Usando los valores predeterminados.")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuración cargada desde {config_path}")
        return config
    except Exception as e:
        print(f"Error al cargar configuración: {e}")
        return {}

def is_audio_file(file_path):
    """Verifica si el archivo es un archivo de audio compatible."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    return Path(file_path).suffix.lower() in audio_extensions

def get_input_files(input_path):
    """Obtiene una lista de archivos de audio desde un directorio o un solo archivo."""
    input_path = Path(input_path)
    
    if input_path.is_file():
        if is_audio_file(input_path):
            return [input_path]
        else:
            print(f"El archivo {input_path} no parece ser un archivo de audio compatible.")
            return []
    
    elif input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.is_file() and is_audio_file(f)]
        print(f"Se encontraron {len(files)} archivos de audio en {input_path}")
        return files
    
    else:
        print(f"La ruta de entrada no existe: {input_path}")
        return []

def ensure_directory(directory):
    """Asegura que un directorio exista."""
    os.makedirs(directory, exist_ok=True)

def is_wav_file(filepath):
    """Verifica si el archivo es un archivo WAV válido."""
    try:
        with wave.open(filepath, 'r') as f:
            print(f"✅ {filepath} es un archivo WAV válido.")
            return True
    except wave.Error as e:
        print(f"❌ {filepath} NO es un archivo WAV válido: {e}")
        return False

def convert_to_wav(input_path, output_path):
    """Convierte un archivo de audio a formato WAV."""
    print(f"🔄 Convirtiendo archivo a formato WAV...")
    try:
        subprocess.call(['ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output_path, '-y'])
        if is_wav_file(output_path):
            print(f"✅ Conversión exitosa: {output_path}")
            return True
        else:
            print(f"❌ La conversión falló.")
            return False
    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        return False

def process_audio_file(file_path, output_dir, num_speakers, language, model_size):
    """Procesa un archivo de audio para transcripción y diarización."""
    print(f"\n{'=' * 50}")
    print(f"Procesando: {file_path}")
    print(f"{'=' * 50}")
    
    # Crear directorio de salida
    ensure_directory(output_dir)
    
    # Preparar el nombre del archivo de salida
    base_name = Path(file_path).stem
    
    # Verificar si es un archivo WAV y convertirlo si es necesario
    temp_wav_path = os.path.join(output_dir, f"{base_name}_temp.wav")
    if not is_wav_file(file_path):
        if not convert_to_wav(file_path, temp_wav_path):
            print(f"No se pudo procesar {file_path} debido a errores en la conversión.")
            return
        converted_path = temp_wav_path
    else:
        converted_path = file_path
        
    # Cargar modelo Whisper
    print(f"⏳ Cargando modelo Whisper {model_size}...")
    model = whisper.load_model(model_size)
    print("✅ Modelo Whisper cargado.")
    
    # Transcribir
    print("⏳ Transcribiendo audio...")
    result = model.transcribe(
        converted_path, 
        language=None if language == "any" else language
    )
    segments = result.get("segments", [])
    
    if not segments:
        print("🚨 No se detectó voz en el archivo de audio.")
        return
    
    print(f"✅ Transcripción completada. Se detectaron {len(segments)} segmentos.")
    
    # Determinar duración del audio
    with contextlib.closing(wave.open(converted_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    
    print(f"📏 Duración del audio: {duration:.2f} segundos")
    
    # Cargar modelo de embedding de hablantes
    print("⏳ Cargando modelo de embedding de hablantes...")
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb", 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    audio = Audio()
    print("✅ Modelo de embedding de hablantes cargado.")
    
    # Función para extraer embedding de hablantes
    def segment_embedding(segment):
        try:
            start, end = segment["start"], min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(converted_path, clip)
            
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            
            if waveform.ndim == 1:  # Convertir tensor 1D a 3D
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:  # Convertir [channels, samples] a [1, channels, samples]
                waveform = waveform.unsqueeze(0)
            
            if waveform.shape[1] > 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            
            waveform = waveform.to(torch.float32)
            
            assert waveform.shape[1] == 1, "🚨 La forma de onda debe ser mono (1 canal)"
            
            return embedding_model(waveform)
        except Exception as e:
            print(f"🚨 Error al procesar segmento {segment['start']} - {segment['end']}: {e}")
            return np.zeros(192)  # Devolver un embedding ficticio
    
    # Calcular embeddings
    embeddings = np.zeros((len(segments), 192))
    for i, segment in enumerate(segments):
        print(f"🎤 Procesando segmento {i+1}/{len(segments)}: {segment['start']} - {segment['end']}")
        embeddings[i] = segment_embedding(segment)
    
    # Manejar valores NaN
    embeddings = np.nan_to_num(embeddings)
    
    # Realizar clustering de hablantes
    print("⏳ Realizando clustering de hablantes...")
    
    # Determinar el número de hablantes si no se especificó
    if num_speakers is None:
        # Método simple: estimar basado en la duración del audio
        estimated_speakers = max(2, min(10, int(duration / 60) + 1))
        print(f"Número de hablantes no especificado. Estimando {estimated_speakers} hablantes.")
        num_speakers = estimated_speakers
    
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings)
    labels = clustering.labels_
    print("✅ Clustering de hablantes completado.")
    
    # Asignar etiquetas de hablantes
    for i in range(len(segments)):
        segments[i]["speaker"] = f"SPEAKER {labels[i] + 1}"
    
    # Formatear marcas de tiempo
    def format_time(seconds):
        return str(datetime.timedelta(seconds=round(seconds)))
    
    # Guardar transcripción
    output_txt = os.path.join(output_dir, f"{base_name}_transcript.txt")
    output_json = os.path.join(output_dir, f"{base_name}_transcript.json")
    output_srt = os.path.join(output_dir, f"{base_name}_transcript.srt")
    
    print(f"💾 Guardando transcripción...")
    
    # Guardar formato texto
    with open(output_txt, "w") as f:
        for i, segment in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write(f"\n{segment['speaker']} [{format_time(segment['start'])}]\n")
            f.write(segment["text"].strip() + " ")
    
    # Guardar formato JSON
    with open(output_json, "w") as f:
        json.dump({"segments": segments}, f, indent=2)
    
    # Guardar formato SRT
    with open(output_srt, "w") as f:
        for i, segment in enumerate(segments, 1):
            start_time = datetime.timedelta(seconds=segment["start"])
            end_time = datetime.timedelta(seconds=segment["end"])
            
            # Formato SRT: HH:MM:SS,mmm
            start_str = str(start_time).replace(".", ",")
            if "." not in str(start_time):
                start_str += ",000"
            
            end_str = str(end_time).replace(".", ",")
            if "." not in str(end_time):
                end_str += ",000"
            
            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"[{segment['speaker']}] {segment['text'].strip()}\n\n")
    
    # Limpiar archivos temporales
    if temp_wav_path != file_path and os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)
        print(f"Archivo temporal eliminado: {temp_wav_path}")
    
    print(f"✅ Transcripción completada. Resultados guardados en:")
    print(f"   - {output_txt}")
    print(f"   - {output_json}")
    print(f"   - {output_srt}")

def main():
    """Función principal del programa."""
    # Verificar dependencias (desactivar en producción si ya están instaladas)
    # check_dependencies()
    
    # Analizar argumentos
    args = parse_arguments()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Priorizar argumentos de línea de comandos sobre configuración
    num_speakers = args.num_speakers
    language = args.language
    model_size = args.model_size
    
    # Sobrescribir con configuración si no se especificaron en los argumentos
    if config:
        if num_speakers is None and "num_speakers" in config:
            num_speakers = config["num_speakers"]
        if args.language == "any" and "language" in config:
            language = config["language"]
        if "model_size" in config:
            model_size = config["model_size"]
    
    # Obtener archivos de entrada
    input_files = get_input_files(args.input)
    
    if not input_files:
        print("No se encontraron archivos de audio para procesar.")
        return
    
    # Procesar cada archivo
    for file_path in input_files:
        process_audio_file(
            file_path, 
            args.output, 
            num_speakers, 
            language, 
            model_size
        )
    
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
