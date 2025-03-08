# Usa una imagen base con Python y CUDA para acelerar procesamiento
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto a la imagen
COPY . /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto si necesitas servir la aplicación (ejemplo Flask)
# EXPOSE 5000

# Comando de ejecución (ajústalo si es necesario)
CMD ["python3", "app.py"]
