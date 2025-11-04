# Dockerfile para el Sistema de Diagnóstico Médico
# Desarrollado para el taller de Pipeline de MLOps + Docker

# Usar imagen base de Python oficial
FROM python:3.11-slim

# Metadatos de la imagen
LABEL maintainer="Felipe Guerra, Mavelyn Sterling"
LABEL description="Sistema de Diagnóstico Médico con MLOps Pipeline"
LABEL version="1.0.0"

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=5000

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copiar archivos de dependencias
COPY src/requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ .

# Crear directorio para logs
RUN mkdir -p /app/logs

# Cambiar permisos de archivos
RUN chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 5000

# Comando por defecto
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
