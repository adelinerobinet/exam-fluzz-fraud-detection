# Image de base Python optimisée
FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Création du user non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copie et installation des requirements Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'application
COPY app/ ./app/

# Copie des modèles et données (si nécessaires)
# COPY ../data/processed/scalers.pkl ./app/models/
# COPY ../data/processed/metadata.pkl ./app/models/

# Création des répertoires nécessaires
RUN mkdir -p /app/logs /app/models

# Changement de propriétaire des fichiers
RUN chown -R appuser:appuser /app

# Basculement vers l'utilisateur non-root
USER appuser

# Port d'exposition
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Commande de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]