"""
Pipeline de Détection de Fraude Bancaire - Néobanque Fluzz
===========================================================

Module principal contenant les composants du pipeline de machine learning
pour la détection automatique des fraudes bancaires.

Modules disponibles:
    - utils: Fonctions utilitaires et validation
    - data_processing: Traitement et preprocessing des données
    - model_training: Entraînement et évaluation des modèles ML
    - synthetic_data: Génération de données synthétiques

Auteur: Équipe Data Science - Néobanque Fluzz
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Équipe Data Science - Néobanque Fluzz"
__email__ = "data-science@fluzz.com"

# Import des modules principaux
from . import utils
from . import data_processing
from . import model_training  
from . import synthetic_data

__all__ = [
    "utils",
    "data_processing", 
    "model_training",
    "synthetic_data"
]