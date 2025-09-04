"""
Configuration du Pipeline de Détection de Fraude Bancaire
===================================================================

Auteur: Équipe Data Science - Néobanque Fluzz
Version: 1.0 (Simplified)
Date: Août 2025
"""

from pathlib import Path
from typing import Dict, Any

# =============================================================================
# CONFIGURATION GÉNÉRALE DU PROJET
# =============================================================================

PROJECT_CONFIG = {
    "name": "fraud_detection_pipeline",
    "version": "1.0.0",
    "description": "Pipeline ML de détection de fraude bancaire pour Néobanque Fluzz",
    "author": "Équipe Data Science - Fluzz",
    "python_version_min": "3.8"
}

# =============================================================================
# CHEMINS ET STRUCTURE DE DONNÉES
# =============================================================================

def get_project_root() -> Path:
    """Retourne le chemin racine du projet."""
    return Path(__file__).parent.parent

# Chemins utilisés par le projet
DATA_PATHS = {
    "root": get_project_root(),
    "data": {
        "raw": get_project_root() / "data" / "raw",
        "processed": get_project_root() / "data" / "processed", 
        "synthetic": get_project_root() / "data" / "synthetic"
    },
    "models": {
        "generators": get_project_root() / "models" / "generators"
    },
    "reports": get_project_root() / "reports"
}

# =============================================================================
# CONFIGURATION DES DONNÉES
# =============================================================================

DATA_CONFIG = {
    # Fichier principal des données
    "source_file": "creditcard.csv",
    "target_column": "Class",
    
    # Schema des données attendu
    "expected_columns": {
        "Time": "float64",
        "Amount": "float64", 
        "Class": "int64",
        **{f"V{i}": "float64" for i in range(1, 29)}
    },
    
    # Contraintes de validation
    "validation": {
        "min_rows": 1000,
        "max_missing_percentage": 0.05,
        "required_columns": ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)],
        "class_values": [0, 1],
        "time_range": {"min": 0, "max": 200000},
        "amount_range": {"min": 0, "max": 30000}
    },
    
    # Configuration du déséquilibre des classes
    "class_balance": {
        "expected_fraud_percentage": 0.17,
        "min_fraud_samples": 100,
        "target_fraud_percentage": 10.0
    }
}

# =============================================================================
# CONFIGURATION DU PREPROCESSING
# =============================================================================

PREPROCESSING_CONFIG = {
    # Paramètres de nettoyage
    "cleaning": {
        "remove_duplicates": True,
        "handle_missing": "median",
        "outlier_method": "iqr",
        "outlier_threshold": {
            "iqr_multiplier": 1.5,
            "zscore_threshold": 3,
            "percentile_cap": 99.5
        }
    },
    
    # Feature engineering
    "feature_engineering": {
        "create_temporal_features": True,
        "create_amount_features": True,
        "create_pca_features": True
    },
    
    # Normalisation
    "scaling": {
        "save_scalers": True,
        "scaler_file": "scalers.pkl"
    },
    
    # Split des données
    "data_split": {
        "test_size": 0.2,
        "validation_size": 0.2,
        "random_state": 42,
        "stratify": True,
        "shuffle": True
    }
}

# =============================================================================
# CONFIGURATION DE LA GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# =============================================================================

SYNTHETIC_DATA_CONFIG = {
    # Configuration générale
    "general": {
        "random_state": 42,
        "target_samples": 2000,
        "save_all_methods": True
    },
    
    # SMOTE et variantes
    "smote": {
        "enabled": True,
        "sampling_strategy": 0.1,
        "k_neighbors": 5
    },
    
    # SDV (Synthetic Data Vault)
    "sdv": {
        "enabled": True,
        "synthesizer": "GaussianCopulaSynthesizer",
        "num_samples": 2000
    },
    
    # Évaluation de la qualité
    "quality_evaluation": {
        "validation_split": 0.2,
        "quality_threshold": 0.8
    }
}

# =============================================================================
# CONFIGURATION AIRFLOW
# =============================================================================

AIRFLOW_CONFIG = {
    # Configuration du DAG
    "dag": {
        "dag_id": "fraud_detection_pipeline",
        "description": "Pipeline de détection de fraude bancaire complet",
        "schedule_interval": "@daily",
        "start_date": "2024-01-01",
        "catchup": False,
        "max_active_runs": 1,
        "tags": ["fraud-detection", "ml-pipeline", "banking"]
    },
    
    # Configuration des tâches
    "tasks": {
        "default_retries": 2,
        "retry_delay": "5m",
        "execution_timeout": "2h",
        "email_on_failure": True,
        "email_on_retry": False
    }
}

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_config_section(section_name: str) -> Dict[str, Any]:
    """Retourne une section spécifique de la configuration."""
    config_sections = {
        "PROJECT": PROJECT_CONFIG,
        "DATA": DATA_CONFIG,
        "PREPROCESSING": PREPROCESSING_CONFIG, 
        "SYNTHETIC": SYNTHETIC_DATA_CONFIG,
        "AIRFLOW": AIRFLOW_CONFIG,
        "PATHS": DATA_PATHS
    }
    
    if section_name.upper() not in config_sections:
        available_sections = list(config_sections.keys())
        raise KeyError(f"Section '{section_name}' non trouvée. Sections disponibles: {available_sections}")
    
    return config_sections[section_name.upper()]

def validate_config() -> Dict[str, bool]:
    """Valide la cohérence de la configuration."""
    results = {}
    
    # Valider que les chemins peuvent être créés
    try:
        for path_category, paths in DATA_PATHS.items():
            if path_category == "root":
                continue
            if isinstance(paths, dict):
                for path_name, path_obj in paths.items():
                    path_obj.mkdir(parents=True, exist_ok=True)
            else:
                paths.mkdir(parents=True, exist_ok=True)
        results["paths_exist"] = True
    except Exception:
        results["paths_exist"] = False
    
    # Valider les paramètres de données
    results["data_config_valid"] = (
        DATA_CONFIG["validation"]["min_rows"] > 0 and
        0 < DATA_CONFIG["validation"]["max_missing_percentage"] < 1
    )
    
    return results

def create_directory_structure() -> None:
    """Crée la structure de répertoires nécessaire."""
    for path_category, paths in DATA_PATHS.items():
        if path_category == "root":
            continue
        if isinstance(paths, dict):
            for path_name, path_obj in paths.items():
                path_obj.mkdir(parents=True, exist_ok=True)
                # Créer un fichier .gitkeep
                gitkeep = path_obj / ".gitkeep"
                if not gitkeep.exists():
                    gitkeep.touch()
        else:
            paths.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("🔧 Validation de la configuration simplifiée...")
    
    validation_results = validate_config()
    for check, result in validation_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")
    
    print("\n📁 Création de la structure de répertoires...")
    create_directory_structure()
    print("✅ Structure créée avec succès")
    
    print(f"\n📊 Projet: {PROJECT_CONFIG['name']} v{PROJECT_CONFIG['version']}")
    print(f"🎯 Objectif fraudes: {DATA_CONFIG['class_balance']['target_fraud_percentage']}%")