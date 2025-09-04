"""
Utilitaires pour le Pipeline de Détection de Fraude Bancaire
==============================================================

Ce module contient des fonctions utilitaires réutilisables pour
le pipeline de machine learning de détection de fraude.

Modules principaux:
- Validation des données
- Génération de données synthétiques
- Logging et reporting
- Métriques de qualité

Auteur: Équipe Data Science - Néobanque Fluzz
Version: 1.0
Date: Août 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import logging
from datetime import datetime
import pickle
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure le système de logging pour le pipeline.
    
    Args:
        log_level (str, optional): Niveau de log ('DEBUG', 'INFO', 'WARNING', 'ERROR').
                                  Défaut: 'INFO'
        log_file (str, optional): Chemin vers le fichier de log. Si None, log sur console.
                                 Défaut: None
    
    Returns:
        logging.Logger: Logger configuré
        
    Examples:
        >>> logger = setup_logging("DEBUG", "logs/pipeline.log")
        >>> logger.info("Pipeline démarré")
    """
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Éviter les doublons de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Format des messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier si spécifié
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_dataset_quality(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1000,
    max_missing_pct: float = 0.05
) -> Dict[str, Union[bool, str, int, float]]:
    """
    Valide la qualité d'un dataset selon plusieurs critères.
    
    Effectue des vérifications complètes sur un DataFrame pour s'assurer
    qu'il respecte les standards de qualité requis pour l'entraînement
    des modèles de détection de fraude.
    
    Args:
        df (pd.DataFrame): Dataset à valider
        required_columns (List[str]): Liste des colonnes obligatoires
        min_rows (int, optional): Nombre minimum de lignes requis. Défaut: 1000
        max_missing_pct (float, optional): Pourcentage maximum de valeurs manquantes autorisé.
                                         Défaut: 0.05 (5%)
    
    Returns:
        Dict[str, Union[bool, str, int, float]]: Rapport de validation détaillé contenant:
            - 'is_valid': bool indiquant si le dataset est valide
            - 'message': str avec message d'erreur ou de succès
            - 'rows_count': int nombre de lignes
            - 'columns_count': int nombre de colonnes
            - 'missing_percentage': float pourcentage de valeurs manquantes
            - 'missing_columns': List[str] colonnes manquantes
            
    Raises:
        TypeError: Si df n'est pas un DataFrame pandas
        ValueError: Si required_columns est vide
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Time': [1, 2, 3, 4, 5],
        ...     'Amount': [10.0, 20.0, None, 40.0, 50.0],
        ...     'Class': [0, 1, 0, 1, 0]
        ... })
        >>> result = validate_dataset_quality(df, ['Time', 'Amount', 'Class'])
        >>> print(result['is_valid'])
        False  # Car 20% de valeurs manquantes > 5%
        
        >>> df_clean = df.dropna()
        >>> result = validate_dataset_quality(df_clean, ['Time', 'Amount', 'Class'])
        >>> print(result['is_valid'])
        True
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Le paramètre 'df' doit être un DataFrame pandas")
    
    if not required_columns:
        raise ValueError("La liste 'required_columns' ne peut pas être vide")
    
    # Initialiser le rapport
    report = {
        'is_valid': True,
        'message': 'Dataset valide',
        'rows_count': len(df),
        'columns_count': len(df.columns),
        'missing_percentage': 0.0,
        'missing_columns': []
    }
    
    # Vérifier si le DataFrame est vide
    if df.empty:
        report.update({
            'is_valid': False,
            'message': 'Dataset vide'
        })
        return report
    
    # Vérifier le nombre minimum de lignes
    if len(df) < min_rows:
        report.update({
            'is_valid': False,
            'message': f'Trop peu de lignes: {len(df)} < {min_rows}'
        })
        return report
    
    # Vérifier les colonnes obligatoires
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        report.update({
            'is_valid': False,
            'message': f'Colonnes manquantes: {missing_columns}',
            'missing_columns': missing_columns
        })
        return report
    
    # Calculer le pourcentage de valeurs manquantes
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    report['missing_percentage'] = missing_percentage
    
    if missing_percentage > max_missing_pct * 100:
        report.update({
            'is_valid': False,
            'message': f'Trop de valeurs manquantes: {missing_percentage:.2f}% > {max_missing_pct*100}%'
        })
        return report
    
    return report


def calculate_class_balance_metrics(y: pd.Series) -> Dict[str, Union[int, float]]:
    """
    Calcule les métriques de déséquilibre des classes.
    
    Analyse la distribution des classes dans la variable cible et calcule
    différentes métriques utiles pour évaluer le déséquilibre.
    
    Args:
        y (pd.Series): Variable cible contenant les labels de classe
        
    Returns:
        Dict[str, Union[int, float]]: Métriques de déséquilibre contenant:
            - 'total_samples': nombre total d'échantillons
            - 'class_counts': dictionnaire {classe: nombre}
            - 'class_percentages': dictionnaire {classe: pourcentage}
            - 'imbalance_ratio': ratio majoritaire/minoritaire
            - 'minority_class': label de la classe minoritaire
            - 'majority_class': label de la classe majoritaire
            
    Raises:
        ValueError: Si y est vide ou contient moins de 2 classes
        TypeError: Si y n'est pas une Series pandas
        
    Examples:
        >>> import pandas as pd
        >>> y = pd.Series([0, 0, 0, 0, 1, 1])  # 4 normales, 2 fraudes
        >>> metrics = calculate_class_balance_metrics(y)
        >>> print(metrics['imbalance_ratio'])
        2.0
        >>> print(metrics['minority_class'])
        1
    """
    if not isinstance(y, pd.Series):
        raise TypeError("Le paramètre 'y' doit être une Series pandas")
    
    if y.empty:
        raise ValueError("La Series 'y' ne peut pas être vide")
    
    # Compter les classes
    class_counts = y.value_counts().to_dict()
    total_samples = len(y)
    
    if len(class_counts) < 2:
        raise ValueError("Il faut au moins 2 classes différentes")
    
    # Calculer les pourcentages
    class_percentages = {
        cls: (count / total_samples) * 100 
        for cls, count in class_counts.items()
    }
    
    # Identifier les classes majoritaire et minoritaire
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)
    
    # Calculer le ratio de déséquilibre
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    
    return {
        'total_samples': total_samples,
        'class_counts': class_counts,
        'class_percentages': class_percentages,
        'imbalance_ratio': imbalance_ratio,
        'minority_class': minority_class,
        'majority_class': majority_class
    }


def save_model_artifacts(
    artifacts: Dict[str, any],
    output_dir: Union[str, Path],
    prefix: str = "model"
) -> Dict[str, str]:
    """
    Sauvegarde les artefacts d'un modèle (modèle, scalers, métadonnées).
    
    Organise et sauvegarde tous les éléments nécessaires pour
    déployer et utiliser un modèle en production.
    
    Args:
        artifacts (Dict[str, any]): Dictionnaire contenant les artefacts à sauvegarder
                                   Clés attendues: 'model', 'scaler', 'metadata'
        output_dir (Union[str, Path]): Répertoire de destination
        prefix (str, optional): Préfixe pour les noms de fichiers. Défaut: 'model'
        
    Returns:
        Dict[str, str]: Dictionnaire des chemins de fichiers sauvegardés
        
    Raises:
        OSError: Si impossible de créer le répertoire ou sauvegarder
        ValueError: Si artifacts est vide ou mal formaté
        
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.preprocessing import StandardScaler
        >>> 
        >>> model = RandomForestClassifier()
        >>> scaler = StandardScaler()
        >>> metadata = {'version': '1.0', 'features': ['V1', 'V2']}
        >>> 
        >>> artifacts = {
        ...     'model': model,
        ...     'scaler': scaler,
        ...     'metadata': metadata
        ... }
        >>> 
        >>> paths = save_model_artifacts(artifacts, 'models/', 'fraud_detector')
        >>> print(paths['model'])
        'models/fraud_detector_model.pkl'
    """
    if not artifacts:
        raise ValueError("Le dictionnaire 'artifacts' ne peut pas être vide")
    
    # Créer le répertoire de sortie
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        for artifact_type, artifact_data in artifacts.items():
            filename = f"{prefix}_{artifact_type}_{timestamp}.pkl"
            file_path = output_path / filename
            
            with open(file_path, 'wb') as f:
                pickle.dump(artifact_data, f)
            
            saved_paths[artifact_type] = str(file_path)
            
        # Sauvegarder aussi un index des fichiers
        index_file = output_path / f"{prefix}_index_{timestamp}.json"
        import json
        with open(index_file, 'w') as f:
            json.dump(saved_paths, f, indent=2)
        
        saved_paths['index'] = str(index_file)
        
        return saved_paths
        
    except Exception as e:
        raise OSError(f"Erreur lors de la sauvegarde: {str(e)}")


def load_model_artifacts(index_file: Union[str, Path]) -> Dict[str, any]:
    """
    Charge les artefacts d'un modèle depuis un fichier d'index.
    
    Args:
        index_file (Union[str, Path]): Chemin vers le fichier d'index JSON
        
    Returns:
        Dict[str, any]: Dictionnaire contenant tous les artefacts chargés
        
    Raises:
        FileNotFoundError: Si le fichier d'index n'existe pas
        OSError: Si impossible de charger les artefacts
        
    Examples:
        >>> artifacts = load_model_artifacts('models/fraud_detector_index_20250801_143000.json')
        >>> model = artifacts['model']
        >>> scaler = artifacts['scaler']
    """
    index_path = Path(index_file)
    
    if not index_path.exists():
        raise FileNotFoundError(f"Fichier d'index introuvable: {index_file}")
    
    # Charger l'index
    import json
    with open(index_path, 'r') as f:
        file_paths = json.load(f)
    
    artifacts = {}
    
    try:
        for artifact_type, file_path in file_paths.items():
            if artifact_type == 'index':
                continue
                
            with open(file_path, 'rb') as f:
                artifacts[artifact_type] = pickle.load(f)
                
        return artifacts
        
    except Exception as e:
        raise OSError(f"Erreur lors du chargement: {str(e)}")


def generate_pipeline_report(
    pipeline_results: Dict[str, any],
    output_file: Optional[Union[str, Path]] = None
) -> str:
    """
    Génère un rapport détaillé d'exécution du pipeline.
    
    Crée un rapport formaté contenant toutes les métriques et résultats
    d'une exécution complète du pipeline de détection de fraude.
    
    Args:
        pipeline_results (Dict[str, any]): Résultats du pipeline contenant
                                          les métriques de chaque étape
        output_file (Optional[Union[str, Path]], optional): Fichier de sortie.
                                                           Si None, retourne le contenu.
                                                           Défaut: None
    
    Returns:
        str: Contenu du rapport formaté
        
    Examples:
        >>> results = {
        ...     'data_validation': {'is_valid': True, 'rows': 50000},
        ...     'preprocessing': {'features_created': 8},
        ...     'synthetic_generation': {'samples_generated': 5000}
        ... }
        >>> report = generate_pipeline_report(results, 'reports/pipeline_report.md')
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_lines = [
        "# Rapport d'Exécution du Pipeline de Détection de Fraude",
        f"**Date d'exécution:** {timestamp}",
        "",
        "## Résumé Exécutif",
        ""
    ]
    
    # Ajouter les résultats de chaque étape
    for step_name, step_results in pipeline_results.items():
        report_lines.extend([
            f"### {step_name.replace('_', ' ').title()}",
            ""
        ])
        
        if isinstance(step_results, dict):
            for key, value in step_results.items():
                report_lines.append(f"- **{key}:** {value}")
        else:
            report_lines.append(f"- **Résultat:** {step_results}")
        
        report_lines.append("")
    
    # Ajouter les recommandations
    report_lines.extend([
        "## Recommandations",
        "",
        "- Surveiller les métriques de performance en continu",
        "- Valider la qualité des données synthétiques générées",
        "- Planifier le réentraînement selon la dérive observée",
        "",
        "---",
        f"*Rapport généré automatiquement le {timestamp}*"
    ])
    
    report_content = "\n".join(report_lines)
    
    # Sauvegarder si fichier spécifié
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return report_content


# Configuration par défaut des chemins
DEFAULT_PATHS = {
    'raw_data': 'data/raw/',
    'processed_data': 'data/processed/',
    'synthetic_data': 'data/synthetic/',
    'models': 'models/',
    'logs': 'logs/',
    'reports': 'reports/'
}


def ensure_directory_structure(base_path: Union[str, Path] = ".") -> None:
    """
    Crée la structure de répertoires nécessaire au projet.
    
    Args:
        base_path (Union[str, Path], optional): Répertoire racine du projet.
                                              Défaut: répertoire courant
    
    Examples:
        >>> ensure_directory_structure("/path/to/project")
    """
    base = Path(base_path)
    
    for path_name, relative_path in DEFAULT_PATHS.items():
        full_path = base / relative_path
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Créer un fichier .gitkeep pour maintenir la structure dans Git
        gitkeep_file = full_path / '.gitkeep'
        if not gitkeep_file.exists():
            gitkeep_file.touch()


if __name__ == "__main__":
    # Tests basiques des fonctions
    print("🧪 Tests des fonctions utilitaires...")
    
    # Test de validation de dataset
    test_df = pd.DataFrame({
        'Time': [1, 2, 3, 4, 5],
        'Amount': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Class': [0, 1, 0, 1, 0]
    })
    
    result = validate_dataset_quality(test_df, ['Time', 'Amount', 'Class'])
    print(f"✅ Validation dataset: {result['is_valid']}")
    
    # Test de métriques de déséquilibre
    metrics = calculate_class_balance_metrics(test_df['Class'])
    print(f"✅ Ratio de déséquilibre: {metrics['imbalance_ratio']:.1f}")
    
    # Test de structure de répertoires
    ensure_directory_structure()
    print("✅ Structure de répertoires créée")
    
    print("🎉 Tous les tests sont passés!")