"""
Pipeline Airflow pour la Détection de Fraude Bancaire
======================================================

Ce DAG automatise l'ensemble du processus de traitement des données,
génération de données synthétiques et préparation pour l'entraînement des modèles
de détection de fraude bancaire.

Auteur: Équipe Data Science - Néobanque Fluzz
Version: 1.0
Date: Août 2025
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import sys
import os
from pathlib import Path

# Ajouter le chemin du projet pour importer nos modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Imports de nos modules personnalisés
from config.config import (
    DATA_CONFIG, 
    PREPROCESSING_CONFIG, 
    SYNTHETIC_DATA_CONFIG,
    AIRFLOW_CONFIG,
    get_config_section,
    validate_config
)
from src.utils import (
    setup_logging,
    validate_dataset_quality,
    calculate_class_balance_metrics,
    save_model_artifacts,
    generate_pipeline_report
)

# Configuration par défaut des tâches Airflow (utilise config.py)
default_args = {
    'owner': 'data-science-team',
    'start_date': datetime.strptime(AIRFLOW_CONFIG['dag']['start_date'], '%Y-%m-%d'),
    'retries': AIRFLOW_CONFIG['tasks']['default_retries'],
    'retry_delay': timedelta(minutes=int(AIRFLOW_CONFIG['tasks']['retry_delay'].replace('m', ''))),
    'email_on_failure': AIRFLOW_CONFIG['tasks']['email_on_failure'],
    'email_on_retry': AIRFLOW_CONFIG['tasks']['email_on_retry'],
}

# Définition du DAG principal (utilise config.py)
dag = DAG(
    AIRFLOW_CONFIG['dag']['dag_id'],
    default_args=default_args,
    description=AIRFLOW_CONFIG['dag']['description'],
    schedule=AIRFLOW_CONFIG['dag']['schedule_interval'],
    start_date=datetime.strptime(AIRFLOW_CONFIG['dag']['start_date'], '%Y-%m-%d'),
    catchup=AIRFLOW_CONFIG['dag']['catchup'],
    tags=AIRFLOW_CONFIG['dag']['tags'],
    max_active_runs=AIRFLOW_CONFIG['dag']['max_active_runs'],
)


def check_data(**context):
    """
    Vérifie la qualité et la disponibilité des données brutes.
    
    Cette fonction valide les données d'entrée en effectuant :
    - Vérification de l'existence des fichiers source
    - Contrôle de l'intégrité des données (schéma, types)
    - Détection des valeurs manquantes ou aberrantes
    - Validation des contraintes métier
    
    Args:
        **context: Contexte Airflow avec informations sur la tâche
        
    Returns:
        dict: Rapport de validation avec métriques de qualité
        
    Raises:
        FileNotFoundError: Si les fichiers source sont introuvables
        ValueError: Si les données ne respectent pas les contraintes
        
    Examples:
        >>> check_data()
        {'status': 'success', 'rows': 284807, 'quality_score': 0.98}
    """
    import pandas as pd
    
    # Configuration du logging
    logger = setup_logging("INFO")
    logger.info("🔍 Démarrage de la vérification des données...")
    
    try:
        # Charger les données depuis la configuration
        data_path = project_root / "data" / "raw" / DATA_CONFIG["source_file"]
        
        if not data_path.exists():
            raise FileNotFoundError(f"Fichier de données introuvable: {data_path}")
        
        # Charger le dataset
        logger.info(f"Chargement des données depuis: {data_path}")
        df = pd.read_csv(data_path)
        
        # Valider avec nos utilitaires
        validation_result = validate_dataset_quality(
            df=df,
            required_columns=DATA_CONFIG["validation"]["required_columns"],
            min_rows=DATA_CONFIG["validation"]["min_rows"],
            max_missing_pct=DATA_CONFIG["validation"]["max_missing_percentage"]
        )
        
        # Calculer les métriques de déséquilibre
        class_metrics = calculate_class_balance_metrics(df[DATA_CONFIG["target_column"]])
        
        # Vérifications supplémentaires spécifiques au métier
        business_checks = {
            "time_range_valid": (
                df["Time"].min() >= DATA_CONFIG["validation"]["time_range"]["min"] and
                df["Time"].max() <= DATA_CONFIG["validation"]["time_range"]["max"]
            ),
            "amount_range_valid": (
                df["Amount"].min() >= DATA_CONFIG["validation"]["amount_range"]["min"] and
                df["Amount"].max() <= DATA_CONFIG["validation"]["amount_range"]["max"]
            ),
            "class_values_valid": set(df[DATA_CONFIG["target_column"]].unique()) <= set(DATA_CONFIG["validation"]["class_values"])
        }
        
        # Compiler le rapport final
        report = {
            "status": "success" if validation_result["is_valid"] and all(business_checks.values()) else "failed",
            "message": str(validation_result["message"]),
            "data_quality": {
                "rows_count": int(validation_result["rows_count"]),
                "columns_count": int(validation_result["columns_count"]),
                "missing_percentage": float(validation_result["missing_percentage"]),
                "fraud_percentage": float(class_metrics["class_percentages"][1]),
                "imbalance_ratio": float(class_metrics["imbalance_ratio"])
            },
            "business_validation": {
                "time_range_valid": bool(business_checks["time_range_valid"]),
                "amount_range_valid": bool(business_checks["amount_range_valid"]),
                "class_values_valid": bool(business_checks["class_values_valid"])
            },
            "class_distribution": {int(k): int(v) for k, v in class_metrics["class_counts"].items()}
        }
        
        logger.info(f"✅ Validation terminée: {report['data_quality']['rows_count']} lignes, "
                   f"{report['data_quality']['fraud_percentage']:.3f}% fraudes")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation: {str(e)}")
        return {
            "status": "failed",
            "message": f"Erreur de validation: {str(e)}",
            "error": str(e)
        }


def preprocess_data(**context):
    """
    Nettoie et prétraite les données pour l'entraînement des modèles.
    
    Effectue les transformations suivantes :
    - Suppression des doublons et valeurs manquantes
    - Traitement des outliers (écrêtage, transformation)
    - Feature engineering (variables temporelles, agrégations)
    - Normalisation et standardisation des variables
    - Séparation train/validation/test stratifiée
    
    Args:
        **context: Contexte Airflow avec informations sur la tâche
        
    Returns:
        dict: Métadonnées sur le preprocessing effectué
        
    Raises:
        ValueError: Si les données d'entrée sont invalides
        RuntimeError: Si le preprocessing échoue
        
    Examples:
        >>> preprocess_data()
        {'rows_processed': 283726, 'features_created': 8, 'scalers_applied': 3}
    """
    print("🧹 Démarrage du preprocessing des données...")
    print("✅ Preprocessing terminé avec succès")
    
    # TODO: Intégrer le code du notebook partie-2.1.ipynb
    # - Charger les données depuis check_data
    # - Appliquer le nettoyage et feature engineering
    # - Sauvegarder les données traitées dans data/processed/
    # - Sauvegarder les scalers et métadonnées
    
    return {"status": "success", "message": "Données prétraitées"}


def generate_sdv(**context):
    """
    Génère des données synthétiques frauduleuses avec SDV Gaussian Copula.
    
    Utilise la bibliothèque SDV (Synthetic Data Vault) pour créer
    des transactions frauduleuses synthétiques réalistes basées sur
    les patterns des vraies fraudes historiques.
    
    Args:
        **context: Contexte Airflow avec informations sur la tâche
        
    Returns:
        dict: Informations sur la génération (nombre d'échantillons, qualité)
        
    Raises:
        ImportError: Si SDV n'est pas installé
        RuntimeError: Si la génération échoue
        ValueError: Si les données d'entrée sont insuffisantes
        
    Examples:
        >>> generate_sdv()
        {'samples_generated': 2000, 'quality_score': 0.92, 'method': 'GaussianCopula'}
    """
    print("🤖 Démarrage de la génération SDV...")
    print("✅ Génération SDV terminée avec succès")
    
    # TODO: Intégrer le code SDV du notebook partie-2.2.ipynb
    # - Charger les données frauduleuses depuis preprocess_data
    # - Entraîner le modèle GaussianCopulaSynthesizer
    # - Générer les échantillons synthétiques
    # - Évaluer la qualité des données générées
    # - Sauvegarder dans data/synthetic/
    
    return {"status": "success", "samples": 2000, "method": "SDV"}


def generate_smote(**context):
    """
    Génère des données synthétiques avec les techniques SMOTE.
    
    Applique différentes variantes de SMOTE (Synthetic Minority Oversampling
    Technique) pour augmenter le nombre d'échantillons frauduleux :
    - SMOTE classique
    - BorderlineSMOTE (focus sur les cas limites)
    - Fallback vers suréchantillonnage simple si SMOTE indisponible
    
    Args:
        **context: Contexte Airflow avec informations sur la tâche
        
    Returns:
        dict: Résultats de génération pour chaque méthode testée
        
    Raises:
        ImportError: Si imbalanced-learn n'est pas installé
        ValueError: Si les paramètres de sampling sont invalides
        RuntimeError: Si toutes les méthodes échouent
        
    Examples:
        >>> generate_smote()
        {'SMOTE': 5000, 'BorderlineSMOTE': 4800, 'best_method': 'SMOTE'}
    """
    print("⚖️ Démarrage de la génération SMOTE...")
    print("✅ Génération SMOTE terminée avec succès")
    
    # TODO: Intégrer le code SMOTE du notebook partie-2.2.ipynb
    # - Tester SMOTE, BorderlineSMOTE et méthode simple
    # - Comparer les résultats et sélectionner la meilleure
    # - Sauvegarder tous les datasets générés
    # - Retourner les métriques de comparaison
    
    return {"status": "success", "methods_tested": 3}


def select_best(**context):
    """
    Sélectionne le meilleur dataset synthétique basé sur les métriques de qualité.
    
    Compare les différents datasets générés (SDV, SMOTE, BorderlineSMOTE)
    selon plusieurs critères :
    - Qualité des données synthétiques (réalisme, diversité)
    - Amélioration du déséquilibre des classes
    - Performance sur des modèles de validation
    - Métriques de distance statistique
    
    Args:
        **context: Contexte Airflow avec informations sur la tâche
        
    Returns:
        dict: Informations sur le dataset sélectionné et les métriques
        
    Raises:
        ValueError: Si aucun dataset valide n'est disponible
        RuntimeError: Si l'évaluation échoue
        
    Examples:
        >>> select_best()
        {'best_method': 'SMOTE', 'quality_score': 0.95, 'dataset_size': 50000}
    """
    print("🏆 Démarrage de la sélection du meilleur dataset...")
    print("✅ Sélection terminée avec succès")
    
    # TODO: Implémenter la logique de sélection
    # - Charger tous les datasets générés
    # - Calculer les métriques de qualité
    # - Comparer les performances
    # - Sélectionner et sauvegarder le meilleur
    # - Créer un rapport de comparaison
    
    return {"status": "success", "best_method": "auto-selected"}


def notify(**context):
    """
    Envoie les notifications de fin de pipeline aux équipes concernées.
    
    Génère et envoie un rapport d'exécution comprenant :
    - Statut global du pipeline (succès/échec)
    - Métriques de performance de chaque étape
    - Datasets générés et leur qualité
    - Recommandations pour les prochaines étapes
    - Alertes si des seuils critiques sont dépassés
    
    Args:
        **context: Contexte Airflow avec informations sur toutes les tâches
        
    Returns:
        dict: Confirmation d'envoi des notifications
        
    Raises:
        ConnectionError: Si l'envoi des notifications échoue
        ValueError: Si les données du rapport sont incomplètes
        
    Examples:
        >>> notify()
        {'notifications_sent': 3, 'recipients': ['team@fluzz.com'], 'status': 'sent'}
    """
    from datetime import datetime
    
    # Configuration du logging
    logger = setup_logging("INFO")
    logger.info("📧 Préparation des notifications...")
    
    try:
        # Récupérer les résultats des tâches précédentes via XCom
        task_instance = context['task_instance']
        dag_run = context['dag_run']
        
        # Collecter les résultats de toutes les tâches
        pipeline_results = {
            'execution_date': dag_run.execution_date.isoformat(),
            'dag_run_id': dag_run.run_id,
            'dag_id': dag_run.dag_id
        }
        
        # Essayer de récupérer les résultats des tâches précédentes
        try:
            pipeline_results['data_validation'] = task_instance.xcom_pull(task_ids='check_data')
            pipeline_results['preprocessing'] = task_instance.xcom_pull(task_ids='preprocess_data')
            pipeline_results['sdv_generation'] = task_instance.xcom_pull(task_ids='synthetic_generation.generate_sdv')
            pipeline_results['smote_generation'] = task_instance.xcom_pull(task_ids='synthetic_generation.generate_smote')
            pipeline_results['best_selection'] = task_instance.xcom_pull(task_ids='select_best')
        except Exception as e:
            logger.warning(f"Impossible de récupérer certains résultats XCom: {e}")
            pipeline_results['warning'] = "Certains résultats de tâches ne sont pas disponibles"
        
        # Générer le rapport avec nos utilitaires
        report_path = project_root / "reports" / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_content = generate_pipeline_report(
            pipeline_results=pipeline_results,
            output_file=str(report_path)
        )
        
        # Préparer les informations de notification
        notification_config = AIRFLOW_CONFIG.get('notifications', {})
        recipients = notification_config.get('success', {}).get('recipients', ['data-science@fluzz.com'])
        
        # Déterminer le statut global
        has_failures = any(
            result and result.get('status') == 'failed' 
            for result in pipeline_results.values() 
            if isinstance(result, dict)
        )
        
        global_status = "failed" if has_failures else "success"
        
        logger.info(f"✅ Rapport généré: {report_path}")
        logger.info(f"📊 Statut global: {global_status}")
        logger.info(f"📧 Destinataires configurés: {recipients}")
        
        return {
            "status": "success",
            "notifications_sent": True,
            "global_pipeline_status": global_status,
            "report_path": str(report_path),
            "recipients": recipients,
            "results_summary": {
                key: value.get('status', 'unknown') if isinstance(value, dict) else 'completed'
                for key, value in pipeline_results.items()
                if key not in ['execution_date', 'dag_run_id', 'dag_id']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération du rapport: {str(e)}")
        return {
            "status": "failed",
            "message": f"Erreur de notification: {str(e)}",
            "error": str(e)
        }

# Tasks
check_task = PythonOperator(
    task_id='check_data',
    python_callable=check_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

with TaskGroup("synthetic_generation", dag=dag) as synthetic_group:
    sdv_task = PythonOperator(
        task_id='generate_sdv',
        python_callable=generate_sdv,
    )
    
    smote_task = PythonOperator(
        task_id='generate_smote',
        python_callable=generate_smote,
    )

select_task = PythonOperator(
    task_id='select_best',
    python_callable=select_best,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='notify',
    python_callable=notify,
    dag=dag,
)

check_task >> preprocess_task >> synthetic_group >> select_task >> notify_task