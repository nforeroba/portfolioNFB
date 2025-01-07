# mlflow_script.py
import mlflow
import mlflow.xgboost
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import os
import requests
import time
from train_model import (
    load_and_validate_data,
    get_model_params,
    validate_model,
    calculate_metrics,
    convert_to_serializable
)
from dialer_ratio_predictor import DialerRatioPredictor

MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"

def setup_mlflow(model_id):
    """Configurar experimento de MLflow para un modelo específico"""
    experiment_name = f"dialer_ratio_optimization_{model_id}"
    
    # Verificar si el servidor MLflow está corriendo
    try:
        response = requests.get(MLFLOW_TRACKING_URI)
        if response.status_code != 200:
            raise ConnectionError("MLflow server not responding")
    except:
        raise ConnectionError("Unable to connect to MLflow server")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def create_model_signature(predictor, features):
    """Crear firma del modelo con schema de entrada y salida"""
    input_schema = Schema([
        ColSpec("double", "agents_connected"),
        ColSpec("double", "agents_available"),
        ColSpec("long", "hour"),
        ColSpec("long", "day_of_week"),
        ColSpec("long", "part_of_day"),
        ColSpec("double", "avg_call_duration")
    ])
    
    output_schema = Schema([ColSpec("double", "ratio")])
    
    return ModelSignature(inputs=input_schema, outputs=output_schema)

def log_model_params(params):
    """Registrar parámetros del modelo en MLflow"""
    params_to_log = {
        "model_type": "global" if params["model_id"] == "global" else 
                     "portfolio" if params["model_id"].startswith("portfolio_") else 
                     "campaign",
        "model_id": params["model_id"],
        "target_contact_rate": convert_to_serializable(params["target_contact_rate"]),
        "target_availability_rate": convert_to_serializable(params["target_availability_rate"]),
        "max_abandon_rate": convert_to_serializable(params["max_abandon_rate"]),
        "contact_rate_weight": convert_to_serializable(params["weights"]["contact_rate"]),
        "availability_rate_weight": convert_to_serializable(params["weights"]["availability_rate"]),
        "abandon_rate_weight": convert_to_serializable(params["weights"]["abandon_rate"])
    }
    
    mlflow.log_params(params_to_log)

def log_model_metrics(metrics, validation_results):
    """Registrar métricas del modelo en MLflow"""
    metrics_to_log = {
        "mse": convert_to_serializable(metrics.get("mse", 0.0)),
        "mae": convert_to_serializable(metrics.get("mae", 0.0)),
        "r2": convert_to_serializable(metrics.get("r2", 0.0)),
    }
    
    # Agregar métricas de validación si existen
    if validation_results:
        if "predictions" in validation_results:
            pred_metrics = validation_results["predictions"]
            metrics_to_log.update({
                "mean_predicted_ratio": convert_to_serializable(pred_metrics.get("mean", 0.0)),
                "std_predicted_ratio": convert_to_serializable(pred_metrics.get("std", 0.0)),
                "min_predicted_ratio": convert_to_serializable(pred_metrics.get("min", 0.0)),
                "max_predicted_ratio": convert_to_serializable(pred_metrics.get("max", 0.0))
            })
        
        if "performance" in validation_results:
            perf_metrics = validation_results["performance"]
            metrics_to_log.update({
                "mean_performance_score": convert_to_serializable(perf_metrics.get("mean", 0.0)),
                "performance_stability": convert_to_serializable(perf_metrics.get("stability", 0.0))
            })
    
    # Agregar métricas de importancia de features
    if "feature_importance" in metrics:
        for feature, importance in metrics["feature_importance"].items():
            metrics_to_log[f"importance_{feature}"] = convert_to_serializable(importance)
    
    mlflow.log_metrics(metrics_to_log)

def log_model_to_mlflow(predictor, model_name, features):
    """Registrar modelo en MLflow con firma y ejemplo"""
    signature = create_model_signature(predictor, features)
    input_example = features.head(1)[predictor.feature_columns]
    
    mlflow.xgboost.log_model(
        xgb_model=predictor.model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example
    )
    
    # Guardar scaler como artefacto
    temp_scaler_path = "temp_scaler.joblib"
    joblib.dump(predictor.scaler, temp_scaler_path)
    mlflow.log_artifact(temp_scaler_path, "scaler")
    os.remove(temp_scaler_path)

def train_and_log_model():
    """Entrenar modelo y registrar en MLflow"""
    model_id = os.getenv('MODEL_ID')
    data_path = os.getenv('DATA_PATH')
    custom_params = os.getenv('MODEL_PARAMS')
    
    if not model_id or not data_path:
        raise ValueError("MODEL_ID and DATA_PATH environment variables required")
    
    setup_mlflow(model_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with mlflow.start_run(run_name=f"training_{model_id}_{timestamp}"):
        try:
            print(f"\nCargando datos para modelo: {model_id}")
            data = pd.read_csv(data_path)
            
            if data.empty:
                mlflow.log_param("status", "failed_no_data")
                print("No hay datos disponibles para entrenamiento")
                return None
            
            # Procesar y validar datos
            data, historical_metrics = load_and_validate_data(data)
            if data.empty:
                mlflow.log_param("status", "failed_invalid_data")
                print("Los datos no pasaron la validación")
                return None
            
            # Registrar información de datos
            mlflow.log_param("data_start_date", data["timestamp"].min().strftime("%Y-%m-%d"))
            mlflow.log_param("data_end_date", data["timestamp"].max().strftime("%Y-%m-%d"))
            mlflow.log_param("total_records", len(data))
            
            # Obtener parámetros del modelo
            if custom_params:
                params = json.loads(custom_params)
            else:
                params = get_model_params(data)
            params['model_id'] = model_id
            
            print("\nConfigurando y entrenando modelo...")
            predictor = DialerRatioPredictor(**params)
            metrics, processed_data = predictor.train(data)
            
            # Validar modelo
            print("\nValidando modelo...")
            last_day = data['timestamp'].max().date()
            test_data = data[data['timestamp'].dt.date == last_day]
            validation_results = validate_model(predictor, test_data)
            
            # Registrar parámetros, métricas y metadata
            log_model_params(params)
            log_model_metrics(metrics, validation_results)
            
            # Preparar features para el ejemplo y firma
            features = predictor.prepare_features(processed_data)
            
            # Registrar modelo
            model_name = f"dialer_ratio_predictor_{model_id}"
            log_model_to_mlflow(predictor, model_name, features)
            
            # Crear directorio local para caché
            local_model_dir = f"models/{model_id}"
            os.makedirs(local_model_dir, exist_ok=True)
            
            # Guardar modelo y scaler en caché local
            joblib.dump(predictor.model, f"{local_model_dir}/model_{mlflow.active_run().info.run_id}.joblib")
            joblib.dump(predictor.scaler, f"{local_model_dir}/scaler_{mlflow.active_run().info.run_id}.joblib")
            
            mlflow.log_param("status", "success")
            print(f"\nEntrenamiento completado exitosamente para modelo: {model_id}")
            
            return predictor
            
        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error_message", str(e))
            print(f"\nError en el proceso de entrenamiento: {str(e)}")
            raise

def main():
    """Función principal"""
    try:
        print("\nIniciando proceso de entrenamiento...")
        predictor = train_and_log_model()
        if predictor:
            print("\nProceso completado exitosamente")
        else:
            print("\nNo se pudo completar el entrenamiento")
    except Exception as e:
        print(f"\nError en el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()