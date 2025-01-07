# train_model.py
import pandas as pd
import numpy as np
import joblib
from dialer_ratio_predictor import DialerRatioPredictor
import os
from datetime import datetime, timedelta
import json
import logging
from sklearn.metrics import r2_score
import warnings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignorar warnings específicos
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def convert_to_serializable(obj):
    """Convertir tipos numpy y pandas a tipos serializables en JSON"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    if hasattr(obj, 'dtype'):
        return obj.item()
    return obj

def load_and_validate_data(data):
    """Cargar y validar datos"""
    required_columns = [
        'timestamp', 'agents_available', 'agents_connected', 
        'dials', 'contacts', 'abandonments', 'call_duration'
    ]
    
    try:
        if data.empty:
            logger.warning("DataFrame vacío")
            return pd.DataFrame(columns=required_columns), {}
        
        # Validar columnas
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
        
        # Procesar timestamp
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Remover filas con timestamp inválido
        invalid_timestamps = data['timestamp'].isna()
        if invalid_timestamps.any():
            logger.warning(f"Se removieron {invalid_timestamps.sum()} filas con timestamps inválidos")
            data = data.dropna(subset=['timestamp'])
        
        # Filtrar por horario de operación
        is_weekday = data['timestamp'].dt.dayofweek.isin([0,1,2,3,4])
        is_saturday = data['timestamp'].dt.dayofweek == 5
        
        weekday_hours = (data['timestamp'].dt.hour >= 7) & (data['timestamp'].dt.hour < 19)
        saturday_hours = (data['timestamp'].dt.hour >= 8) & (data['timestamp'].dt.hour < 15)
        
        data = data[
            ((is_weekday & weekday_hours) | (is_saturday & saturday_hours))
        ]
        
        # Limpiar columnas numéricas
        numeric_columns = [col for col in required_columns if col != 'timestamp']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].clip(lower=0)
        
        # Eliminar duplicados y ordenar
        data = data.fillna(0)
        data = data.drop_duplicates(subset=['timestamp'], keep='first')
        data = data.sort_values('timestamp')
        
        # Calcular métricas
        metrics = calculate_metrics(data)
        
        logger.info(f"Procesados {len(data)} registros desde {data['timestamp'].min()} hasta {data['timestamp'].max()}")
        logger.info("Métricas calculadas exitosamente")
        
        return data, metrics
        
    except Exception as e:
        logger.error(f"Error procesando datos: {str(e)}")
        raise

def calculate_metrics(data):
    """Calcular métricas operativas clave"""
    metrics = {'rates': {}, 'volume': {}, 'time_gaps': {}}
    
    try:
        # Calcular tasas
        mask_dials = data['dials'] > 0
        mask_connected = data['agents_connected'] > 0
        mask_contacts = data['contacts'] > 0
        
        # Tasa de contacto
        contact_rate = data.loc[mask_dials, 'contacts'] / data.loc[mask_dials, 'dials']
        metrics['rates']['contact_rate'] = {
            'mean': float(contact_rate.mean()),
            'median': float(contact_rate.median()),
            'std': float(contact_rate.std()),
            'min': float(contact_rate.min()),
            'max': float(contact_rate.max())
        }
        
        # Tasa de abandono
        abandon_rate = data.loc[mask_dials, 'abandonments'] / data.loc[mask_dials, 'dials']
        metrics['rates']['abandon_rate'] = {
            'mean': float(abandon_rate.mean()),
            'median': float(abandon_rate.median()),
            'std': float(abandon_rate.std()),
            'min': float(abandon_rate.min()),
            'max': float(abandon_rate.max())
        }
        
        # Tasa de disponibilidad
        availability_rate = data.loc[mask_connected, 'agents_available'] / data.loc[mask_connected, 'agents_connected']
        metrics['rates']['availability_rate'] = {
            'mean': float(availability_rate.mean()),
            'median': float(availability_rate.median()),
            'std': float(availability_rate.std()),
            'min': float(availability_rate.min()),
            'max': float(availability_rate.max())
        }
        
        # Duración promedio de llamadas
        avg_duration = data.loc[mask_contacts, 'call_duration']
        metrics['rates']['avg_call_duration'] = {
            'mean': float(avg_duration.mean()),
            'median': float(avg_duration.median()),
            'std': float(avg_duration.std()),
            'min': float(avg_duration.min()),
            'max': float(avg_duration.max())
        }
        
        # Métricas de volumen
        volume_metrics = ['agents_available', 'agents_connected', 'dials', 'contacts']
        for metric in volume_metrics:
            metrics['volume'][metric] = {
                'mean': float(data[metric].mean()),
                'median': float(data[metric].median()),
                'std': float(data[metric].std()),
                'min': float(data[metric].min()),
                'max': float(data[metric].max())
            }
        
        # Análisis de gaps temporales
        time_diffs = data['timestamp'].diff()
        metrics['time_gaps'] = {
            'max_gap_minutes': float(time_diffs.max().total_seconds() / 60),
            'mean_gap_minutes': float(time_diffs.mean().total_seconds() / 60),
            'total_gaps': int(time_diffs.gt(pd.Timedelta(minutes=5)).sum())
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculando métricas: {str(e)}")
        raise

def get_model_params(data, custom_params=None):
    """Determinar parámetros del modelo basados en métricas históricas o parámetros personalizados"""
    try:
        if custom_params:
            logger.info("Usando parámetros personalizados para el modelo")
            return custom_params
            
        logger.info("Calculando parámetros del modelo basados en métricas históricas")
        
        # Calcular métricas base
        contact_rates = data.loc[data['dials'] > 0, 'contacts'] / data.loc[data['dials'] > 0, 'dials']
        abandon_rates = data.loc[data['dials'] > 0, 'abandonments'] / data.loc[data['dials'] > 0, 'dials']
        availability_rates = data.loc[data['agents_connected'] > 0, 'agents_available'] / data.loc[data['agents_connected'] > 0, 'agents_connected']
        
        # Parámetros por defecto
        params = {
            'target_contact_rate': float(np.clip(contact_rates.mean(), 0.0, 1.0)),
            'target_availability_rate': float(np.clip(availability_rates.mean(), 0.0, 1.0)),
            'max_abandon_rate': float(np.clip(abandon_rates.mean() * 1.5, 0.0, 0.10)),
            'weights': {
                'contact_rate': 0.4,
                'availability_rate': 0.3,
                'abandon_rate': 0.3
            }
        }
        
        logger.info(f"Parámetros calculados: {json.dumps(params, indent=2)}")
        return params
        
    except Exception as e:
        logger.error(f"Error calculando parámetros del modelo: {str(e)}")
        raise

def validate_model(predictor, test_data):
    """Validar modelo con métricas específicas"""
    if test_data.empty:
        logger.warning("No hay datos de prueba para validación")
        return None
        
    try:
        # Obtener predicciones y performance
        features = predictor.prepare_features(test_data)
        predictions = predictor.predict(features)
        performance = predictor.calculate_performance_score(test_data)
        
        # Calcular ratios actuales
        mask = test_data['agents_available'] > 0
        actual_ratios = np.where(mask,
                               test_data['dials'] / test_data['agents_available'],
                               1)  # Valor por defecto de 1 cuando no hay agentes disponibles
        actual_ratios = np.clip(actual_ratios, 1, None)  # Clip con mínimo de 1
        
        # Preparar resultados
        results = {
            'predictions': {
                'mean': float(predictions.mean()),
                'std': float(predictions.std()),
                'min': float(predictions.min()),
                'max': float(predictions.max()),
                'percentile_95': float(np.percentile(predictions, 95))
            },
            'performance': {
                'mean': float(performance.mean()),
                'std': float(performance.std()),
                'stability': float(1 - performance.std() / (performance.mean() + 1e-10))
            }
        }
        
        if mask.any():
            results['comparison'] = {
                'actual_mean': float(actual_ratios.mean()),
                'predicted_mean': float(predictions[mask].mean()),
                'mae': float(np.abs(actual_ratios - predictions).mean()),
                'mse': float(((actual_ratios - predictions) ** 2).mean()),
                'rmse': float(np.sqrt(((actual_ratios - predictions) ** 2).mean())),
                'r2': float(r2_score(actual_ratios[mask], predictions[mask])),
                'correlation': float(np.corrcoef(actual_ratios[mask], predictions[mask])[0,1])
            }
        
        # Métricas de XGBoost
        if hasattr(predictor.model, 'best_score_'):
            results['xgboost_metrics'] = {
                'best_score': float(predictor.model.best_score_),
                'best_iteration': int(predictor.model.best_iteration_) if hasattr(predictor.model, 'best_iteration_') else None,
                'feature_importance': {k: float(v) for k, v in predictor.feature_importance.items()}
            }
        
        logger.info("Validación del modelo completada exitosamente")
        return results
        
    except Exception as e:
        logger.error(f"Error en validación del modelo: {str(e)}")
        raise

def save_model_artifacts(predictor, metadata, output_dir, timestamp):
    """Guardar artefactos del modelo"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Definir rutas
        paths = {
            'model': os.path.join(output_dir, f'model_{timestamp}.joblib'),
            'metadata': os.path.join(output_dir, f'metadata_{timestamp}.json')
        }
        
        # Guardar modelo
        predictor.save(paths['model'])
        
        # Guardar metadata
        with open(paths['metadata'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Artefactos guardados en: {output_dir}")
        return paths
        
    except Exception as e:
        logger.error(f"Error guardando artefactos: {str(e)}")
        raise

def main():
    """Flujo principal de entrenamiento"""
    try:
        # Configuración
        data_path = 'data/training_data.csv'
        output_dir = 'models'
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar parámetros personalizados si existen
        custom_params_path = os.getenv('CUSTOM_PARAMS_PATH')
        custom_params = None
        if custom_params_path and os.path.exists(custom_params_path):
            with open(custom_params_path, 'r') as f:
                custom_params = json.load(f)
        
        # Cargar y validar datos
        logger.info("Iniciando carga de datos...")
        data = pd.read_csv(data_path)
        data, metrics = load_and_validate_data(data)
        
        if data.empty:
            logger.warning("Entrenamiento cancelado: No hay datos válidos")
            return
        
        # Obtener parámetros del modelo
        logger.info("Calculando parámetros del modelo...")
        params = get_model_params(data, custom_params)
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        predictor = DialerRatioPredictor(**params)
        training_metrics, _ = predictor.train(data)
        
        # Validar modelo
        logger.info("Validando modelo...")
        last_day = data['timestamp'].max().date()
        test_data = data[data['timestamp'].dt.date == last_day]
        validation_results = validate_model(predictor, test_data)
        
        # Preparar metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            'timestamp': timestamp,
            'training_metrics': convert_to_serializable(training_metrics),
            'validation_results': convert_to_serializable(validation_results),
            'parameters': convert_to_serializable(params),
            'historical_metrics': convert_to_serializable(metrics),
            'data_info': {
                'total_records': len(data),
                'date_range': {
                    'start': data['timestamp'].min().isoformat(),
                    'end': data['timestamp'].max().isoformat()
                },
                'operating_hours': {
                    'weekdays': '07:00-19:00',
                    'saturday': '08:00-15:00'
                }
            }
        }
        
        # Guardar artefactos
        logger.info("Guardando artefactos del modelo...")
        paths = save_model_artifacts(predictor, metadata, output_dir, timestamp)
        
        logger.info("\nEntrenamiento y validación completados exitosamente!")
        logger.info(f"Modelo guardado en: {paths['model']}")
        logger.info(f"Metadata guardada en: {paths['metadata']}")
        
        # Imprimir resumen de métricas
        if validation_results:
            logger.info("\nResumen de Validación:")
            logger.info(f"Error Absoluto Medio: {validation_results['comparison']['mae']:.3f}")
            logger.info(f"RMSE: {validation_results['comparison']['rmse']:.3f}")
            logger.info(f"R²: {validation_results['comparison']['r2']:.3f}")
            logger.info(f"Score de Rendimiento Medio: {validation_results['performance']['mean']:.3f}")
            logger.info(f"Estabilidad del Rendimiento: {validation_results['performance']['stability']:.3f}")
            
            logger.info("\nImportancia de Features:")
            importance_sorted = sorted(
                training_metrics['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in importance_sorted:
                logger.info(f"{feature}: {importance:.3f}")
        
        return {
            'status': 'success',
            'model_path': paths['model'],
            'metadata_path': paths['metadata'],
            'metrics': validation_results
        }
        
    except Exception as e:
        logger.error(f"\nError en el proceso de entrenamiento: {str(e)}")
        raise

def load_latest_model(output_dir='models'):
    """Cargar el modelo más reciente"""
    try:
        # Buscar el último modelo
        model_files = [f for f in os.listdir(output_dir) if f.startswith('model_') and f.endswith('.joblib')]
        if not model_files:
            logger.warning("No se encontraron modelos guardados")
            return None
        
        latest_model = max(model_files)
        model_path = os.path.join(output_dir, latest_model)
        
        # Cargar modelo
        predictor = DialerRatioPredictor.load(model_path)
        logger.info(f"Modelo cargado exitosamente desde: {model_path}")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        logger.info("Iniciando proceso de entrenamiento...")
        result = main()
        if result and result['status'] == 'success':
            logger.info("Proceso completado exitosamente")
        else:
            logger.warning("El proceso no se completó correctamente")
    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}")
        raise