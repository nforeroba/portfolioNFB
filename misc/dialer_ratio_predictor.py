# dialer_ratio_predictor.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
from datetime import datetime, timedelta

class DialerRatioPredictor:
    def __init__(self, 
                 target_contact_rate=0.3,
                 target_availability_rate=0.6,
                 max_abandon_rate=0.03,
                 weights={
                     'contact_rate': 0.4,
                     'availability_rate': 0.3,
                     'abandon_rate': 0.3
                 },
                 model_id=None):
        """
        Inicializar el predictor de ratio de marcación.
        
        Args:
            target_contact_rate (float): Meta de tasa de contacto (0.0 a 1.0)
            target_availability_rate (float): Meta de tasa de disponibilidad (0.0 a 1.0)
            max_abandon_rate (float): Máxima tasa de abandono permitida (0.0 a 0.1)
            weights (dict): Pesos para cada componente del score
            model_id (str): Identificador del modelo
        """
        # Validar y establecer metas
        self.target_contact_rate = np.clip(target_contact_rate, 0.0, 1.0)
        self.target_availability_rate = np.clip(target_availability_rate, 0.0, 1.0)
        self.max_abandon_rate = np.clip(max_abandon_rate, 0.0, 0.1)
        
        # Validar y normalizar pesos
        total_weight = sum(weights.values())
        self.weights = {k: v/total_weight for k, v in weights.items()}
        
        # Configuración del modelo XGBoost
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            tree_method='exact',
            enable_categorical=False,
            eval_metric=['rmse', 'mae'],
            early_stopping_rounds=10,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.feature_importance = {}
        self.model_id = model_id
        
        # Cache para datos procesados
        self._cache = {}

    def _cache_key(self, df):
        """Generar clave de cache basada en el dataframe"""
        if df is None or df.empty:
            return None
        return hash((df.index[0], df.index[-1], len(df)))

    def resample_data(self, df):
        """Procesar datos a nivel de minuto con cache"""
        cache_key = self._cache_key(df)
        if cache_key in self._cache:
            return self._cache[cache_key]['resampled']
            
        df = df.copy()
        
        # Convertir timestamp si es necesario
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Redondear timestamps al minuto
        df['timestamp'] = df['timestamp'].dt.floor('T')
        
        # Almacenar en cache
        self._cache[cache_key] = {'resampled': df}
        
        return df

    def calculate_performance_score(self, df):
        """
        Calcular score de rendimiento basado en métricas objetivo.
        
        El score se calcula como una combinación ponderada de:
        - Qué tan cerca está la tasa de contacto del objetivo
        - Qué tan cerca está la tasa de disponibilidad del objetivo
        - Qué tan lejos está la tasa de abandono del máximo permitido
        
        Returns:
            numpy.ndarray: Array de scores normalizados entre 0 y 1
        """
        # Calcular tasas clave
        dials_gt_0 = df['dials'] > 0
        agents_connected_gt_0 = df['agents_connected'] > 0
        
        # Tasa de contacto
        contact_rate = np.where(dials_gt_0,
                              df['contacts'] / df['dials'],
                              0)
        
        # Tasa de disponibilidad
        availability_rate = np.where(agents_connected_gt_0,
                                   df['agents_available'] / df['agents_connected'],
                                   0)
        
        # Tasa de abandono
        abandon_rate = np.where(dials_gt_0,
                              df['abandonments'] / df['dials'],
                              0)
        
        # Calcular scores individuales
        contact_score = 1 - np.clip(
            abs(contact_rate - self.target_contact_rate) / max(self.target_contact_rate, 0.001), 
            0, 
            1
        )
        
        availability_score = 1 - np.clip(
            abs(availability_rate - self.target_availability_rate) / max(self.target_availability_rate, 0.001),
            0,
            1
        )
        
        abandon_score = 1 - np.clip(
            abandon_rate / max(self.max_abandon_rate, 0.001),
            0,
            1
        )
        
        # Calcular score ponderado
        return (
            self.weights['contact_rate'] * contact_score +
            self.weights['availability_rate'] * availability_score +
            self.weights['abandon_rate'] * abandon_score
        )

    def prepare_features(self, df):
        """Preparar features para XGBoost con cache"""
        cache_key = self._cache_key(df)
        if cache_key in self._cache and 'features' in self._cache[cache_key]:
            return self._cache[cache_key]['features']
            
        features = pd.DataFrame(index=df.index)
        
        # Features básicas
        features['agents_connected'] = df['agents_connected']
        features['agents_available'] = df['agents_available']
        
        # Features temporales
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Período del día
        conditions = [
            # Lunes a viernes
            ((df['timestamp'].dt.dayofweek.isin([0,1,2,3,4])) & 
             (df['timestamp'].dt.hour >= 7) & 
             (df['timestamp'].dt.hour < 11)),
            ((df['timestamp'].dt.dayofweek.isin([0,1,2,3,4])) & 
             (df['timestamp'].dt.hour >= 11) & 
             (df['timestamp'].dt.hour < 14)),
            ((df['timestamp'].dt.dayofweek.isin([0,1,2,3,4])) & 
             (df['timestamp'].dt.hour >= 14) & 
             (df['timestamp'].dt.hour < 19)),
            # Sábados
            ((df['timestamp'].dt.dayofweek == 5) & 
             (df['timestamp'].dt.hour >= 8) & 
             (df['timestamp'].dt.hour < 11)),
            ((df['timestamp'].dt.dayofweek == 5) & 
             (df['timestamp'].dt.hour >= 11) & 
             (df['timestamp'].dt.hour < 15))
        ]
        choices = [0, 1, 2, 0, 1]  # 0: mañana, 1: medio día, 2: tarde
        features['part_of_day'] = np.select(conditions, choices, default=-1)
        
        # Feature de duración de llamada
        features['avg_call_duration'] = df['call_duration']
        
        # Limpieza final
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        if self.feature_columns is None:
            self.feature_columns = features.columns.tolist()
        
        # Almacenar en cache
        self._cache[cache_key] = self._cache.get(cache_key, {})
        self._cache[cache_key]['features'] = features
        
        return features

    def find_optimal_ratios(self, df):
        """Encontrar ratios óptimos basados en rendimiento con cache"""
        cache_key = self._cache_key(df)
        if cache_key in self._cache and 'optimal_ratios' in self._cache[cache_key]:
            return self._cache[cache_key]['optimal_ratios']
            
        df = df.copy()
        
        # Calcular ratio de marcación
        agents_available_gt_0 = df['agents_available'] > 0
        df['dialing_ratio'] = np.where(agents_available_gt_0,
                                     df['dials'] / df['agents_available'],
                                     1)  # Valor por defecto de 1 cuando no hay agentes disponibles
        df['dialing_ratio'] = np.clip(df['dialing_ratio'], 1, None)  # Clip con mínimo de 1
        
        # Calcular score de rendimiento
        df['performance_score'] = self.calculate_performance_score(df)
        
        # Agregar features temporales
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Agrupar y encontrar ratio óptimo
        feature_cols = ['hour', 'day_of_week', 'agents_available']
        optimal_ratios = df.groupby(feature_cols).apply(
            lambda x: pd.Series({
                'dialing_ratio': x.loc[x['performance_score'].idxmax(), 'dialing_ratio'],
                'max_performance': x['performance_score'].max()
            })
        ).reset_index()
        
        # Almacenar en cache
        self._cache[cache_key] = self._cache.get(cache_key, {})
        self._cache[cache_key]['optimal_ratios'] = optimal_ratios
        
        return optimal_ratios

    def train(self, df):
        """Entrenar modelo con validación cruzada y early stopping"""
        if len(df) < 100:
            df = pd.concat([df] * (100 // len(df) + 1))
            
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Procesar datos
        formatted_data = self.resample_data(df)
        optimal_ratios = self.find_optimal_ratios(formatted_data)
        features = self.prepare_features(formatted_data)
        
        # Preparar datos de entrenamiento
        training_data = pd.merge(
            features,
            optimal_ratios[['hour', 'day_of_week', 'agents_available', 'dialing_ratio']],
            on=['hour', 'day_of_week', 'agents_available'],
            how='left'
        )
        
        # Calcular ratio por defecto
        default_ratio = 1.0
        if formatted_data['agents_available'].median() > 0:
            default_ratio = max(1.0, (formatted_data['dials'].median() / 
                              formatted_data['agents_available'].median()))
        
        # Limpiar datos
        training_data['dialing_ratio'] = training_data['dialing_ratio'].fillna(default_ratio)
        training_data = training_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Dividir datos
        X = training_data[self.feature_columns]
        y = training_data['dialing_ratio']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Configurar early stopping
        eval_set = [(X_test_scaled, y_test)]
        
        # Entrenar modelo
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Calcular importancia de features
        importance_scores = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, importance_scores))
        
        # Calcular métricas
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'feature_importance': self.feature_importance,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
            'best_score': self.model.best_score if hasattr(self.model, 'best_score') else None
        }
        
        # Limpiar cache
        self._cache.clear()
        
        return metrics, formatted_data

    def predict(self, features_df):
        """Realizar predicciones con el modelo entrenado"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.feature_columns is None:
            raise ValueError("Feature columns not initialized. Run train() first")
        
        # Preparar features para predicción
        features = features_df[self.feature_columns].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Escalar features y predecir
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        # Asegurar que ninguna predicción sea menor a 1
        return np.clip(predictions, 1, None)

    def save(self, path):
        """Guardar modelo y parámetros"""
        model_state = {
            'model': self.model,
            'scaler': self.scaler,
            'target_contact_rate': self.target_contact_rate,
            'target_availability_rate': self.target_availability_rate,
            'max_abandon_rate': self.max_abandon_rate,
            'weights': self.weights,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'model_id': self.model_id
        }
        joblib.dump(model_state, path)

    @classmethod
    def load(cls, path):
        """Cargar modelo desde archivo"""
        model_state = joblib.load(path)
        
        predictor = cls(
            target_contact_rate=model_state['target_contact_rate'],
            target_availability_rate=model_state['target_availability_rate'],
            max_abandon_rate=model_state['max_abandon_rate'],
            weights=model_state['weights'],
            model_id=model_state['model_id']
        )
        
        predictor.model = model_state['model']
        predictor.scaler = model_state['scaler']
        predictor.feature_columns = model_state['feature_columns']
        predictor.is_trained = model_state['is_trained']
        predictor.feature_importance = model_state['feature_importance']
        
        return predictor

    def get_model_info(self):
        """Obtener información del modelo"""
        return {
            'model_id': self.model_id,
            'target_metrics': {
                'contact_rate': self.target_contact_rate,
                'availability_rate': self.target_availability_rate,
                'max_abandon_rate': self.max_abandon_rate
            },
            'weights': self.weights,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance,
            'model_params': self.model.get_params(),
            'training_info': {
                'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
                'best_score': self.model.best_score if hasattr(self.model, 'best_score') else None
            }
        }

    def update_model(self, new_data):
        """Actualizar modelo con nuevos datos"""
        if not self.is_trained:
            return self.train(new_data)
        
        # Procesar nuevos datos
        if not pd.api.types.is_datetime64_any_dtype(new_data['timestamp']):
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        
        formatted_data = self.resample_data(new_data)
        optimal_ratios = self.find_optimal_ratios(formatted_data)
        features = self.prepare_features(formatted_data)
        
        # Preparar datos de actualización
        update_data = pd.merge(
            features,
            optimal_ratios[['hour', 'day_of_week', 'agents_available', 'dialing_ratio']],
            on=['hour', 'day_of_week', 'agents_available'],
            how='left'
        )
        
        # Calcular ratio por defecto
        default_ratio = 1.0
        if formatted_data['agents_available'].median() > 0:
            default_ratio = max(1.0, (formatted_data['dials'].median() / 
                              formatted_data['agents_available'].median()))
        
        # Limpiar datos
        update_data['dialing_ratio'] = update_data['dialing_ratio'].fillna(default_ratio)
        update_data = update_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Preparar datos para actualización
        X = update_data[self.feature_columns]
        y = update_data['dialing_ratio']
        
        # Escalar features
        X_scaled = self.scaler.transform(X)
        
        # Actualizar modelo
        self.model.fit(
            X_scaled, 
            y,
            xgb_model=self.model.get_booster() if hasattr(self.model, 'get_booster') else None
        )
        
        # Actualizar importancia de features
        importance_scores = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, importance_scores))
        
        # Calcular métricas de actualización
        update_metrics = {
            'status': 'Model updated successfully',
            'samples_processed': len(new_data),
            'feature_importance': self.feature_importance,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
            'best_score': self.model.best_score if hasattr(self.model, 'best_score') else None
        }
        
        # Limpiar cache
        self._cache.clear()
        
        return update_metrics