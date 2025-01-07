# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import subprocess
import webbrowser
import os
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import joblib
from sklearn.preprocessing import StandardScaler
import json
from dialer_ratio_predictor import DialerRatioPredictor
import requests
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración inicial
MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"

def obtener_datos_historicos(cartera, campania):
    """Obtener datos históricos usando Aut_Historic.py"""
    try:
        # Paths de configuración
        paths = {
            'config_click': "Actualizaciones/func/conf_clickhouse/config_click.ini",
            'key_click': "Actualizaciones/func/conf_clickhouse/clave_click.key",
            'credenciales_click': "Actualizaciones/func/conf_clickhouse/credenciales_click.txt",
            'consult': "Actualizaciones/func/conf_clickhouse/consult_info_var_discado.sql",
            'ultima_fecha': "Actualizaciones/func/conf_clickhouse/fecha_max_variables.sql"
        }
        
        # Fechas de ejecución
        fecha_final = (datetime.now() - timedelta(days=1))
        
        # Si no hay cartera ni campaña seleccionada, usar últimos 7 días
        if not cartera and not campania:
            fecha_inicial = fecha_final - timedelta(days=7)
        else:
            fecha_inicial = datetime.strptime("2024-12-02", "%Y-%m-%d")
            
        # Convertir a string en formato YYYY-MM-DD
        fecha_inicial = fecha_inicial.strftime("%Y-%m-%d")
        fecha_final = fecha_final.strftime("%Y-%m-%d")
        
        # Construir filtro
        filtro_general = f"where timestamp between '{fecha_inicial} 00:00:00' and '{fecha_final} 23:00:00'"
        if campania:
            filtro = f"{filtro_general} and campania = '{campania}'"
        elif cartera:
            filtro = f"{filtro_general} and cartera = '{cartera}'"
        else:
            filtro = filtro_general
            
        # Obtener datos
        eventos_disponibles = ['Conectado', 'GestionReady', 'GestionManual', 'Disponible', 'Colgar', 'Colgo']
        eventos_conectado = ['Conectado', 'GestionReady', 'GestionLlamada', 'ACW', 'GestionManual', 'MarcacionFallo', 'GestionMarcacion', 'Disponible', 'MarcandoManual', 'Enllamada', 'Colgar', 'Colgo', 'ATENCION AL CLIENTE']
        filtro_col_discado = ["timestamp", "estado_registro", "segundos_llamada"]
        filtro_col_conexion = ["timestamp", "id_usuario", "evento"]
        
        # Obtener datos de discado y conexión usando las funciones importadas
        from Actualizaciones.func.fun_historico import extracion_datos, ordenar_discado, ordenar_conexion
        
        # Obtener datos de discado
        db_d, rows_discado = extracion_datos("tbl_variables_discado", fecha_inicial, fecha_final, 
                                           paths['config_click'], paths['key_click'], 
                                           paths['credenciales_click'], paths['consult'], 
                                           filtro, filtro_col_discado)
        db_discado = ordenar_discado(db_d, "tbl_variables_discado")
        
        # Obtener datos de conexión
        db_c, rows_conexion = extracion_datos("tbl_variables_conexion", fecha_inicial, fecha_final,
                                            paths['config_click'], paths['key_click'],
                                            paths['credenciales_click'], paths['consult'],
                                            filtro, filtro_col_conexion)
        db_conexion = ordenar_conexion(db_c, "tbl_variables_conexion", eventos_disponibles, eventos_conectado)
        
        # Unir datos
        if rows_discado > 0 and rows_conexion > 0:
            return pd.merge(db_discado, db_conexion, on="timestamp")
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error obteniendo datos históricos: {e}")
        return pd.DataFrame()

def obtener_datos_actuales(cartera, campania):
    """Obtener datos actuales usando Aut_Real_Time.py"""
    try:
        # Paths de configuración
        paths = {
            'config_mysql': "Actualizaciones/func/conf_mysql/config_mysql.ini",
            'key_mysql': "Actualizaciones/func/conf_mysql/clave_mysql.key",
            'credenciales_mysql': "Actualizaciones/func/conf_mysql/credenciales_mysql.txt",
            'sql_mysql_discado': "Actualizaciones/func/conf_mysql/ConsultaAct_discado.sql",
            'sql_mysql_usuarios': "Actualizaciones/func/conf_mysql/ConsultaAct_Usuarios.sql"
        }
        
        # Construir filtro
        if campania:
            filtro = f"and campania = '{campania}'"
        elif cartera:
            filtro = f"and cartera = '{cartera}'"
        else:
            filtro = ""
            
        # Obtener datos actuales
        fecha_actual = (pd.Timestamp.now()).strftime("%Y-%m-%d")
        from Actualizaciones.func.conexion_act import conexion_base_act
        return conexion_base_act(
            paths['config_mysql'],
            paths['key_mysql'],
            paths['credenciales_mysql'],
            paths['sql_mysql_discado'],
            paths['sql_mysql_usuarios'],
            fecha_actual,
            filtro,
            "aecsoft_marcador"
        )
        
    except Exception as e:
        logger.error(f"Error obteniendo datos actuales: {e}")
        return pd.DataFrame()

def safe_int_convert(value, default=0):
    """Convertir valor a entero de manera segura"""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default

def initialize_mlflow():
    """Iniciar servidor MLflow de manera optimizada"""
    try:
        # Verificar si el servidor ya está corriendo
        try:
            response = requests.get(MLFLOW_TRACKING_URI)
            if response.status_code == 200:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                return
        except:
            pass
        
        # Si no está corriendo, iniciarlo
        subprocess.Popen(["mlflow", "server", "--host", "127.0.0.1", "--port", "8080"])
        
        # Esperar a que el servidor esté listo con timeout
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 segundos máximo
            try:
                response = requests.get(MLFLOW_TRACKING_URI)
                if response.status_code == 200:
                    break
            except:
                time.sleep(0.5)
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        webbrowser.open(MLFLOW_TRACKING_URI)
    except Exception as e:
        logger.error(f"Error iniciando MLflow: {e}")
        st.error(f"Error iniciando MLflow: {e}")

def get_latest_model_id(model_id):
    """Obtener el último modelo registrado para un ID específico"""
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        
        model_name = f"dialer_ratio_predictor_{model_id}"
        matching_models = [m for m in models if m.name == model_name]
        
        if not matching_models:
            return None
            
        latest_model = max(matching_models, key=lambda m: m.creation_timestamp)
        return latest_model.latest_versions[0].run_id
    except Exception as e:
        logger.error(f"Error obteniendo ID del último modelo: {e}")
        return None

def should_train_model(model_id):
    """Determinar si se debe entrenar un nuevo modelo"""
    try:
        latest_model_id = get_latest_model_id(model_id)
        if not latest_model_id:
            return True
            
        # Verificar si existe caché local
        local_model_path = f"models/{model_id}/model_{latest_model_id}.joblib"
        local_scaler_path = f"models/{model_id}/scaler_{latest_model_id}.joblib"
        
        if not os.path.exists(local_model_path) or not os.path.exists(local_scaler_path):
            return True
        
        run = mlflow.get_run(latest_model_id)
        training_date = datetime.fromtimestamp(run.info.start_time/1000.0).date()
        return training_date != datetime.now().date()
    except:
        return True
    
def load_model_from_mlflow(model_id, mlflow_run_id):
    """Cargar modelo desde MLflow"""
    try:
        client = MlflowClient()
        run = client.get_run(mlflow_run_id)
        params = run.data.params
        
        predictor = DialerRatioPredictor(
            target_contact_rate=float(params.get('target_contact_rate', 0.3)),
            target_availability_rate=float(params.get('target_availability_rate', 0.6)),
            max_abandon_rate=float(params.get('max_abandon_rate', 0.03)),
            weights={
                'contact_rate': float(params.get('contact_rate_weight', 0.4)),
                'availability_rate': float(params.get('availability_rate_weight', 0.3)),
                'abandon_rate': float(params.get('abandon_rate_weight', 0.3))
            },
            model_id=model_id
        )
        
        local_model_path = f"models/{model_id}/model_{mlflow_run_id}.joblib"
        if os.path.exists(local_model_path):
            predictor.model = joblib.load(local_model_path)
        else:
            predictor.model = mlflow.xgboost.load_model(f"runs:/{mlflow_run_id}/model")
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            joblib.dump(predictor.model, local_model_path)
        
        predictor.is_trained = True
        
        try:
            local_scaler_path = f"models/{model_id}/scaler_{mlflow_run_id}.joblib"
            if os.path.exists(local_scaler_path):
                predictor.scaler = joblib.load(local_scaler_path)
            else:
                scaler_path = client.download_artifacts(mlflow_run_id, "scaler.joblib")
                predictor.scaler = joblib.load(scaler_path)
                os.makedirs(os.path.dirname(local_scaler_path), exist_ok=True)
                joblib.dump(predictor.scaler, local_scaler_path)
            
            if not hasattr(predictor.scaler, 'mean_') or not hasattr(predictor.scaler, 'scale_'):
                raise ValueError("Scaler cargado no está ajustado correctamente")
                
        except Exception as scaler_error:
            logger.error(f"Error cargando scaler: {scaler_error}")
            st.error(f"Error cargando scaler: {str(scaler_error)}")
            return None
        
        return predictor
        
    except Exception as e:
        logger.error(f"Error cargando modelo desde MLflow: {e}")
        st.error(f"Error cargando modelo desde MLflow: {str(e)}")
        return None

def calculate_window_statistics(data, window_minutes):
    """Calcular estadísticas para una ventana de tiempo"""
    if data.empty:
        return None
        
    try:
        current_time = data['timestamp'].max()
        window_start = current_time - pd.Timedelta(minutes=window_minutes)
        window_data = data[data['timestamp'] >= window_start].copy()
        
        # Calcular tasas
        dials_gt_0 = window_data['dials'] > 0
        agents_connected_gt_0 = window_data['agents_connected'] > 0
        agents_available_gt_0 = window_data['agents_available'] > 0
        
        contact_rates = np.where(dials_gt_0,
                               window_data['contacts'] / window_data['dials'],
                               0)
        
        abandon_rates = np.where(dials_gt_0,
                               window_data['abandonments'] / window_data['dials'],
                               0)
        
        availability_rates = np.where(agents_connected_gt_0,
                                    window_data['agents_available'] / window_data['agents_connected'],
                                    0)
        
        avg_call_duration = np.where(window_data['contacts'] > 0,
                                   window_data['call_duration'],
                                   0)
        
        dialing_ratio = np.where(agents_available_gt_0,
                                window_data['dials'] / window_data['agents_available'],
                                1)
        dialing_ratio = np.clip(dialing_ratio, 1, None)
        
        if 'predictions' in st.session_state:
            window_predictions = st.session_state.predictions[
                st.session_state.predictions['timestamp'] >= window_start
            ]
            if not window_predictions.empty:
                optimal_ratio = np.clip(window_predictions['optimal_ratio'], 1, None)
        
        stats = {
            'Time Window': f'Last {window_minutes} min',
            'Dialing Ratio Mean': np.mean(dialing_ratio),
            'Dialing Ratio Min': np.min(dialing_ratio),
            'Dialing Ratio Max': np.max(dialing_ratio),
            'Dialing Ratio StdDev': np.std(dialing_ratio),
            'Optimal Ratio Mean': np.mean(optimal_ratio) if 'optimal_ratio' in locals() else np.nan,
            'Optimal Ratio Min': np.min(optimal_ratio) if 'optimal_ratio' in locals() else np.nan,
            'Optimal Ratio Max': np.max(optimal_ratio) if 'optimal_ratio' in locals() else np.nan,
            'Optimal Ratio StdDev': np.std(optimal_ratio) if 'optimal_ratio' in locals() else np.nan,
            'Contact Rate Mean': np.mean(contact_rates),
            'Contact Rate Min': np.min(contact_rates),
            'Contact Rate Max': np.max(contact_rates),
            'Contact Rate StdDev': np.std(contact_rates),
            'Abandon Rate Mean': np.mean(abandon_rates),
            'Abandon Rate Min': np.min(abandon_rates),
            'Abandon Rate Max': np.max(abandon_rates),
            'Abandon Rate StdDev': np.std(abandon_rates),
            'Availability Rate Mean': np.mean(availability_rates),
            'Availability Rate Min': np.min(availability_rates),
            'Availability Rate Max': np.max(availability_rates),
            'Availability Rate StdDev': np.std(availability_rates),
            'Avg Call Duration Mean': np.mean(avg_call_duration),
            'Avg Call Duration Min': np.min(avg_call_duration),
            'Avg Call Duration Max': np.max(avg_call_duration),
            'Avg Call Duration StdDev': np.std(avg_call_duration)
        }
        
        return pd.Series(stats)
        
    except Exception as e:
        logger.error(f"Error calculando estadísticas de ventana: {e}")
        return None
    
def create_statistics_df(data):
    """Crear DataFrame de estadísticas para diferentes ventanas de tiempo"""
    try:
        time_windows = [10, 20, 30, 50, 80]
        stats_list = []
        
        data = data.sort_values('timestamp')
        total_elapsed = data['timestamp'].max() - data['timestamp'].min()
        
        for window in time_windows:
            if total_elapsed >= pd.Timedelta(minutes=window):
                window_stats = calculate_window_statistics(data, window)
                if window_stats is not None:
                    stats_list.append(window_stats)
        
        if not stats_list:
            return pd.DataFrame()
            
        stats_df = pd.DataFrame(stats_list)
        stats_df.set_index('Time Window', inplace=True)
        return stats_df
        
    except Exception as e:
        logger.error(f"Error creando DataFrame de estadísticas: {e}")
        return pd.DataFrame()

def display_statistics_table(stats_df):
    """Mostrar tablas de estadísticas"""
    try:
        st.subheader("Estadísticas")
        
        if stats_df.empty:
            st.info("Las estadísticas aparecerán cuando se completen las ventanas de tiempo (10, 20, 30, 50 y 80 minutos)")
            return
        
        # Primera fila: Estadísticas de Ratio y Duración de Llamadas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Estadísticas de Ratio de Marcación**")
            dialing_stats = pd.DataFrame({
                'Media': stats_df['Dialing Ratio Mean'],
                'Mínimo': stats_df['Dialing Ratio Min'],
                'Máximo': stats_df['Dialing Ratio Max'],
                'Desv.Est': stats_df['Dialing Ratio StdDev']
            })
            st.dataframe(
                dialing_stats.style.format("{:.3f}")
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
            )
        
        with col2:
            st.markdown("**Estadísticas de Ratio Óptimo**")
            optimal_stats = pd.DataFrame({
                'Media': stats_df['Optimal Ratio Mean'],
                'Mínimo': stats_df['Optimal Ratio Min'],
                'Máximo': stats_df['Optimal Ratio Max'],
                'Desv.Est': stats_df['Optimal Ratio StdDev']
            })
            st.dataframe(
                optimal_stats.style.format("{:.3f}")
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
            )
        
        with col3:
            st.markdown("**Estadísticas de Duración de Llamadas (seg)**")
            duration_stats = pd.DataFrame({
                'Media': stats_df['Avg Call Duration Mean'],
                'Mínimo': stats_df['Avg Call Duration Min'],
                'Máximo': stats_df['Avg Call Duration Max'],
                'Desv.Est': stats_df['Avg Call Duration StdDev']
            })
            st.dataframe(
                duration_stats.style.format("{:.1f}")
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
            )
        
        # Segunda fila: Estadísticas de tasas
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("**Estadísticas de Tasa de Contacto**")
            contact_stats = pd.DataFrame({
                'Media': stats_df['Contact Rate Mean'],
                'Mínimo': stats_df['Contact Rate Min'],
                'Máximo': stats_df['Contact Rate Max'],
                'Desv.Est': stats_df['Contact Rate StdDev']
            })
            st.dataframe(
                contact_stats.style.format("{:.3f}")
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
            )
        
        with col5:
            st.markdown("**Estadísticas de Tasa de Abandono**")
            abandon_stats = pd.DataFrame({
                'Media': stats_df['Abandon Rate Mean'],
                'Mínimo': stats_df['Abandon Rate Min'],
                'Máximo': stats_df['Abandon Rate Max'],
                'Desv.Est': stats_df['Abandon Rate StdDev']
            })
            st.dataframe(
                abandon_stats.style.format("{:.3f}")
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
            )
        
        with col6:
            st.markdown("**Estadísticas de Tasa de Disponibilidad**")
            availability_stats = pd.DataFrame({
                'Media': stats_df['Availability Rate Mean'],
                'Mínimo': stats_df['Availability Rate Min'],
                'Máximo': stats_df['Availability Rate Max'],
                'Desv.Est': stats_df['Availability Rate StdDev']
            })
            st.dataframe(
                availability_stats.style.format("{:.3f}")
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'center')]},
                    {'selector': 'td', 'props': [('text-align', 'center')]}
                ])
            )
            
    except Exception as e:
        logger.error(f"Error mostrando tablas de estadísticas: {e}")
        st.error("Error al mostrar las estadísticas")

def train_or_load_model(model_id, db_historico, model_params=None):
    """Entrenar nuevo modelo o cargar el existente con parámetros personalizados"""
    try:
        if should_train_model(model_id):
            # Guardar datos históricos temporalmente
            temp_csv = f"temp_{model_id}_historico.csv"
            db_historico.to_csv(temp_csv, index=False)
            
            # Establecer parámetros del modelo en variables de entorno
            os.environ['MODEL_ID'] = model_id
            os.environ['DATA_PATH'] = temp_csv
            if model_params:
                os.environ['MODEL_PARAMS'] = json.dumps(model_params)
            
            # Entrenar nuevo modelo con los parámetros personalizados
            import mlflow_script
            predictor = mlflow_script.train_and_log_model()
            
            os.remove(temp_csv)
            return predictor
        else:
            # Cargar modelo existente
            mlflow_run_id = get_latest_model_id(model_id)
            return load_model_from_mlflow(model_id, mlflow_run_id)
    except Exception as e:
        logger.error(f"Error en train_or_load_model: {e}")
        st.error(f"Error al entrenar o cargar modelo: {str(e)}")
        return None

def create_realtime_dashboard(cartera, campania, model_id, model_params=None):
    """Crear dashboard en tiempo real"""
    try:
        # Título dinámico según el nivel de análisis
        if campania:
            titulo = f"Campaña: {campania}"
            if cartera:
                titulo += f" (Cartera: {cartera})"
        elif cartera:
            titulo = f"Cartera: {cartera}"
        else:
            titulo = "Análisis Global"
            
        st.title(f"Modelo de Recomendación de Ratio de Marcación - {titulo}")

        # No necesitamos la explicación aquí ya que se movió al inicio
        
        if 'predictor' not in st.session_state or st.session_state.get('current_model_id') != model_id:
            # Obtener datos históricos
            db_historico = obtener_datos_historicos(cartera, campania)
            
            if db_historico.empty:
                st.error("No hay datos históricos disponibles para el análisis")
                return
                
            # Entrenar o cargar modelo
            predictor = train_or_load_model(model_id, db_historico, model_params)
            
            if predictor is None:
                st.error("Error al cargar o entrenar el modelo")
                return
                
            st.session_state.predictor = predictor
            st.session_state.current_model_id = model_id
            st.session_state.data_buffer = pd.DataFrame()
            st.session_state.predictions = pd.DataFrame()
        
        placeholder = st.empty()
        
        # Agregar botón para detener el dashboard
        if 'stop_dashboard' not in st.session_state:
            st.session_state.stop_dashboard = False
        
        col1, col2, col3 = st.columns([2,1,2])
        with col2:
            if st.button('Detener Dashboard', type="secondary"):
                st.session_state.stop_dashboard = True
                st.experimental_rerun()
        
        while not st.session_state.stop_dashboard:
            # Obtener datos actuales
            db_real_time = obtener_datos_actuales(cartera, campania)
            
            if not db_real_time.empty:
                # Asegurar que timestamp es datetime
                if not pd.api.types.is_datetime64_any_dtype(db_real_time['timestamp']):
                    db_real_time['timestamp'] = pd.to_datetime(db_real_time['timestamp'])
                
                if st.session_state.data_buffer.empty:
                    st.session_state.data_buffer = db_real_time
                    features = st.session_state.predictor.prepare_features(db_real_time)
                    predictions = st.session_state.predictor.predict(features)
                    st.session_state.predictions = pd.DataFrame({
                        'timestamp': db_real_time['timestamp'],
                        'optimal_ratio': predictions
                    })
                else:
                    new_timestamps = db_real_time[~db_real_time['timestamp'].isin(st.session_state.data_buffer['timestamp'])]
                    if not new_timestamps.empty:
                        st.session_state.data_buffer = pd.concat(
                            [st.session_state.data_buffer, new_timestamps],
                            ignore_index=True
                        )
                        features = st.session_state.predictor.prepare_features(new_timestamps)
                        new_predictions = st.session_state.predictor.predict(features)
                        new_predictions_df = pd.DataFrame({
                            'timestamp': new_timestamps['timestamp'],
                            'optimal_ratio': new_predictions
                        })
                        st.session_state.predictions = pd.concat(
                            [st.session_state.predictions, new_predictions_df],
                            ignore_index=True
                        )
                
                with placeholder.container():
                    # Mostrar métricas actuales
                    latest = db_real_time.iloc[-1]
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Agentes Disponibles", safe_int_convert(latest['agents_available']))
                        st.markdown("ℹ️ *Número de agentes listos para atender llamadas en el último minuto*")
                        st.metric("Agentes Conectados", safe_int_convert(latest['agents_connected']))
                        st.markdown("ℹ️ *Número total de agentes conectados en el último minuto*")
                    
                    with metrics_col2:
                        st.metric("Marcaciones Actuales", safe_int_convert(latest['dials']))
                        st.markdown("ℹ️ *Número de llamadas realizadas en el último minuto*")
                        st.metric("Contactos", safe_int_convert(latest['contacts']))
                        st.markdown("ℹ️ *Número de llamadas contestadas en el último minuto*")
                    
                    with metrics_col3:
                        current_ratio = 1
                        if safe_int_convert(latest['agents_available']) > 0:
                            current_ratio = max(1, safe_int_convert(latest['dials']) / 
                                                safe_int_convert(latest['agents_available']))
                        
                        latest_prediction = (st.session_state.predictions['optimal_ratio'].iloc[-1] 
                                           if not st.session_state.predictions.empty else np.nan)
                        
                        st.metric("Ratio Actual", f"{current_ratio:.2f}")
                        st.markdown("ℹ️ *Número de marcaciones por agente disponible en el último minuto*")
                        if not np.isnan(latest_prediction):
                            st.metric("Ratio Recomendado", f"{latest_prediction:.2f}")
                            st.markdown("ℹ️ *Ratio óptimo sugerido por el modelo en el último minuto*")
                    
                    stats_df = create_statistics_df(st.session_state.data_buffer)
                    display_statistics_table(stats_df)
                    
                    plot_data = st.session_state.data_buffer.sort_values('timestamp')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Ratio de Marcación en el Tiempo**")
                        st.markdown("ℹ️ *Comparación del ratio actual vs. el ratio óptimo recomendado*")
                        
                        fig_ratio = go.Figure()
                        actual_ratio = np.where(
                            plot_data['agents_available'] > 0,
                            plot_data['dials'] / plot_data['agents_available'],
                            1
                        )
                        actual_ratio = np.clip(actual_ratio, 1, None)
                        
                        fig_ratio.add_trace(go.Scatter(
                            x=plot_data['timestamp'],
                            y=actual_ratio,
                            name='Ratio Actual',
                            mode='lines',
                            line_shape='hv',
                            connectgaps=False
                        ))
                        
                        if not st.session_state.predictions.empty:
                            optimal_ratios = st.session_state.predictions['optimal_ratio']
                            fig_ratio.add_trace(go.Scatter(
                                x=st.session_state.predictions['timestamp'],
                                y=optimal_ratios,
                                name='Ratio Óptimo',
                                mode='lines',
                                line_shape='hv',
                                connectgaps=False
                            ))
                        
                        # Calcular el máximo valor para el eje Y
                        max_y_value = max(
                            np.nanmax(actual_ratio) if len(actual_ratio) > 0 else 1,
                            np.nanmax(optimal_ratios) if 'optimal_ratios' in locals() and len(optimal_ratios) > 0 else 1
                        )
                        
                        # Añadir un 10% de margen al máximo
                        max_y_value = max_y_value * 1.1
                        
                        fig_ratio.update_layout(
                            title="Ratio de Marcación en el Tiempo",
                            xaxis_title="Tiempo",
                            yaxis_title="Ratio",
                            yaxis=dict(
                                range=[1, max_y_value],
                                tickformat='.1f'  # Mostrar un decimal
                            ),
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                type='date'
                            )
                        )
                        st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Puntuación de Rendimiento**")
                        st.markdown("ℹ️ *Indicador compuesto que mide la eficiencia general del sistema*")
                        
                        performance = st.session_state.predictor.calculate_performance_score(plot_data)
                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Scatter(
                            x=plot_data['timestamp'],
                            y=performance,
                            name='Rendimiento',
                            mode='lines',
                            line_shape='hv',
                            connectgaps=False
                        ))
                        fig_perf.update_layout(
                            title="Puntuación de Rendimiento",
                            xaxis_title="Tiempo",
                            yaxis_title="Puntuación",
                            yaxis_range=[0, 1],
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                type='date'
                            )
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                    
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.markdown("**Tasa de Contacto**")
                        st.markdown("ℹ️ *Contactos / Números Marcados*")
                        
                        fig_contact = go.Figure()
                        contact_rate = np.where(
                            plot_data['dials'] > 0,
                            plot_data['contacts'] / plot_data['dials'],
                            0
                        )
                        fig_contact.add_trace(go.Scatter(
                            x=plot_data['timestamp'],
                            y=contact_rate,
                            name='Tasa de Contacto',
                            mode='lines',
                            line_shape='hv',
                            connectgaps=False
                        ))
                        fig_contact.add_hline(
                            y=st.session_state.predictor.target_contact_rate,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Objetivo ({st.session_state.predictor.target_contact_rate:.3f})"
                        )
                        fig_contact.update_layout(
                            title="Tasa de Contacto en el Tiempo",
                            xaxis_title="Tiempo",
                            yaxis_title="Tasa",
                            yaxis_range=[0, 1],
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                type='date'
                            )
                        )
                        st.plotly_chart(fig_contact, use_container_width=True)
                    
                    with col4:
                        st.markdown("**Tasa de Disponibilidad**")
                        st.markdown("ℹ️ *Agentes Disponibles / Agentes Conectados*")
                        
                        fig_avail = go.Figure()
                        availability_rate = np.where(
                            plot_data['agents_connected'] > 0,
                            plot_data['agents_available'] / plot_data['agents_connected'],
                            0
                        )
                        fig_avail.add_trace(go.Scatter(
                            x=plot_data['timestamp'],
                            y=availability_rate,
                            name='Tasa de Disponibilidad',
                            mode='lines',
                            line_shape='hv',
                            connectgaps=False
                        ))
                        fig_avail.add_hline(
                            y=st.session_state.predictor.target_availability_rate,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Objetivo ({st.session_state.predictor.target_availability_rate:.3f})"
                        )
                        fig_avail.update_layout(
                            title="Tasa de Disponibilidad en el Tiempo",
                            xaxis_title="Tiempo",
                            yaxis_title="Tasa",
                            yaxis_range=[0, 1],
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                type='date'
                            )
                        )
                        st.plotly_chart(fig_avail, use_container_width=True)
                    
                    with col5:
                        st.markdown("**Tasa de Abandono**")
                        st.markdown("ℹ️ *Abandonos / Números Marcados*")
                        
                        fig_abandon = go.Figure()
                        abandon_rate = np.where(
                            plot_data['dials'] > 0,
                            plot_data['abandonments'] / plot_data['dials'],
                            0
                        )
                        fig_abandon.add_trace(go.Scatter(
                            x=plot_data['timestamp'],
                            y=abandon_rate,
                            name='Tasa de Abandono',
                            mode='lines',
                            line_shape='hv',
                            connectgaps=False
                        ))
                        fig_abandon.add_hline(
                            y=st.session_state.predictor.max_abandon_rate,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Máximo ({st.session_state.predictor.max_abandon_rate:.3f})"
                        )
                        fig_abandon.update_layout(
                            title="Tasa de Abandono en el Tiempo",
                            xaxis_title="Tiempo",
                            yaxis_title="Tasa",
                            yaxis_range=[0, 0.2],
                            xaxis=dict(
                                rangeslider=dict(visible=True),
                                type='date'
                            )
                        )
                        st.plotly_chart(fig_abandon, use_container_width=True)
            
            time.sleep(60)
            
    except Exception as e:
        logger.error(f"Error en dashboard en tiempo real: {e}")
        st.error(f"Error en el dashboard: {str(e)}")
        
def main():
    """Función principal de la aplicación"""
    try:
        st.set_page_config(page_title="Ratio de Marcación", layout="wide")
        
        st.title("Sistema de Optimización de Ratio de Marcación")
        
        # Explicación inicial del sistema
        st.markdown("""
        ### Descripción del Sistema
        
        Esta herramienta ayuda a optimizar las operaciones de call centers outbound mediante la recomendación 
        de ratios óptimos de marcación (número de llamadas por agente disponible).
        
        #### Funcionalidades Principales:
        - **Predicción en Tiempo Real**: Recomienda el ratio óptimo de marcación basándose en las condiciones actuales.
        - **Monitoreo de KPIs**: Muestra métricas clave como contactabilidad, disponibilidad de agentes y tasa de abandono.
        - **Configuración Flexible**: Permite establecer metas y pesos para los KPIs de forma manual o automática 
          basándose en el comportamiento histórico.
        
        #### Variables Consideradas:
        - **Variables Temporales**: Hora del día, día de la semana
        - **Variables Operativas**: Agentes conectados y disponibles
        - **Métricas de Rendimiento**: Duración promedio de llamadas
        
        #### Objetivos de Optimización:
        - Maximizar la tasa de contacto
        - Mantener una alta disponibilidad de agentes
        - Minimizar la tasa de abandono
        """)
        
        # Inicializar MLflow
        if 'mlflow_initialized' not in st.session_state:
            initialize_mlflow()
            st.session_state.mlflow_initialized = True
        
        # Contenedor para la selección
        with st.container():
            st.write("### Configuración")
            
            # Selector de cartera y campaña
            col1, col2 = st.columns(2)
            
            with col1:
                carteras = ["", "cartera1", "cartera2", "cartera3"]  # Obtener de la base de datos
                cartera = st.selectbox("Seleccionar Cartera (Opcional)", carteras, index=0)
            
            with col2:
                campanias = ["", "campaña1", "campaña2", "campaña3"]  # Obtener de la base de datos
                campania = st.selectbox("Seleccionar Campaña (Opcional)", campanias, index=0)
            
            # Determinar el ID del modelo
            if campania:
                nivel_analisis = f"campaña: {campania}"
                id_modelo = f"campaign_{campania}"
                if cartera:
                    nivel_analisis += f" (cartera seleccionada: {cartera})"
            elif cartera:
                nivel_analisis = f"cartera: {cartera}"
                id_modelo = f"portfolio_{cartera}"
            else:
                nivel_analisis = "global"
                id_modelo = "global"
            
            st.info(f"Análisis a nivel {nivel_analisis}")

            # Selector de modo de configuración
            st.write("### Configuración de Metas y Pesos")
            config_mode = st.radio(
                "Seleccione el modo de configuración:",
                ["Automático basado en histórico", "Manual"],
                help="Modo automático: determina metas y pesos basándose en el comportamiento histórico.\n"
                     "Modo manual: permite configurar metas y pesos específicos."
            )

            can_execute = True  # Por defecto, se puede ejecutar en modo automático

            if config_mode == "Manual":
                # Primera fila: Metas
                st.write("#### Metas")
                col1, col2, col3 = st.columns(3)

                with col1:
                    target_contact_rate = st.slider(
                        "Meta de Tasa de Contacto",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.01,
                        format="%.2f",
                        help="Porcentaje objetivo de llamadas que resultan en contacto (0-100%)"
                    )

                with col2:
                    target_availability_rate = st.slider(
                        "Meta de Tasa de Disponibilidad",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        step=0.01,
                        format="%.2f",
                        help="Porcentaje objetivo de agentes disponibles del total conectado (0-100%)"
                    )

                with col3:
                    max_abandon_rate = st.slider(
                        "Máxima Tasa de Abandono",
                        min_value=0.0,
                        max_value=0.10,
                        value=0.03,
                        step=0.001,
                        format="%.3f",
                        help="Porcentaje máximo aceptable de llamadas abandonadas (0-10%)"
                    )

                # Segunda fila: Pesos
                st.write("#### Pesos")
                st.write("Asigne la importancia relativa a cada métrica (debe sumar 100%):")
                col1, col2, col3 = st.columns(3)

                with col1:
                    contact_rate_weight = st.number_input(
                        "Peso de Tasa de Contacto (%)",
                        min_value=0,
                        max_value=100,
                        value=40,
                        step=5,
                        help="Importancia relativa de la tasa de contacto en el score final"
                    )

                with col2:
                    availability_rate_weight = st.number_input(
                        "Peso de Tasa de Disponibilidad (%)",
                        min_value=0,
                        max_value=100,
                        value=30,
                        step=5,
                        help="Importancia relativa de la tasa de disponibilidad en el score final"
                    )

                with col3:
                    abandon_rate_weight = st.number_input(
                        "Peso de Tasa de Abandono (%)",
                        min_value=0,
                        max_value=100,
                        value=30,
                        step=5,
                        help="Importancia relativa de la tasa de abandono en el score final"
                    )

                # Validación de pesos
                total_weight = contact_rate_weight + availability_rate_weight + abandon_rate_weight
                if total_weight != 100:
                    st.error(f"La suma de los pesos debe ser 100%. Actualmente: {total_weight}%")
                    can_execute = False

                # Configurar parámetros del modelo en modo manual
                if can_execute:
                    model_params = {
                        'target_contact_rate': target_contact_rate,
                        'target_availability_rate': target_availability_rate,
                        'max_abandon_rate': max_abandon_rate,
                        'weights': {
                            'contact_rate': contact_rate_weight / 100,
                            'availability_rate': availability_rate_weight / 100,
                            'abandon_rate': abandon_rate_weight / 100
                        }
                    }
            else:
                # Modo automático
                st.info("Se determinarán automáticamente las metas y pesos basados en el comportamiento histórico.")
                model_params = None

            # Botón de ejecución
            col1, col2, col3 = st.columns([2,1,2])
            with col2:
                ejecutar = st.button("Ejecutar Análisis", type="primary", disabled=not can_execute)
            
            if ejecutar:
                # Resetear el estado de detención si existe
                if 'stop_dashboard' in st.session_state:
                    st.session_state.stop_dashboard = False
                
                # Iniciar dashboard con los parámetros configurados
                create_realtime_dashboard(cartera, campania, id_modelo, model_params)
                
    except Exception as e:
        logger.error(f"Error en la aplicación principal: {e}")
        st.error(f"Error en la aplicación: {str(e)}")

if __name__ == "__main__":
    main()