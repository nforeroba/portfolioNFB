# pages/1_Model_API.py
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import json
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://127.0.0.1:8080"

def initialize_mlflow():
    """Inicializar conexión con MLflow"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        return client
    except Exception as e:
        logger.error(f"Error iniciando MLflow: {e}")
        st.error(f"Error conectando con MLflow: {str(e)}")
        return None

def get_registered_models(client):
    """Obtener lista de modelos registrados"""
    try:
        models = client.search_registered_models()
        return [(model.name, model.latest_versions[-1]) for model in models]
    except Exception as e:
        logger.error(f"Error obteniendo modelos registrados: {e}")
        return []

def get_model_metrics(run_id):
    """Obtener métricas del modelo"""
    try:
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params
        return metrics, params
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        return {}, {}

def make_prediction(model, features, mae):
    """Realizar predicción con intervalo de confianza"""
    try:
        prediction = model.predict(features)[0]
        lower_bound = max(1, prediction - mae)
        upper_bound = prediction + mae
        return prediction, lower_bound, upper_bound
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise

def main():
    st.set_page_config(page_title="Model API", layout="wide")
    st.title("API de Predicción de Ratio de Marcación")
    
    # Inicializar MLflow
    client = initialize_mlflow()
    if not client:
        st.error("No se pudo conectar con MLflow")
        return
    
    # Obtener modelos registrados
    models = get_registered_models(client)
    if not models:
        st.error("No hay modelos registrados disponibles")
        return
    
    # Selector de modelo
    st.write("### Selección de Modelo")
    model_options = {model[0]: model for model in models}
    selected_model_name = st.selectbox(
        "Seleccione un modelo registrado",
        options=list(model_options.keys()),
        format_func=lambda x: x.replace("dialer_ratio_predictor_", "")
    )
    
    selected_model = model_options[selected_model_name]
    version = selected_model[1]
    
    # Mostrar información del modelo
    st.write("### Métricas de Desempeño del Modelo")
    
    # Explicación general
    st.markdown("""
    Las siguientes métricas indican qué tan preciso es el modelo al predecir el ratio óptimo de marcación:
    """)
    
    metrics, params = get_model_metrics(version.run_id)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Error Absoluto Medio (MAE)", 
            f"{metrics.get('mae', 0):.3f}",
            help="Representa el error promedio en el ratio de marcación predicho. "
                 "Por ejemplo, un MAE de 0.5 significa que, en promedio, las predicciones "
                 "del ratio de marcación difieren en ±0.5 llamadas por agente del valor óptimo real."
        )
    with col2:
        st.metric(
            "Coeficiente de Determinación (R²)", 
            f"{metrics.get('r2', 0):.3f}",
            help="Indica qué porcentaje de la variabilidad en el ratio óptimo de marcación "
                 "es explicado por el modelo. Un R² de 0.8 significa que el modelo explica "
                 "el 80% de la variación en los ratios óptimos observados históricamente."
        )
    with col3:
        st.metric(
            "Error Cuadrático Medio (RMSE)", 
            f"{metrics.get('mse', 0)**0.5:.3f}",
            help="Similar al MAE, pero penaliza más los errores grandes. Un RMSE de 1.0 "
                 "sugiere que pueden existir algunos casos donde la predicción del ratio "
                 "se desvía significativamente del óptimo."
        )
    
    # Interpretación adicional
    st.markdown("""
    #### Interpretación de las Métricas
    - **MAE bajo** (cercano a 0): Las predicciones del ratio son consistentemente cercanas al óptimo.
    - **R² alto** (cercano a 1): El modelo captura bien los patrones que determinan el ratio óptimo.
    - **RMSE cercano al MAE**: Las predicciones tienen errores consistentes sin grandes desviaciones.
    
    > 💡 En el contexto de ratios de marcación, errores menores a 1.0 son generalmente aceptables, 
    ya que representan menos de una llamada de diferencia por agente disponible.
    """)
    
    # Formulario para predicción
    st.write("### Variables de Entrada")
    st.write("Ingrese los valores para obtener una predicción:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Usar session state para mantener los valores
            if 'agents_connected' not in st.session_state:
                st.session_state.agents_connected = 10
            if 'agents_available' not in st.session_state:
                st.session_state.agents_available = 5

            def update_agents_connected():
                st.session_state.agents_available = min(
                    st.session_state.agents_available,
                    st.session_state.agents_connected
                )

            def update_agents_available():
                if st.session_state.agents_available > st.session_state.agents_connected:
                    st.session_state.agents_available = st.session_state.agents_connected

            agents_connected = st.number_input(
                "Agentes Conectados",
                min_value=0,
                value=st.session_state.agents_connected,
                on_change=update_agents_connected,
                key='agents_connected',
                help="Número total de agentes conectados"
            )
            
            agents_available = st.number_input(
                "Agentes Disponibles",
                min_value=0,
                max_value=st.session_state.agents_connected,
                value=st.session_state.agents_available,
                on_change=update_agents_available,
                key='agents_available',
                help="Número de agentes disponibles para atender llamadas"
            )
            
            avg_call_duration = st.number_input(
                "Duración Promedio de Llamada (seg)",
                min_value=0,
                value=180,
                help="Duración promedio de las llamadas en segundos"
            )
            
            hour = st.slider(
                "Hora del Día",
                min_value=0,
                max_value=23,
                value=9,
                help="Hora del día (0-23)"
            )
        
        with col2:
            day_of_week = st.selectbox(
                "Día de la Semana",
                options=[(0, "Lunes"), (1, "Martes"), (2, "Miércoles"),
                        (3, "Jueves"), (4, "Viernes"), (5, "Sábado")],
                format_func=lambda x: x[1]
            )[0]
        
        submit = st.form_submit_button("Calcular Predicción")
    
    if submit:
        try:
            # Determinar parte del día
            part_of_day = -1
            if day_of_week <= 4:  # Lunes a viernes
                if 7 <= hour < 11:
                    part_of_day = 0
                elif 11 <= hour < 14:
                    part_of_day = 1
                elif 14 <= hour < 19:
                    part_of_day = 2
            elif day_of_week == 5:  # Sábado
                if 8 <= hour < 11:
                    part_of_day = 0
                elif 11 <= hour < 15:
                    part_of_day = 1
            
            # Crear DataFrame con features
            features = pd.DataFrame({
                'agents_connected': [agents_connected],
                'agents_available': [agents_available],
                'hour': [hour],
                'day_of_week': [day_of_week],
                'part_of_day': [part_of_day],
                'avg_call_duration': [avg_call_duration]
            })
            
            # Cargar modelo
            model = mlflow.xgboost.load_model(f"runs:/{version.run_id}/model")
            
            # Realizar predicción
            prediction, lower_bound, upper_bound = make_prediction(
                model,
                features,
                metrics.get('mae', 0)
            )
            
            # Mostrar resultados
            st.write("### Resultados de la Predicción")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Ratio Óptimo de Marcación",
                    f"{prediction:.2f}",
                    help="Número recomendado de llamadas por agente disponible"
                )
            
            with col2:
                st.metric(
                    "Límite Inferior",
                    f"{lower_bound:.2f}",
                    help="Límite inferior del intervalo de confianza"
                )
            
            with col3:
                st.metric(
                    "Límite Superior",
                    f"{upper_bound:.2f}",
                    help="Límite superior del intervalo de confianza"
                )
            
            st.info(
                f"Con un Error Absoluto Medio de {metrics.get('mae', 0):.3f}, " 
                f"el ratio de marcación óptimo debería estar entre {lower_bound:.2f} "
                f"y {upper_bound:.2f} llamadas por agente disponible."
            )
            
        except Exception as e:
            st.error(f"Error al realizar la predicción: {str(e)}")

if __name__ == "__main__":
    main()