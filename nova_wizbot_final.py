import streamlit as st
import time
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp

# Configuración de la página
st.set_page_config(
    page_title="Bienvenid@ a NOVA Wizbot",
    page_icon="🤖",
    layout="wide",
)

# Semáforo global para controlar concurrencia
SEMAPHORE = asyncio.Semaphore(2)  # Máximo 2 solicitudes concurrentes

# Estilos CSS personalizados
st.markdown("""
<style>
    /* Layout principal */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 80px; /* Reducido para dar espacio al chat_input */
        max-width: 100%;
    }
    
    /* Título más compacto */
    .main h1 {
        margin-bottom: 1rem !important;
        padding-bottom: 0 !important;
    }
    
    /* Botones personalizados */
    .stButton button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    
    .stButton button:hover {
        background-color: #45a049;
    }
    
    /* Estilo para mensajes de usuario y bot */
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .message-header {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    /* Pie de página */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #4caf50;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-weight: bold;
        font-size: 14px;
        z-index: 999;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tooltips adaptativos que funcionan en modo claro y oscuro */
    .tooltip-phi4 {
        background-color: #f0f9ff; /* Fondo sólido azul muy claro */
        border: 1px solid #0ea5e9;
        color: #000000; /* Negro sólido para modo claro */
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-top: 5px;
        font-weight: 500;
    }
    
    .tooltip-phi4-mini {
        background-color: #fffbeb; /* Fondo sólido amarillo muy claro */
        border: 1px solid #f59e0b;
        color: #000000; /* Negro sólido para modo claro */
        padding: 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Modo oscuro - fondo más oscuro y texto blanco */
    @media (prefers-color-scheme: dark) {
        .tooltip-phi4 {
            background-color: #1e3a8a !important; /* Azul oscuro sólido */
            border: 1px solid #3b82f6 !important;
            color: #ffffff !important; /* Blanco sólido */
        }
        
        .tooltip-phi4-mini {
            background-color: #92400e !important; /* Naranja oscuro sólido */
            border: 1px solid #f59e0b !important;
            color: #ffffff !important; /* Blanco sólido */
        }
    }
    
    /* Ocultar elementos innecesarios */
    .stDeployButton {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Inicialización del estado de la sesión
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'model' not in st.session_state:
    st.session_state.model = "phi4:14b"  # Default model

if 'models_list' not in st.session_state:
    st.session_state.models_list = []

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7

if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.9

if 'top_k' not in st.session_state:
    st.session_state.top_k = 40

if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 5000

# URL del endpoint de Ollama
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODELS_ENDPOINT = "http://127.0.0.1:11434/api/tags"

# Función asíncrona para obtener la lista de modelos disponibles
async def get_available_models_async():
    async with SEMAPHORE:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(OLLAMA_MODELS_ENDPOINT) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        return [model['name'] for model in models_data.get('models', [])]
                    else:
                        return ["Error al cargar modelos"]
        except Exception as e:
            return ["Error de conexión"]

# Wrapper síncrono para obtener modelos
def get_available_models():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(get_available_models_async())
    except Exception as e:
        return ["Error de conexión"]

# Función asíncrona para chat con Ollama
async def chat_with_ollama_async(message, model, history, temperature, top_p, top_k, max_tokens):
    async with SEMAPHORE:
        try:
            # Preparar el historial en formato esperado por Ollama
            messages = []
            for entry in history:
                if entry["role"] == "user":
                    messages.append({"role": "user", "content": entry["content"]})
                else:
                    messages.append({"role": "assistant", "content": entry["content"]})
            
            # Añadir el mensaje actual
            messages.append({"role": "user", "content": message})
            
            # Crear la solicitud a la API
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": max_tokens
                }
            }
            
            # Medir el tiempo de respuesta
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.post(OLLAMA_ENDPOINT, json=payload) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        response_data = await response.json()
                        return response_data['message']['content'], response_time
                    else:
                        return f"Error: {response.status}", response_time
                        
        except Exception as e:
            return f"Error de conexión: {str(e)}", 0

# Wrapper síncrono para chat
def chat_with_ollama(message, model, history, temperature, top_p, top_k, max_tokens):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            chat_with_ollama_async(message, model, history, temperature, top_p, top_k, max_tokens)
        )
    except Exception as e:
        return f"Error: {str(e)}", 0

# Auto-cargar modelos al inicio
if not st.session_state.models_loaded:
    with st.spinner("🔄 Cargando modelos disponibles..."):
        st.session_state.models_list = get_available_models()
        st.session_state.models_loaded = True

# Interfaz de usuario
st.title("💬 Pregúntale a NOVA Wizbot!")

# Sidebar para configuración
with st.sidebar:
    
    logo_url = "nova_wizbot.png"
    st.image(logo_url, width=250)
    
    st.subheader("🤖 Selección de Modelo")
    
    # Radio buttons para modelos principales con tooltips
    model_choice = st.radio(
        "Modelo principal:",
        ["phi4:14b", "phi4-mini:3.8b"],
        index=0,  # phi4:14b por defecto
        key="main_model_choice"
    )
    
    # Tooltips informativos con clases CSS adaptativas
    if model_choice == "phi4:14b":
        st.markdown("""
        <div class="tooltip-phi4">
        🔹 <b>phi4:14b</b>: Modelo completo de 14B parámetros. Mayor precisión y capacidad de razonamiento.
        Ideal para tareas complejas y conversaciones detalladas.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="tooltip-phi4-mini">
        ⚡ <b>phi4-mini:3.8b</b>: Modelo ligero de 3.8B parámetros. Respuestas más rápidas con menor
        consumo de recursos. Ideal para consultas rápidas y conversaciones casuales.
        </div>
        """, unsafe_allow_html=True)
    
    st.session_state.model = model_choice
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas", expanded=False):
        # Dropdown con todos los modelos disponibles
        if st.session_state.models_list and len(st.session_state.models_list) > 2:
            advanced_model = st.selectbox(
                "Modelo alternativo:",
                ["Usar modelo principal"] + st.session_state.models_list,
                index=0
            )
            
            if advanced_model != "Usar modelo principal":
                st.session_state.model = advanced_model
        
        # Parámetros de generación
        st.subheader("Parámetros de Generación")
        
        st.session_state.temperature = st.slider(
            "🌡️ Temperatura", 
            min_value=0.1, 
            max_value=1.0, 
            value=st.session_state.temperature, 
            step=0.1,
            help="Controla la creatividad. Valores bajos = más conservador, valores altos = más creativo"
        )
        
        st.session_state.top_p = st.slider(
            "🎯 Top P", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.top_p, 
            step=0.05,
            help="Controla la diversidad de vocabulario. Valores menores = más enfocado"
        )
        
        st.session_state.top_k = st.slider(
            "🔢 Top K", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.top_k, 
            step=1,
            help="Limita las opciones de palabras consideradas"
        )
        
        st.session_state.max_tokens = st.slider(
            "📏 Máximo tokens", 
            min_value=100, 
            max_value=8000, 
            value=st.session_state.max_tokens, 
            step=100,
            help="Longitud máxima de la respuesta"
        )
    
    st.markdown("---")
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        # Mover el botón de exportar aquí para que aparezca siempre
        if st.button("📥 Descargar historial", use_container_width=True):
            if st.session_state.chat_history:
                # Crear un dataframe con el historial completo
                df = pd.DataFrame(st.session_state.chat_history)
                
                # Crear el nombre del archivo con la fecha y hora actual
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.csv"
                
                # Convertir a CSV
                csv = df.to_csv(index=False)
                
                # Crear un enlace de descarga
                st.download_button(
                    label="💾 Confirmar descarga",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                    key="download_csv"
                )
            else:
                st.warning("No hay conversación para exportar.")

# Mostrar conversación usando st.chat_message
if not st.session_state.chat_history:
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: #666;">
        <h3>👋 ¡Hola! Soy tu asistente de IA</h3>
        <p>Envía un mensaje para comenzar.</p>
        <p><strong>Modelo activo:</strong> {st.session_state.model}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                # Para el asistente, mostrar el contenido y la información adicional
                st.write(message["content"])
                st.caption(f"⏱️ Tiempo de respuesta: {message.get('response_time', 'N/A')} | 🕐 {message['timestamp']}")

# Chat input fijo en la parte inferior
if user_input := st.chat_input("Escribe tu mensaje aquí..."):
    # Agregar mensaje del usuario al historial
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner(f"🤔 Generando respuesta con {st.session_state.model}..."):
            response, response_time = chat_with_ollama(
                user_input, 
                st.session_state.model, 
                st.session_state.chat_history,
                st.session_state.temperature,
                st.session_state.top_p,
                st.session_state.top_k,
                st.session_state.max_tokens
            )
            
            # Mostrar respuesta
            st.write(response)
            st.caption(f"⏱️ Tiempo de respuesta: {response_time:.2f} segundos | 🕐 {datetime.now().strftime('%H:%M:%S')}")
            
            # Agregar respuesta al historial
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "response_time": f"{response_time:.2f} segundos"
            })

# Pie de página
st.markdown("""
<div class="footer">
    AECSA BI - 2025
</div>
""", unsafe_allow_html=True)