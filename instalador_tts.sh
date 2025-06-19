#!/bin/bash

# Instalador completo del sistema de síntesis de voz TTS con Fish Speech
# Versión 2.0 - Sistema de producción para Ubuntu Server
# Crea usuario tts_service y configura todo el entorno

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con formato
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar si se está ejecutando como root
if [ "$EUID" -ne 0 ]; then
    print_error "Este script debe ejecutarse como root (sudo)"
    exit 1
fi

print_message "=== Instalador TTS v2.0 - Sistema de Síntesis de Voz ==="
print_message "Configurando sistema completo para usuario tts_service"

# Crear usuario tts_service si no existe
print_message "Configurando usuario tts_service..."
if id "tts_service" &>/dev/null; then
    print_warning "Usuario tts_service ya existe"
else
    useradd -m -s /bin/bash tts_service
    usermod -aG sudo tts_service
    print_success "Usuario tts_service creado"
fi

# Definir directorio base
TTS_HOME="/home/tts_service"
print_message "Directorio base: $TTS_HOME"

# Verificar la instalación de Python 3.10
print_message "Verificando la instalación de Python 3.10..."
if command -v python3.10 >/dev/null 2>&1; then
    print_success "Python 3.10 ya está instalado"
else
    print_message "Instalando Python 3.10..."
    apt update
    apt install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils python3.10-pip
    print_success "Python 3.10 instalado correctamente"
fi

# Verificar si los drivers NVIDIA están instalados
print_message "Verificando drivers NVIDIA..."
if command -v nvidia-smi >/dev/null 2>&1; then
    print_success "Los drivers NVIDIA ya están instalados"
    nvidia-smi | head -3
else
    print_message "Instalando drivers NVIDIA Open 570..."
    apt update
    apt install -y software-properties-common
    add-apt-repository ppa:graphics-drivers/ppa -y
    apt update
    
    # Instalar drivers Open
    apt install -y nvidia-driver-570-open
    
    print_success "Drivers NVIDIA instalados correctamente"
    print_warning "Es posible que sea necesario reiniciar el sistema para activar los drivers"
fi

# Verificar si CUDA está instalado
print_message "Verificando instalación de CUDA..."
if [ -d "/usr/local/cuda" ]; then
    print_success "CUDA ya está instalado"
else
    print_message "Instalando CUDA 12.8..."
    
    # Agregar repositorio de CUDA
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    rm cuda-keyring_1.0-1_all.deb
    apt update
    
    # Instalar CUDA Toolkit
    apt install -y cuda-toolkit-12-8
    
    print_success "CUDA 12.8 instalado correctamente"
    
    # Configurar variables de entorno para CUDA
    echo 'export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh
    chmod +x /etc/profile.d/cuda.sh
    source /etc/profile.d/cuda.sh
    
    print_message "Variables de entorno para CUDA configuradas"
fi

# Instalar dependencias del sistema
print_message "Instalando dependencias del sistema..."
apt install -y build-essential cmake git \
               libsox-dev ffmpeg \
               portaudio19-dev libportaudio2 libportaudiocpp0 \
               librubberband-dev \
               pkg-config curl wget
print_success "Dependencias del sistema instaladas correctamente"

# Cambiar al usuario tts_service para las operaciones de usuario
print_message "Configurando entorno para usuario tts_service..."

# Crear estructura de directorios
sudo -u tts_service mkdir -p "$TTS_HOME"/{scripts,samples,config,IN,OUT,logs}

# Clonar el repositorio de Fish Speech
print_message "Clonando el repositorio de Fish Speech..."
cd "$TTS_HOME"
if [ -d "fish-speech" ]; then
    print_warning "El directorio fish-speech ya existe. Omitiendo clonación."
else
    sudo -u tts_service git clone https://github.com/fishaudio/fish-speech.git
    print_success "Repositorio de Fish Speech clonado correctamente"
fi

# Crear y activar el entorno virtual
print_message "Configurando el entorno virtual Python..."
cd "$TTS_HOME"
if [ -d "ambiente_tts" ]; then
    print_warning "El entorno virtual ya existe"
else
    sudo -u tts_service python3.10 -m venv ambiente_tts
    print_success "Entorno virtual ambiente_tts creado"
fi

# Activar entorno virtual y instalar dependencias
print_message "Instalando dependencias de Python..."
sudo -u tts_service bash -c "
    source $TTS_HOME/ambiente_tts/bin/activate
    
    # Actualizar pip
    python -m pip install --upgrade pip setuptools wheel
    
    # Instalar dependencias de fish-speech
    cd $TTS_HOME/fish-speech
    pip install -e .[stable]
    
    # Instalar versiones específicas de PyTorch con CUDA
    pip uninstall -y torch torchaudio
    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    
    # Instalar dependencias adicionales para v2.0
    pip install watchdog pyrubberband soundfile
"

print_success "Dependencias de Python instaladas correctamente"

# Descargar los modelos preentrenados
print_message "Descargando modelos preentrenados (esto puede tardar varios minutos)..."
sudo -u tts_service bash -c "
    source $TTS_HOME/ambiente_tts/bin/activate
    cd $TTS_HOME/fish-speech
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
"
print_success "Modelos preentrenados descargados correctamente"

# Crear archivo de configuración
print_message "Creando archivo de configuración..."
sudo -u tts_service cat > "$TTS_HOME/config/config.json" << 'EOF'
{
    "model_path": "/home/tts_service/fish-speech/checkpoints/fish-speech-1.5",
    "samples_dir": "/home/tts_service/samples",
    "outputs_dir": "/home/tts_service/OUT",
    "inputs_dir": "/home/tts_service/IN",
    "logs_dir": "/home/tts_service/logs",
    "sample_rate": 44100,
    "device": "cuda",
    
    "synthesis_config": {
        "voz": "mi_voz",
        "velocidad": 1.0,
        "chunk_length": 150,
        "streaming": false,
        
        "parametros_avanzados": {
            "temperature": 0.7,
            "top_p": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 1024,
            "seed": null,
            "use_memory_cache": "on"
        },
        
        "calidad_audio": {
            "formato": "wav",
            "normalizacion": true
        }
    },
    
    "processing_config": {
        "scan_interval_seconds": 5,
        "debounce_delay_seconds": 2,
        "max_retries": 3,
        "enable_file_watcher": true,
        "process_on_startup": true
    },
    
    "json_text_fields": [
        "text",
        "texto", 
        "content",
        "contenido",
        "message",
        "mensaje",
        "description",
        "descripcion"
    ],
    
    "logging_config": {
        "level": "INFO",
        "max_log_files": 10,
        "log_rotation_days": 7
    }
}
EOF

print_success "Archivo de configuración creado"

# Crear el script batch_service.py v2.0
print_message "Creando batch_service.py v2.0..."
sudo -u tts_service cat > "$TTS_HOME/scripts/batch_service.py" << 'EOF'
#!/usr/bin/env python3
"""
Servicio de Síntesis de Voz por Lotes - VERSIÓN 2.0
Sistema de producción con watchdog, múltiples textos y estadísticas completas

Ubicación: /home/tts_service/scripts/batch_service.py
"""

import os
import sys
import json
import time
import gc
import logging
import hashlib
import csv
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import torchaudio
import numpy as np

# Importar watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ Watchdog no disponible - usando solo polling")

# Agregar Fish Speech al path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "fish-speech"))

def setup_logging(logs_dir):
    """Configurar sistema de logging completo"""
    os.makedirs(logs_dir, exist_ok=True)
    
    log_filename = f"batch_synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def load_config():
    """Cargar configuración desde config.json"""
    config_path = project_root / "config" / "config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validar estructura de configuración
        required_keys = ['model_path', 'inputs_dir', 'outputs_dir', 'samples_dir', 'logs_dir']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Falta la clave requerida '{key}' en config.json")
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de configuración en {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error al parsear config.json: {e}")

# Cargar configuración global
config = load_config()
logger = setup_logging(config['logs_dir'])

# Importar módulos de Fish Speech
try:
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.vqgan.inference import load_model as load_decoder_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
    from fish_speech.text.spliter import split_text
    logger.info("✅ Módulos de Fish Speech importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error al importar Fish Speech: {e}")
    sys.exit(1)

class StatisticsManager:
    """Gestor de estadísticas CSV"""
    
    def __init__(self, logs_dir):
        self.csv_path = os.path.join(logs_dir, "synthesis_stats.csv")
        self._ensure_csv_exists()
    
    def _ensure_csv_exists(self):
        """Crear archivo CSV con headers si no existe"""
        if not os.path.exists(self.csv_path):
            headers = [
                "timestamp",
                "archivo_origen",
                "texto_key",
                "archivo_audio",
                "texto_sintetizado",
                "voz_usada",
                "velocidad",
                "duracion_audio_segundos",
                "tiempo_sintesis_segundos",
                "factor_tiempo_real",
                "caracteres_texto",
                "size_audio_bytes",
                "metodo_ajuste_velocidad",
                "temperature",
                "top_p",
                "repetition_penalty",
                "max_new_tokens",
                "chunk_length",
                "intentos_realizados",
                "seed_utilizada"
            ]
            
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            logger.info(f"📊 Archivo de estadísticas creado: {self.csv_path}")
    
    def log_synthesis(self, stats_data):
        """Registrar estadísticas de síntesis en CSV"""
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Escapar texto sintetizado para CSV
                texto_escaped = f'"{stats_data["texto_sintetizado"].replace('"', '""')}"'
                
                row = [
                    datetime.now().isoformat(),
                    stats_data["archivo_origen"],
                    stats_data["texto_key"],
                    stats_data["archivo_audio"],
                    texto_escaped,
                    stats_data["voz_usada"],
                    stats_data["velocidad"],
                    stats_data["duracion_audio"],
                    stats_data["tiempo_sintesis"],
                    stats_data["factor_tiempo_real"],
                    stats_data["caracteres_texto"],
                    stats_data["size_audio_bytes"],
                    stats_data["metodo_ajuste_velocidad"],
                    stats_data["temperature"],
                    stats_data["top_p"],
                    stats_data["repetition_penalty"],
                    stats_data["max_new_tokens"],
                    stats_data["chunk_length"],
                    stats_data["intentos_realizados"],
                    stats_data["seed_utilizada"]
                ]
                
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Error escribiendo estadísticas CSV: {e}")

class VoiceSynthesizer:
    """Clase para manejar la síntesis de voz v2.0"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.synthesis_config = config.get('synthesis_config', {})
        self.llama_queue = None
        self.decoder_model = None
        self.inference_engine = None
        self.reference_cache = {}
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Inicializando VoiceSynthesizer en dispositivo: {self.device}")
    
    def configure_optimizations(self):
        """Configurar optimizaciones de GPU"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            capabilities = torch.cuda.get_device_capability()
            logger.info(f"GPU detectada: {gpu_name} (CUDA {capabilities[0]}.{capabilities[1]})")
            
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        else:
            logger.warning("⚠️ No se detectó GPU con CUDA")
        
        torch.use_deterministic_algorithms(False)
    
    def initialize_models(self, compile_models=True):
        """Inicializar los modelos de síntesis"""
        logger.info("🚀 Inicializando modelos de síntesis de voz...")
        start_time = time.time()
        
        precision = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Inicializar modelo text2semantic
        try:
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.config["model_path"],
                device=self.device,
                precision=precision,
                compile=compile_models
            )
        except Exception as e:
            logger.warning(f"Error al compilar modelo: {e}")
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.config["model_path"],
                device=self.device,
                precision=precision,
                compile=False
            )
        
        # Cargar modelo VQGAN
        decoder_path = os.path.join(self.config["model_path"], "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
        self.decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=decoder_path,
            device=self.device
        )
        
        # Inicializar motor de inferencia
        self.inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=precision,
            compile=compile_models
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Modelos inicializados en {elapsed_time:.2f} segundos")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_available_voices(self) -> List[str]:
        """Obtener lista de voces disponibles"""
        samples_dir = self.config["samples_dir"]
        
        if not os.path.exists(samples_dir):
            return []
        
        voices = []
        for file in os.listdir(samples_dir):
            if file.endswith('.wav'):
                voice_name = os.path.splitext(file)[0]
                voices.append(voice_name)
        
        return voices
    
    def load_reference_audio(self, voice_name: str) -> bytes:
        """Cargar audio de referencia para una voz"""
        if voice_name in self.reference_cache:
            return self.reference_cache[voice_name]
        
        sample_path = os.path.join(self.config["samples_dir"], f"{voice_name}.wav")
        
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Archivo de audio no encontrado: {sample_path}")
        
        with open(sample_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        self.reference_cache[voice_name] = audio_bytes
        return audio_bytes
    
    def load_reference_text(self, voice_name: str) -> str:
        """Cargar texto de referencia para una voz"""
        txt_path = os.path.join(self.config["samples_dir"], f"{voice_name}.txt")
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Error leyendo texto de referencia para {voice_name}: {e}")
        
        return f"Audio de referencia para {voice_name}"
    
    def apply_speed_adjustment(self, audio_data: np.ndarray, sample_rate: int, speed_factor: float) -> tuple[np.ndarray, str]:
        """Aplicar ajuste de velocidad sin distorsión usando jerarquía de métodos"""
        if speed_factor == 1.0:
            return audio_data, "sin_ajuste"
        
        # Método 1: pyrubberband (profesional)
        try:
            import pyrubberband as pyrb
            logger.debug(f"Aplicando ajuste con pyrubberband: {speed_factor}")
            
            # Intentar con corrección de formantes
            try:
                audio_processed = pyrb.time_stretch(
                    audio_data, 
                    sample_rate, 
                    speed_factor,
                    rbargs={'--formant-corrected': ''}
                )
                return audio_processed, "pyrubberband_formants"
            except:
                # Fallback a pyrubberband básico
                audio_processed = pyrb.time_stretch(audio_data, sample_rate, speed_factor)
                return audio_processed, "pyrubberband_basic"
                
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error con pyrubberband: {e}")
        
        # Método 2: WSOLA (académico)
        try:
            from scipy import signal
            logger.debug(f"Aplicando ajuste con WSOLA: {speed_factor}")
            
            # Parámetros adaptativos
            frame_length = int(sample_rate * 0.032)
            hop_length = int(frame_length * 0.55)
            window = signal.windows.hann(frame_length)
            
            synthesis_hop = int(hop_length / speed_factor)
            output_length = int(len(audio_data) / speed_factor)
            output_audio = np.zeros(output_length + frame_length)
            
            analysis_pos = 0
            synthesis_pos = 0
            
            while analysis_pos + frame_length < len(audio_data):
                frame = audio_data[analysis_pos:analysis_pos + frame_length] * window
                
                if synthesis_pos > 0:
                    search_range = min(frame_length // 4, synthesis_pos)
                    best_offset = 0
                    best_correlation = -np.inf
                    
                    for offset in range(-search_range, search_range + 1):
                        pos = synthesis_pos + offset
                        if pos >= 0 and pos + frame_length < len(output_audio):
                            existing = output_audio[pos:pos + frame_length]
                            correlation = np.correlate(frame, existing, mode='valid')[0]
                            if correlation > best_correlation:
                                best_correlation = correlation
                                best_offset = offset
                    
                    synthesis_pos += best_offset
                
                if synthesis_pos + frame_length < len(output_audio):
                    output_audio[synthesis_pos:synthesis_pos + frame_length] += frame
                
                analysis_pos += hop_length
                synthesis_pos += synthesis_hop
            
            output_audio = output_audio[:output_length]
            return output_audio.astype(np.float32), "wsola"
            
        except Exception as e:
            logger.warning(f"Error con WSOLA: {e}")
        
        # Método 3: Resampling inteligente (último recurso)
        try:
            from scipy.signal import resample_poly
            logger.debug(f"Aplicando resampling inteligente: {speed_factor}")
            
            if speed_factor >= 1.0:
                up_factor = 1000
                down_factor = int(1000 * speed_factor)
            else:
                up_factor = int(1000 / speed_factor)
                down_factor = 1000
            
            audio_processed = resample_poly(audio_data, up_factor, down_factor)
            
            # Ajustar longitud exacta
            target_length = int(len(audio_data) / speed_factor)
            if len(audio_processed) != target_length:
                from scipy import interpolate
                f = interpolate.interp1d(
                    np.linspace(0, 1, len(audio_processed)), 
                    audio_processed, 
                    kind='cubic'
                )
                audio_processed = f(np.linspace(0, 1, target_length))
            
            return audio_processed.astype(np.float32), "resampling"
            
        except Exception as e:
            logger.error(f"Error con resampling: {e}")
            return audio_data, "sin_ajuste"
    
    def synthesize_voice(self, text: str, voice_name: str) -> tuple[np.ndarray, int, Dict[str, Any]]:
        """Sintetizar voz para un texto dado"""
        start_time = time.time()
        
        # Cargar referencias
        audio_bytes = self.load_reference_audio(voice_name)
        prompt_text = self.load_reference_text(voice_name)
        
        references = [ServeReferenceAudio(audio=audio_bytes, text=prompt_text)]
        
        # Configurar parámetros
        params = self.synthesis_config.get('parametros_avanzados', {})
        
        tts_request = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            max_new_tokens=params.get('max_new_tokens', 1024),
            chunk_length=self.synthesis_config.get('chunk_length', 150),
            top_p=params.get('top_p', 0.7),
            repetition_penalty=params.get('repetition_penalty', 1.2),
            temperature=params.get('temperature', 0.7),
            streaming=False,
            format='wav',
            use_memory_cache=params.get('use_memory_cache', 'on'),
            seed=params.get('seed', None)
        )
        
        # Realizar síntesis con reintentos
        max_retries = self.config.get('processing_config', {}).get('max_retries', 3)
        
        for attempt in range(max_retries):
            try:
                results = list(self.inference_engine.inference(tts_request))
                
                audio_segments = []
                sample_rate = None
                
                for result in results:
                    if result.code == "error":
                        if attempt < max_retries - 1:
                            break
                        else:
                            raise Exception(str(result.error))
                    elif result.code in ["final", "segment"]:
                        sample_rate, audio_data = result.audio
                        audio_segments.append(audio_data)
                
                if audio_segments:
                    break
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
        
        if not audio_segments:
            raise Exception("No se generó audio después de todos los intentos")
        
        # Combinar segmentos
        audio_combined = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
        
        # Normalización básica
        if audio_combined.max() != 0:
            audio_combined = audio_combined / np.max(np.abs(audio_combined)) * 0.9
        
        # Ajustar velocidad si es necesario
        speed_factor = self.synthesis_config.get('velocidad', 1.0)
        metodo_ajuste = "sin_ajuste"
        
        if speed_factor != 1.0:
            audio_combined, metodo_ajuste = self.apply_speed_adjustment(audio_combined, sample_rate, speed_factor)
        
        # Calcular métricas
        synthesis_time = time.time() - start_time
        audio_duration = len(audio_combined) / sample_rate
        
        metrics = {
            "synthesis_time": synthesis_time,
            "audio_duration": audio_duration,
            "real_time_factor": audio_duration / synthesis_time,
            "text_length": len(text),
            "voice_used": voice_name,
            "speed_factor": speed_factor,
            "attempts": attempt + 1,
            "sample_rate": int(sample_rate),
            "metodo_ajuste_velocidad": metodo_ajuste,
            "temperature": params.get('temperature', 0.7),
            "top_p": params.get('top_p', 0.7),
            "repetition_penalty": params.get('repetition_penalty', 1.2),
            "max_new_tokens": params.get('max_new_tokens', 1024),
            "chunk_length": self.synthesis_config.get('chunk_length', 150),
            "seed_utilizada": params.get('seed', None)
        }
        
        return audio_combined.astype(np.float32), int(sample_rate), metrics

class FileProcessor:
    """Procesador de archivos con soporte para múltiples textos"""
    
    def __init__(self, config: Dict[str, Any], synthesizer: VoiceSynthesizer):
        self.config = config
        self.synthesizer = synthesizer
        self.processed_files = {}  # Cambio: dict para rastrear textos procesados por archivo
        self.processed_files_path = os.path.join(config['logs_dir'], 'processed_files.json')
        self.stats_manager = StatisticsManager(config['logs_dir'])
        self.text_fields = config.get('json_text_fields', ['text', 'texto', 'content', 'contenido'])
        
        self._load_processed_files()
    
    def _load_processed_files(self):
        """Cargar estado de archivos procesados"""
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                    self.processed_files = json.load(f)
                logger.info(f"Cargado estado de {len(self.processed_files)} archivos")
            except Exception as e:
                logger.warning(f"Error cargando estado: {e}")
                self.processed_files = {}
    
    def _save_processed_files(self):
        """Guardar estado de archivos procesados"""
        try:
            with open(self.processed_files_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando estado: {e}")
    
    def _get_file_hash(self, filepath: str) -> str:
        """Obtener hash del contenido del archivo"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def extract_multiple_texts(self, content: str) -> Dict[str, str]:
        """Extraer múltiples textos de contenido JSON"""
        try:
            data = json.loads(content)
            texts_found = {}
            
            # Buscar todos los campos de texto posibles
            for field in self.text_fields:
                # Buscar campo base
                if field in data and isinstance(data[field], str):
                    text = data[field].strip()
                    if len(text) > 5:
                        texts_found[field] = text
                
                # Buscar campos numerados (text_1, text_2, etc.)
                counter = 1
                while True:
                    numbered_field = f"{field}_{counter}"
                    if numbered_field in data and isinstance(data[numbered_field], str):
                        text = data[numbered_field].strip()
                        if len(text) > 5:
                            texts_found[numbered_field] = text
                        counter += 1
                    else:
                        break
            
            return texts_found
            
        except json.JSONDecodeError:
            logger.error("Error parsing JSON")
            return {}
        except Exception as e:
            logger.error(f"Error extrayendo textos: {e}")
            return {}
    
    def save_audio_wav(self, audio_data: np.ndarray, sample_rate: int, output_path: str) -> int:
        """Guardar audio como archivo WAV y retornar tamaño"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convertir a tensor para torchaudio
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            torchaudio.save(output_path, audio_tensor, sample_rate)
            
            file_size = os.path.getsize(output_path)
            return file_size
            
        except Exception as e:
            raise Exception(f"Error guardando audio: {e}")
    
    def process_file(self, input_filepath: str) -> bool:
        """Procesar un archivo individual con soporte para múltiples textos"""
        filename = os.path.basename(input_filepath)
        
        try:
            # Leer contenido del archivo
            with open(input_filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"Archivo vacío: {filename}")
                return False
            
            # Obtener hash del archivo
            file_hash = self._get_file_hash(input_filepath)
            
            # Verificar si el archivo cambió
            if filename in self.processed_files:
                if self.processed_files[filename].get('hash') == file_hash:
                    logger.debug(f"⏭️ Archivo sin cambios: {filename}")
                    return True
            
            # Extraer todos los textos posibles
            texts_found = self.extract_multiple_texts(content)
            
            if not texts_found:
                logger.error(f"No se encontraron textos válidos en {filename}")
                return False
            
            # Determinar voz a usar
            voice_name = self.config['synthesis_config']['voz']
            if not voice_name or voice_name == "mi_voz":
                voices = self.synthesizer.get_available_voices()
                if not voices:
                    logger.error("No hay voces disponibles")
                    return False
                voice_name = voices[0]
                logger.info(f"Usando voz por defecto: {voice_name}")
            
            # Verificar qué textos son nuevos
            processed_texts = self.processed_files.get(filename, {}).get('texts_processed', {})
            texts_to_process = {}
            
            for text_key, text_content in texts_found.items():
                text_hash = hashlib.md5(text_content.encode()).hexdigest()
                if processed_texts.get(text_key) != text_hash:
                    texts_to_process[text_key] = text_content
            
            if not texts_to_process:
                logger.debug(f"⏭️ Todos los textos ya procesados en {filename}")
                return True
            
            logger.info(f"📄 Procesando {filename}: {len(texts_to_process)} textos nuevos")
            
            # Procesar cada texto nuevo
            successfully_processed = {}
            
            for text_key, text_content in texts_to_process.items():
                try:
                    logger.info(f"🎤 Sintetizando {text_key}: '{text_content[:50]}...'")
                    
                    # Sintetizar voz
                    audio_data, sample_rate, metrics = self.synthesizer.synthesize_voice(text_content, voice_name)
                    
                    # Crear nombre de archivo de salida
                    base_name = os.path.splitext(filename)[0]
                    if text_key == "text":
                        output_filename = f"{base_name}.wav"
                    else:
                        output_filename = f"{base_name}_{text_key}.wav"
                    
                    output_path = os.path.join(self.config['outputs_dir'], output_filename)
                    
                    # Guardar audio
                    file_size = self.save_audio_wav(audio_data, sample_rate, output_path)
                    
                    # Registrar estadísticas en CSV
                    stats_data = {
                        "archivo_origen": filename,
                        "texto_key": text_key,
                        "archivo_audio": output_filename,
                        "texto_sintetizado": text_content,
                        "voz_usada": voice_name,
                        "velocidad": metrics["speed_factor"],
                        "duracion_audio": metrics["audio_duration"],
                        "tiempo_sintesis": metrics["synthesis_time"],
                        "factor_tiempo_real": metrics["real_time_factor"],
                        "caracteres_texto": metrics["text_length"],
                        "size_audio_bytes": file_size,
                        "metodo_ajuste_velocidad": metrics["metodo_ajuste_velocidad"],
                        "temperature": metrics["temperature"],
                        "top_p": metrics["top_p"],
                        "repetition_penalty": metrics["repetition_penalty"],
                        "max_new_tokens": metrics["max_new_tokens"],
                        "chunk_length": metrics["chunk_length"],
                        "intentos_realizados": metrics["attempts"],
                        "seed_utilizada": metrics["seed_utilizada"]
                    }
                    
                    self.stats_manager.log_synthesis(stats_data)
                    
                    # Marcar texto como procesado
                    text_hash = hashlib.md5(text_content.encode()).hexdigest()
                    successfully_processed[text_key] = text_hash
                    
                    logger.info(f"✅ Completado: {text_key} → {output_filename}")
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando {text_key}: {e}")
            
            # Actualizar estado de archivo procesado
            if successfully_processed:
                if filename not in self.processed_files:
                    self.processed_files[filename] = {'texts_processed': {}}
                
                self.processed_files[filename]['hash'] = file_hash
                self.processed_files[filename]['texts_processed'].update(successfully_processed)
                self.processed_files[filename]['last_processed'] = datetime.now().isoformat()
                
                self._save_processed_files()
                
                logger.info(f"📊 Procesados {len(successfully_processed)} textos de {filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error procesando {filename}: {e}")
            return False
    
    def scan_and_process(self):
        """Escanear carpeta IN y procesar archivos"""
        inputs_dir = self.config['inputs_dir']
        
        if not os.path.exists(inputs_dir):
            logger.debug(f"Directorio IN no existe: {inputs_dir}")
            return
        
        txt_files = [f for f in os.listdir(inputs_dir) if f.endswith('.txt')]
        
        if not txt_files:
            logger.debug("No hay archivos .txt en IN")
            return
        
        processed_count = 0
        for filename in txt_files:
            filepath = os.path.join(inputs_dir, filename)
            if self.process_file(filepath):
                processed_count += 1
        
        if processed_count > 0:
            logger.info(f"📊 Archivos procesados en este ciclo: {processed_count}")

class FileWatcher(FileSystemEventHandler):
    """Observador de cambios usando watchdog"""
    
    def __init__(self, processor: FileProcessor):
        super().__init__()
        self.processor = processor
        self.last_process_time = 0
        self.debounce_delay = 2
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logger.info(f"📁 Nuevo archivo detectado: {os.path.basename(event.src_path)}")
            self._schedule_processing()
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logger.debug(f"📝 Archivo modificado: {os.path.basename(event.src_path)}")
            self._schedule_processing()
    
    def _schedule_processing(self):
        """Programar procesamiento con debounce"""
        current_time = time.time()
        if current_time - self.last_process_time > self.debounce_delay:
            self.last_process_time = current_time
            
            def delayed_process():
                time.sleep(1)
                self.processor.scan_and_process()
            
            import threading
            threading.Thread(target=delayed_process, daemon=True).start()

def main():
    """Función principal del servicio"""
    logger.info("🚀 Iniciando servicio de síntesis de voz por lotes v2.0")
    
    try:
        # Inicializar sintetizador
        synthesizer = VoiceSynthesizer(config)
        synthesizer.configure_optimizations()
        synthesizer.initialize_models()
        
        # Mostrar voces disponibles
        voices = synthesizer.get_available_voices()
        if voices:
            logger.info(f"🎤 Voces disponibles ({len(voices)}): {', '.join(voices)}")
        else:
            logger.warning("⚠️ No hay voces disponibles en samples/")
        
        # Inicializar procesador
        processor = FileProcessor(config, synthesizer)
        
        # Configurar sistema de monitoreo
        observer = None
        enable_watcher = config.get('processing_config', {}).get('enable_file_watcher', True)
        scan_interval = config.get('processing_config', {}).get('scan_interval_seconds', 5)
        
        if WATCHDOG_AVAILABLE and enable_watcher:
            logger.info("👀 Configurando monitoreo con watchdog...")
            event_handler = FileWatcher(processor)
            observer = Observer()
            observer.schedule(event_handler, config["inputs_dir"], recursive=False)
            observer.start()
            logger.info(f"👀 Watchdog monitoreando: {config['inputs_dir']}")
        else:
            if not WATCHDOG_AVAILABLE:
                logger.info("ℹ️ Watchdog no disponible, usando polling")
            else:
                logger.info("ℹ️ Watchdog deshabilitado, usando polling")
        
        logger.info(f"📂 Carpeta de entrada: {config['inputs_dir']}")
        logger.info(f"💾 Carpeta de salida: {config['outputs_dir']}")
        logger.info(f"📊 Estadísticas CSV: {config['logs_dir']}/synthesis_stats.csv")
        logger.info("🔄 Servicio activo. Presione Ctrl+C para detener.")
        
        # Procesamiento inicial
        if config.get('processing_config', {}).get('process_on_startup', True):
            logger.info("🔍 Realizando escaneo inicial...")
            processor.scan_and_process()
        
        # Bucle principal
        try:
            while True:
                if not (WATCHDOG_AVAILABLE and enable_watcher):
                    # Modo polling
                    time.sleep(scan_interval)
                    processor.scan_and_process()
                else:
                    # Modo watchdog - solo mantenimiento
                    time.sleep(30)
                
                # Limpiar memoria periódicamente
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        except KeyboardInterrupt:
            logger.info("🛑 Deteniendo servicio...")
            
            if observer:
                observer.stop()
                observer.join()
        
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("✅ Servicio detenido correctamente")

if __name__ == "__main__":
    main()
EOF

print_success "batch_service.py v2.0 creado correctamente"

# Crear archivo de ejemplo para pruebas
print_message "Creando archivos de ejemplo..."
sudo -u tts_service cat > "$TTS_HOME/IN/ejemplo_multiple.txt" << 'EOF'
{
    "text": "Este es el primer texto para sintetizar con el sistema de lotes versión 2.0.",
    "text_1": "Este es un segundo texto en el mismo archivo JSON.",
    "text_2": "Y este es un tercer texto que también será procesado automáticamente.",
    "metadata": {
        "source": "ejemplo_instalador",
        "version": "2.0"
    }
}
EOF

# Crear README con instrucciones
sudo -u tts_service cat > "$TTS_HOME/README.md" << 'EOF'
# Sistema de Síntesis de Voz TTS v2.0

## Estructura del sistema

- `fish-speech/` - Repositorio de Fish Speech con modelos
- `ambiente_tts/` - Entorno virtual de Python
- `samples/` - Voces para clonación (archivos .wav + .txt)
- `scripts/` - Servicio de procesamiento por lotes
- `config/` - Archivo de configuración
- `IN/` - Archivos .txt con contenido JSON para procesar
- `OUT/` - Archivos .wav generados
- `logs/` - Logs del sistema y estadísticas CSV

## Configuración inicial

1. Coloque archivos de voz en `samples/`:
   - `mi_voz.wav` - Archivo de audio (5-30 segundos)
   - `mi_voz.txt` - Transcripción exacta del audio

2. Edite `config/config.json` y cambie:
   ```json
   "voz": "mi_voz"
   ```

3. Ejecute el servicio manualmente:
   ```bash
   source /home/tts_service/ambiente_tts/bin/activate
   python3 /home/tts_service/scripts/batch_service.py
   ```

## Uso del sistema

1. Cree archivos .txt en la carpeta `IN/` con contenido JSON:
   ```json
   {
       "text": "Primer texto a sintetizar",
       "text_1": "Segundo texto opcional",
       "text_2": "Tercer texto opcional"
   }
   ```

2. El sistema detectará automáticamente los archivos y generará:
   - `OUT/archivo.wav` (para el campo "text")
   - `OUT/archivo_text_1.wav` (para el campo "text_1")
   - `OUT/archivo_text_2.wav` (para el campo "text_2")

3. Consulte las estadísticas detalladas en:
   - `logs/synthesis_stats.csv`

## Configuración como servicio systemd

Para configurar como daemon del sistema, ejecute:
```bash
sudo ./setup_daemon.sh
```

Luego use los comandos:
```bash
sudo systemctl start tts-batch
sudo systemctl stop tts-batch
sudo systemctl status tts-batch
sudo journalctl -u tts-batch -f
```
EOF

# Crear instrucciones de samples
sudo -u tts_service cat > "$TTS_HOME/samples/README.md" << 'EOF'
# Carpeta de Voces (Samples)

Para agregar nuevas voces al sistema:

1. **Archivo de audio**: Coloque un archivo `.wav` con una muestra de la voz
   - Formato: WAV, 44.1kHz o 22.05kHz
   - Duración: 5-30 segundos de audio claro
   - Nombre: `nombre_voz.wav`

2. **Archivo de transcripción**: Cree un archivo `.txt` con el mismo nombre
   - Contenido: Transcripción exacta del audio
   - Codificación: UTF-8
   - Nombre: `nombre_voz.txt`

## Ejemplo:
- `mi_voz.wav` - Archivo de audio
- `mi_voz.txt` - "Hola, mi nombre es Juan y esta es una muestra de mi voz para clonación."

## Configuración:
Después de agregar los archivos, edite `config/config.json`:
```json
"synthesis_config": {
    "voz": "mi_voz"
}
```
EOF

# Ajustar permisos
print_message "Configurando permisos..."
chown -R tts_service:tts_service "$TTS_HOME"
chmod +x "$TTS_HOME/scripts/batch_service.py"

# Completar instalación
print_success "🎉 Instalación completada exitosamente!"
print_message ""
print_message "📋 Próximos pasos:"
print_message "1. Coloque archivos de voz en: $TTS_HOME/samples/"
print_message "2. Edite la configuración en: $TTS_HOME/config/config.json"
print_message "3. Pruebe el sistema:"
print_message "   sudo -u tts_service bash -c 'source $TTS_HOME/ambiente_tts/bin/activate && python3 $TTS_HOME/scripts/batch_service.py'"
print_message "4. Configure como daemon: sudo ./setup_daemon.sh"
print_message ""
print_message "📁 Archivos de ejemplo creados en: $TTS_HOME/IN/"
print_message "📖 Consulte el README.md para más información"

# Verificar si se necesita reinicio para drivers NVIDIA
if [ -n "$(dpkg -l | grep -i nvidia | grep -i 'ii' | grep '570')" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    print_warning "Se han instalado nuevos drivers NVIDIA. Es recomendable reiniciar el sistema."
    echo ""
    read -p "¿Desea reiniciar ahora? (s/n): " -n 1 -r
    echo
    if [[ "$REPLY" =~ ^[SsYy]$ ]]; then
        print_message "Reiniciando el sistema..."
        reboot
    else
        print_warning "Recuerde reiniciar el sistema antes de usar el servicio."
    fi
fi

print_success "✅ Sistema TTS v2.0 listo para usar!"