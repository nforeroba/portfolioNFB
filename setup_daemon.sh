#!/bin/bash

# Script para configurar el servicio TTS como daemon systemd
# Configura tts-batch.service para ejecución automática

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

print_message "=== Configurador de Daemon TTS v2.0 ==="
print_message "Configurando servicio systemd para tts-batch"

# Verificar que existe el usuario tts_service
if ! id "tts_service" &>/dev/null; then
    print_error "Usuario tts_service no existe. Ejecute primero instalador_tts.sh"
    exit 1
fi

# Verificar que existe el script batch_service.py
TTS_HOME="/home/tts_service"
BATCH_SCRIPT="$TTS_HOME/scripts/batch_service.py"

if [ ! -f "$BATCH_SCRIPT" ]; then
    print_error "No se encontró batch_service.py en $BATCH_SCRIPT"
    print_error "Ejecute primero instalador_tts.sh"
    exit 1
fi

# Verificar que existe el ambiente virtual
VENV_PATH="$TTS_HOME/ambiente_tts"
if [ ! -d "$VENV_PATH" ]; then
    print_error "No se encontró el ambiente virtual en $VENV_PATH"
    print_error "Ejecute primero instalador_tts.sh"
    exit 1
fi

print_success "Verificaciones iniciales completadas"

# Crear archivo de servicio systemd
print_message "Creando archivo de servicio systemd..."

cat > /etc/systemd/system/tts-batch.service << EOF
[Unit]
Description=TTS Batch Service v2.0 - Fish Speech Voice Synthesis
Documentation=file://$TTS_HOME/README.md
After=network.target multi-user.target
Wants=network.target

[Service]
Type=simple
User=tts_service
Group=tts_service
WorkingDirectory=$TTS_HOME

# Variables de entorno
Environment=PATH=$VENV_PATH/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64
Environment=VIRTUAL_ENV=$VENV_PATH
Environment=PYTHONPATH=$TTS_HOME/fish-speech
Environment=CUDA_VISIBLE_DEVICES=0

# Comando de ejecución
ExecStart=$VENV_PATH/bin/python $BATCH_SCRIPT

# Configuración de reinicio
Restart=always
RestartSec=10
StartLimitBurst=3
StartLimitIntervalSec=60

# Configuración de logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tts-batch

# Configuración de seguridad
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$TTS_HOME

# Configuración de recursos
TimeoutStartSec=120
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

print_success "Archivo de servicio creado: /etc/systemd/system/tts-batch.service"

# Ajustar permisos del archivo de servicio
chmod 644 /etc/systemd/system/tts-batch.service

# Recargar configuración de systemd
print_message "Recargando configuración de systemd..."
systemctl daemon-reload

# Habilitar el servicio para inicio automático
print_message "Habilitando servicio para inicio automático..."
systemctl enable tts-batch.service

print_success "Servicio habilitado para inicio automático"

# Verificar configuración
print_message "Verificando configuración del servicio..."
if systemctl is-enabled tts-batch.service >/dev/null 2>&1; then
    print_success "✅ Servicio correctamente habilitado"
else
    print_error "❌ Error: Servicio no está habilitado"
    exit 1
fi

# Probar que el servicio puede iniciarse
print_message "Realizando prueba de inicio del servicio..."
if systemctl start tts-batch.service; then
    sleep 5
    
    if systemctl is-active tts-batch.service >/dev/null 2>&1; then
        print_success "✅ Servicio iniciado correctamente"
        
        # Mostrar estado actual
        print_message "Estado actual del servicio:"
        systemctl status tts-batch.service --no-pager -l
        
        # Detener el servicio para dejarlo listo para uso manual
        print_message "Deteniendo servicio de prueba..."
        systemctl stop tts-batch.service
        
    else
        print_error "❌ El servicio no se pudo iniciar correctamente"
        print_message "Revisando logs de error..."
        journalctl -u tts-batch.service --no-pager -l --since="5 minutes ago"
        exit 1
    fi
else
    print_error "❌ Error al iniciar el servicio"
    print_message "Revisando logs de error..."
    journalctl -u tts-batch.service --no-pager -l --since="5 minutes ago"
    exit 1
fi

print_success "✅ Configuración de daemon completada exitosamente!"

print_message ""
print_message "🎉 Servicio TTS configurado como daemon systemd"
print_message ""
print_message "📋 Comandos disponibles:"
print_message "   sudo systemctl start tts-batch      # Iniciar servicio"
print_message "   sudo systemctl stop tts-batch       # Detener servicio"
print_message "   sudo systemctl restart tts-batch    # Reiniciar servicio"
print_message "   sudo systemctl status tts-batch     # Ver estado del servicio"
print_message "   sudo systemctl enable tts-batch     # Habilitar inicio automático"
print_message "   sudo systemctl disable tts-batch    # Deshabilitar inicio automático"
print_message ""
print_message "📊 Comandos de monitoreo:"
print_message "   sudo journalctl -u tts-batch -f     # Ver logs en tiempo real"
print_message "   sudo journalctl -u tts-batch --since=\"1 hour ago\"  # Ver logs recientes"
print_message "   sudo journalctl -u tts-batch --since=\"today\"       # Ver logs del día"
print_message ""
print_message "📁 Ubicaciones importantes:"
print_message "   Configuración: $TTS_HOME/config/config.json"
print_message "   Archivos de entrada: $TTS_HOME/IN/"
print_message "   Archivos de salida: $TTS_HOME/OUT/"
print_message "   Logs del sistema: $TTS_HOME/logs/"
print_message "   Estadísticas CSV: $TTS_HOME/logs/synthesis_stats.csv"
print_message ""
print_message "🔧 Antes de iniciar el servicio:"
print_message "1. Coloque archivos de voz en: $TTS_HOME/samples/"
print_message "2. Configure la voz en: $TTS_HOME/config/config.json"
print_message "3. Coloque archivos JSON en: $TTS_HOME/IN/"
print_message ""
print_message "▶️ Para iniciar el servicio ahora:"
print_message "   sudo systemctl start tts-batch"
print_message ""
print_warning "NOTA: El servicio está configurado para reinicio automático."
print_warning "Si hay errores de configuración, se reintentará 3 veces antes de detenerse."

print_success "🎯 Daemon TTS listo para usar!"