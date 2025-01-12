import os

def speak_result(text, pitch=0, rate=3, volume=100, voice="Microsoft Dalia"):
    """
    Función para sintetizar texto a voz usando PowerShell y System.Speech.
    
    Args:
        text (str): El texto que se desea sintetizar.
        pitch (int): Tonalidad del habla (valor relativo).
        rate (int): Velocidad del habla.
        volume (int): Volumen del habla (0 a 100).
        voice (str): Nombre de la voz a usar.
    """
    # Escapar comillas simples en el texto
    escaped_text = text.replace("'", "''")
    
    # Crear el comando de PowerShell
    command = f'''powershell -Command "Add-Type -AssemblyName System.Speech; ''' \
              f'''$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer; ''' \
              f'''$synthesizer.SelectVoice('{voice}'); ''' \
              f'''$synthesizer.Volume = {volume}; ''' \
              f'''$synthesizer.Rate = {rate}; ''' \
              f'''$synthesizer.Speak('{escaped_text}');"'''
    
    # Ejecutar el comando
    os.system(command)
