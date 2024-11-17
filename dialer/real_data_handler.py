from func.conexion_ant import actualizar_historico
from func.conexion_act import conexion_base
import time
import pandas as pd
from collections import deque
import queue
import threading
from datetime import datetime
from Ejec import (
    path_config,
    path_key,
    path_credenciales,
    path_sql_act,
    path_sql_ant,
    path_csv
)

class RealDataHandler:
    def __init__(self, 
                 buffer_size=4800,  # Store up to 80 minutes of per-second data
                 update_interval=1):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.data_buffer = deque(maxlen=buffer_size)
        self.data_queue = queue.Queue()
        self.running = False
        
    def start(self):
        """Start the data collection thread"""
        self.running = True
        # First, update historical data
        actualizar_historico(
            path_config,
            path_key,
            path_credenciales,
            path_sql_ant,
            path_csv
        )
        # Start real-time data collection thread
        thread = threading.Thread(target=self._collect_data)
        thread.daemon = True
        thread.start()

    def stop(self):
        """Stop the data collection"""
        self.running = False
        
    def generate_initial_data(self, minutes=60):
        """Load historical data for initial display"""
        try:
            df = pd.read_csv(path_csv)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Get the last 'minutes' worth of data
            cutoff_time = datetime.now() - pd.Timedelta(minutes=minutes)
            initial_data = df[df['timestamp'] >= cutoff_time].copy()
            return initial_data
        except Exception as e:
            print(f"Error loading initial data: {e}")
            return pd.DataFrame()

    def _collect_data(self):
        """Continuously collect real-time data"""
        while self.running:
            try:
                fecha_actual = datetime.now().strftime("%Y-%m-%d")
                # Get new data from database
                db = conexion_base(
                    path_config,
                    path_key,
                    path_credenciales,
                    path_sql_act,
                    fecha_actual
                )
                
                if not db.empty:
                    # Ensure data has correct types
                    if not pd.api.types.is_datetime64_any_dtype(db['timestamp']):
                        db['timestamp'] = pd.to_datetime(db['timestamp'])
                    
                    # Add to both buffer and queue
                    self.data_buffer.append(db.iloc[-1].to_dict())
                    self.data_queue.put(db.iloc[-1].to_dict())
                
            except Exception as e:
                print(f"Error collecting data: {e}")
            
            time.sleep(self.update_interval)

    def get_current_data(self):
        """Get all data from the buffer as a DataFrame"""
        return pd.DataFrame(list(self.data_buffer))

    def get_new_data(self):
        """Get new data points from the queue"""
        new_data = []
        while not self.data_queue.empty():
            new_data.append(self.data_queue.get())
        return pd.DataFrame(new_data) if new_data else None