import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import queue
from collections import deque
import copy
from dialer_ratio_predictor import DialerRatioPredictor

class ContactCenterSimulator:
    def __init__(self, 
                 buffer_size=3600,  # Store 1 hour of per-second data
                 update_interval=1):  # Generate data every second
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.data_buffer = deque(maxlen=buffer_size)
        self.data_queue = queue.Queue()
        self.running = False
        self.initialize_patterns()
        
    def initialize_patterns(self):
        """Initialize time-based patterns for realistic data generation"""
        # Hourly patterns (24-hour cycle)
        self.hour_patterns = {
            'agent_factor': {
                i: 1 + 0.5 * np.sin((i - 12) * np.pi / 12) 
                for i in range(24)
            },
            'contact_rate_factor': {
                i: 0.3 + 0.1 * np.sin((i - 14) * np.pi / 12)
                for i in range(24)
            }
        }
        
        # Base metrics
        self.base_metrics = {
            'agents_available': 20,
            'utilization_rate': 0.85,
            'contact_rate': 0.3,
            'abandon_rate': 0.1,
            'mean_call_duration': 180  # seconds
        }

    def generate_datapoint(self, timestamp):
        """Generate a single datapoint with realistic patterns"""
        hour = timestamp.hour
        
        # Apply time-based patterns with some random variation
        agents_available = int(
            self.base_metrics['agents_available'] * 
            self.hour_patterns['agent_factor'][hour] * 
            (1 + np.random.normal(0, 0.1))
        )
        
        # Calculate dependent metrics
        agents_connected = int(
            agents_available * 
            self.base_metrics['utilization_rate'] * 
            (1 + np.random.normal(0, 0.05))
        )
        
        contact_rate = (
            self.base_metrics['contact_rate'] * 
            self.hour_patterns['contact_rate_factor'][hour] * 
            (1 + np.random.normal(0, 0.1))
        )
        
        # Generate dials based on agents and some randomness
        dials = int(agents_available * 4 * (1 + np.random.normal(0, 0.2)))
        
        # Calculate contacts and abandonments
        contacts = int(dials * contact_rate)
        abandonments = int(
            contacts * 
            self.base_metrics['abandon_rate'] * 
            (1 + np.random.normal(0, 0.2))
        )
        
        # Generate call durations
        call_duration = max(0, np.random.normal(
            self.base_metrics['mean_call_duration'],
            30
        ))
        
        # Ensure non-negative values and logical constraints
        agents_connected = max(0, min(agents_connected, agents_available))
        contacts = max(0, min(contacts, dials))
        abandonments = max(0, min(abandonments, contacts))
        
        return {
            'timestamp': timestamp,
            'agents_available': agents_available,
            'agents_connected': agents_connected,
            'dials': dials,
            'contacts': contacts,
            'abandonments': abandonments,
            'call_duration': call_duration
        }

    def start(self):
        """Start the data generation thread"""
        self.running = True
        thread = threading.Thread(target=self._generate_data)
        thread.daemon = True
        thread.start()

    def stop(self):
        """Stop the data generation"""
        self.running = False
        
    def generate_initial_data(self, minutes=60):
        """Generate initial training data"""
        data = []
        base_time = datetime.now() - timedelta(minutes=minutes)
        
        for i in range(minutes * 60):  # Generate per-second data
            current_time = base_time + timedelta(seconds=i)
            datapoint = self.generate_datapoint(current_time)
            data.append(datapoint)
        
        return pd.DataFrame(data)

    def _generate_data(self):
        """Continuously generate data points"""
        while self.running:
            current_time = datetime.now()
            datapoint = self.generate_datapoint(current_time)
            
            # Add to both buffer and queue
            self.data_buffer.append(datapoint)
            self.data_queue.put(datapoint)
            
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

def create_realtime_dashboard():
    st.set_page_config(page_title="Real-time Dialer Optimizer", layout="wide")
    st.title("Real-time Contact Center Optimizer")

    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = ContactCenterSimulator()
        st.session_state.predictor = DialerRatioPredictor(
            rolling_window=30,
            target_contact_rate=0.3,
            target_utilization=0.85,
            max_abandon_rate=0.1
        )
        st.session_state.started = False
        st.session_state.data_buffer = pd.DataFrame()

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Simulation" if not st.session_state.started else "Stop Simulation"):
            if not st.session_state.started:
                # Generate initial training data
                initial_data = []
                base_time = datetime.now() - timedelta(minutes=60)  # Last hour
                for i in range(60 * 60):  # One hour of per-second data
                    current_time = base_time + timedelta(seconds=i)
                    datapoint = st.session_state.simulator.generate_datapoint(current_time)
                    initial_data.append(datapoint)
                
                # Train the model with initial data
                initial_df = pd.DataFrame(initial_data)
                metrics, _ = st.session_state.predictor.train(initial_df)
                st.write("Model trained successfully. Metrics:", metrics)
                
                # Start simulation
                st.session_state.simulator.start()
                st.session_state.started = True
            else:
                st.session_state.simulator.stop()
                st.session_state.started = False

    # Real-time metrics
    if st.session_state.started:
        placeholder = st.empty()
        while st.session_state.started:
            # Get new data
            new_data = st.session_state.simulator.get_new_data()
            if new_data is not None:
                st.session_state.data_buffer = pd.concat([
                    st.session_state.data_buffer, new_data
                ]).tail(3600)  # Keep last hour
                
                # Prepare data for prediction
                resampled_data = st.session_state.predictor.resample_data(
                    st.session_state.data_buffer
                )
                features = st.session_state.predictor.prepare_features(resampled_data)
                predictions = st.session_state.predictor.predict(features)
                
                with placeholder.container():
                    # Current metrics
                    latest = new_data.iloc[-1]
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Agents Available", int(latest['agents_available']))
                        st.metric("Agents Connected", int(latest['agents_connected']))
                    
                    with metrics_col2:
                        st.metric("Current Dials", int(latest['dials']))
                        st.metric("Contacts", int(latest['contacts']))
                    
                    with metrics_col3:
                        current_ratio = latest['dials'] / latest['agents_available']
                        optimal_ratio = predictions[-1] if len(predictions) > 0 else 0
                        st.metric("Current Ratio", f"{current_ratio:.2f}")
                        st.metric("Recommended Ratio", f"{optimal_ratio:.2f}")
                    
                    # Real-time plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Ratio comparison plot
                        fig_ratio = go.Figure()
                        fig_ratio.add_trace(go.Scatter(
                            x=resampled_data['timestamp'],
                            y=resampled_data['dials'] / resampled_data['agents_available'],
                            name='Actual Ratio'
                        ))
                        fig_ratio.add_trace(go.Scatter(
                            x=resampled_data['timestamp'],
                            y=predictions,
                            name='Optimal Ratio'
                        ))
                        fig_ratio.update_layout(title="Dialing Ratio Over Time")
                        st.plotly_chart(fig_ratio, use_container_width=True)
                    
                    with col2:
                        # Performance metrics plot
                        performance = st.session_state.predictor.calculate_performance_score(
                            resampled_data
                        )
                        fig_perf = px.line(
                            x=resampled_data['timestamp'],
                            y=performance,
                            title="Performance Score"
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
            
            time.sleep(1)

if __name__ == "__main__":
    create_realtime_dashboard()