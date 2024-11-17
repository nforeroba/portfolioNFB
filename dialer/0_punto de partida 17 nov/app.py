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
                 buffer_size=4800,  # Store up to 80 minutes of per-second data
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
            'agents_available': 1000,
            'utilization_rate': 0.85,
            'contact_rate': 0.3,
            'abandon_rate': 0.1,
            'mean_call_duration': 60  # seconds
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

def calculate_window_statistics(data, window_minutes):
    """Calculate statistics for a specific time window from the data."""
    if data.empty:
        print("No data available")
        return None
        
    # Convert window_minutes to seconds for filtering
    window_seconds = window_minutes * 60
    current_time = data['timestamp'].max()
    window_start = current_time - pd.Timedelta(minutes=window_minutes)
    
    # Filter data for the time window
    window_data = data[data['timestamp'] >= window_start].copy()
    
    print(f"Window {window_minutes}min - Total points: {len(window_data)}")
    
    if len(window_data) == 0:
        print(f"No data in {window_minutes}min window")
        return None
    
    # Calculate actual ratios for each row
    window_data['actual_ratio'] = window_data['dials'] / window_data['agents_available'].replace(0, 1)
    
    # Less strict requirement: Require at least 5% of the expected points with a minimum of 10 points
    min_required_points = max(10, int(window_seconds * 0.05))
    
    print(f"Window {window_minutes}min - Required points: {min_required_points}")
    
    if len(window_data) < min_required_points:
        print(f"Insufficient points in {window_minutes}min window")
        return None
    
    # Calculate statistics
    stats = {
        'Time Window': f'Last {window_minutes} min',
        'Actual Mean': window_data['actual_ratio'].mean(),
        'Actual Min': window_data['actual_ratio'].min(),
        'Actual Max': window_data['actual_ratio'].max(),
        'Actual StdDev': window_data['actual_ratio'].std(),
    }
    
    # Add optimal ratio statistics if available
    if 'optimal_ratio' in window_data.columns and not window_data['optimal_ratio'].isna().all():
        stats.update({
            'Optimal Mean': window_data['optimal_ratio'].mean(),
            'Optimal Min': window_data['optimal_ratio'].min(),
            'Optimal Max': window_data['optimal_ratio'].max(),
            'Optimal StdDev': window_data['optimal_ratio'].std()
        })
    
    print(f"Window {window_minutes}min - Stats calculated successfully")
    return pd.Series(stats)

def create_statistics_df(data):
    """Create a DataFrame with statistics for all time windows"""
    time_windows = [10, 20, 30, 50, 80]  # Time windows in minutes
    stats_list = []
    
    print("\nCalculating statistics...")
    print(f"Total data points available: {len(data)}")
    
    for window in time_windows:
        print(f"\nProcessing {window} minute window:")
        window_stats = calculate_window_statistics(data, window)
        if window_stats is not None:
            stats_list.append(window_stats)
    
    if not stats_list:
        print("No statistics calculated for any window")
        return pd.DataFrame()
        
    stats_df = pd.DataFrame(stats_list)
    stats_df.set_index('Time Window', inplace=True)
    print(f"\nStatistics calculated for {len(stats_df)} windows")
    return stats_df

def display_statistics_table(stats_df):
    """Display the statistics table in a formatted way"""
    st.subheader("Ratio Statistics")
    
    if stats_df.empty:
        st.info("Collecting data... Statistics will appear once sufficient data is available.")
        return
        
    # Create two columns for actual and optimal ratios
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Actual Ratio Statistics**")
        if 'Actual Mean' in stats_df.columns:
            actual_stats = stats_df[['Actual Mean', 'Actual Min', 'Actual Max', 'Actual StdDev']]
            actual_stats.columns = ['Mean', 'Min', 'Max', 'StdDev']
            st.dataframe(actual_stats.style.format("{:.2f}"))
        else:
            st.info("Collecting actual ratio statistics...")
        
    with col2:
        st.markdown("**Optimal Ratio Statistics**")
        if 'Optimal Mean' in stats_df.columns:
            optimal_stats = stats_df[['Optimal Mean', 'Optimal Min', 'Optimal Max', 'Optimal StdDev']]
            optimal_stats.columns = ['Mean', 'Min', 'Max', 'StdDev']
            st.dataframe(optimal_stats.style.format("{:.2f}"))
        else:
            st.info("Optimal ratios will appear once the model is trained.")

def create_realtime_dashboard():
    st.set_page_config(page_title="Real-time Dialer Optimizer", layout="wide")
    st.title("Real-time Contact Center Optimizer")

    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = ContactCenterSimulator()
        st.session_state.predictor = DialerRatioPredictor(
            rolling_window=5,
            target_contact_rate=0.3,
            target_utilization=0.85,
            max_abandon_rate=0.1
        )
        st.session_state.started = False
        st.session_state.data_buffer = pd.DataFrame()
        st.session_state.start_time = None
        st.session_state.model_trained = False

    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Simulation" if not st.session_state.started else "Stop Simulation"):
            if not st.session_state.started:
                st.session_state.simulator.start()
                st.session_state.started = True
                st.session_state.start_time = datetime.now()
                st.session_state.data_buffer = pd.DataFrame()
                st.session_state.model_trained = False
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
                if st.session_state.data_buffer.empty:
                    st.session_state.data_buffer = new_data
                else:
                    st.session_state.data_buffer = pd.concat(
                        [st.session_state.data_buffer, new_data],
                        ignore_index=True
                    )
                
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(st.session_state.data_buffer['timestamp']):
                    st.session_state.data_buffer['timestamp'] = pd.to_datetime(st.session_state.data_buffer['timestamp'])
                
                # Keep only the last 80 minutes of data (plus a small buffer)
                current_time = st.session_state.data_buffer['timestamp'].max()
                cutoff_time = current_time - pd.Timedelta(minutes=85)  # Keep a little extra for calculations
                st.session_state.data_buffer = st.session_state.data_buffer[
                    st.session_state.data_buffer['timestamp'] >= cutoff_time
                ].reset_index(drop=True)
                
                
                
                resampled_data = st.session_state.predictor.resample_data(
                    st.session_state.data_buffer
                )
                
                # Train model when we have enough data
                if len(resampled_data) >= 30 and not st.session_state.model_trained:
                    try:
                        metrics, _ = st.session_state.predictor.train(st.session_state.data_buffer)
                        st.session_state.model_trained = True
                    except Exception as e:
                        print(f"Error training model: {e}")
                
                # Make predictions if model is trained
                if st.session_state.model_trained:
                    features = st.session_state.predictor.prepare_features(resampled_data)
                    predictions = st.session_state.predictor.predict(features)
                    resampled_data['optimal_ratio'] = predictions
                else:
                    resampled_data['optimal_ratio'] = np.nan
                    predictions = []
                
                # Ensure timestamp is datetime
                if not pd.api.types.is_datetime64_any_dtype(resampled_data['timestamp']):
                    resampled_data['timestamp'] = pd.to_datetime(resampled_data['timestamp'])
                
                with placeholder.container():
                    # Display current metrics
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
                        optimal_ratio = predictions[-1] if len(predictions) > 0 else np.nan
                        st.metric("Current Ratio", f"{current_ratio:.2f}")
                        if not np.isnan(optimal_ratio):
                            st.metric("Recommended Ratio", f"{optimal_ratio:.2f}")
                        else:
                            st.metric("Recommended Ratio", "Collecting data...")

                    # Calculate and display statistics
                    stats_df = create_statistics_df(resampled_data)
                    display_statistics_table(stats_df)
                    
                    # Real-time plots
                    # First row of plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Ratio comparison plot
                        fig_ratio = go.Figure()
                        fig_ratio.add_trace(go.Scatter(
                            x=resampled_data['timestamp'],
                            y=resampled_data['dials'] / resampled_data['agents_available'],
                            name='Actual Ratio'
                        ))
                        if st.session_state.model_trained:
                            fig_ratio.add_trace(go.Scatter(
                                x=resampled_data['timestamp'],
                                y=predictions,
                                name='Optimal Ratio'
                            ))
                        fig_ratio.update_layout(
                            title="Dialing Ratio Over Time",
                            xaxis_title="Time",
                            yaxis_title="Ratio"
                        )
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
                        fig_perf.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Score",
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)

                    # Second row of plots
                    col3, col4, col5 = st.columns(3)

                    with col3:
                        # Contact Rate plot
                        fig_contact = go.Figure()
                        contact_rate = resampled_data['contacts'] / resampled_data['dials']
                        fig_contact.add_trace(go.Scatter(
                            x=resampled_data['timestamp'],
                            y=contact_rate,
                            name='Contact Rate'
                        ))
                        fig_contact.add_hline(
                            y=st.session_state.predictor.target_contact_rate,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Target"
                        )
                        fig_contact.update_layout(
                            title="Contact Rate Over Time",
                            xaxis_title="Time",
                            yaxis_title="Rate",
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig_contact, use_container_width=True)

                    with col4:
                        # Utilization plot
                        fig_util = go.Figure()
                        utilization = resampled_data['agents_connected'] / resampled_data['agents_available']
                        fig_util.add_trace(go.Scatter(
                            x=resampled_data['timestamp'],
                            y=utilization,
                            name='Utilization'
                        ))
                        fig_util.add_hline(
                            y=st.session_state.predictor.target_utilization,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Target"
                        )
                        fig_util.update_layout(
                            title="Agent Utilization Over Time",
                            xaxis_title="Time",
                            yaxis_title="Rate",
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig_util, use_container_width=True)

                    with col5:
                        # Abandon Rate plot
                        fig_abandon = go.Figure()
                        abandon_rate = resampled_data['abandonments'] / resampled_data['contacts']
                        fig_abandon.add_trace(go.Scatter(
                            x=resampled_data['timestamp'],
                            y=abandon_rate,
                            name='Abandon Rate'
                        ))
                        fig_abandon.add_hline(
                            y=st.session_state.predictor.max_abandon_rate,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Max"
                        )
                        fig_abandon.update_layout(
                            title="Abandon Rate Over Time",
                            xaxis_title="Time",
                            yaxis_title="Rate",
                            yaxis_range=[0, 1]
                        )
                        st.plotly_chart(fig_abandon, use_container_width=True)
            
            time.sleep(1)

if __name__ == "__main__":
    create_realtime_dashboard()