import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DialerRatioPredictor:
    def __init__(self, 
                 rolling_window=30,
                 target_contact_rate=0.3,
                 target_utilization=0.85,
                 max_abandon_rate=0.1,
                 weights={
                     'contact_rate': 0.4,
                     'utilization': 0.3,
                     'abandon_rate': 0.3
                 }):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.rolling_window = rolling_window
        self.target_contact_rate = target_contact_rate
        self.target_utilization = target_utilization
        self.max_abandon_rate = max_abandon_rate
        self.weights = weights
        self.feature_columns = None
        self.is_trained = False

    def resample_data(self, df):
        """Resample data and normalize counts by window size"""
        df = df.copy()
        
        # Convert to datetime if string
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        df = df.set_index('timestamp')
        
        # Resample data
        resampled = pd.DataFrame()
        
        # Average metrics over the window
        window_size = f'{self.rolling_window}S'  # Using seconds for real-time
        
        # Calculate averages
        resampled = df.resample(window_size).agg({
            'agents_available': 'mean',
            'agents_connected': 'mean',
            'dials': 'sum',
            'contacts': 'sum',
            'abandonments': 'sum',
            'call_duration': 'mean'
        }).fillna(method='ffill')
        
        # Normalize counts by window size
        resampled['dials'] = resampled['dials'] / self.rolling_window
        resampled['contacts'] = resampled['contacts'] / self.rolling_window
        resampled['abandonments'] = resampled['abandonments'] / self.rolling_window
        
        return resampled.reset_index()

    def calculate_performance_score(self, df):
        """Calculate performance score based on target metrics"""
        contact_rate = df['contacts'] / df['dials'].replace(0, 1)
        utilization = df['agents_connected'] / df['agents_available']
        abandon_rate = df['abandonments'] / df['contacts'].replace(0, 1)
        
        contact_score = 1 - abs(contact_rate - self.target_contact_rate) / self.target_contact_rate
        util_score = 1 - abs(utilization - self.target_utilization) / self.target_utilization
        abandon_score = 1 - np.clip(abandon_rate / self.max_abandon_rate, 0, 1)
        
        return (
            self.weights['contact_rate'] * contact_score +
            self.weights['utilization'] * util_score +
            self.weights['abandon_rate'] * abandon_score
        )

    def prepare_features(self, df):
        """Prepare features for prediction"""
        features = pd.DataFrame(index=df.index)
        
        # Basic features
        features['agents_connected'] = df['agents_connected']
        features['agents_available'] = df['agents_available']
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Performance metrics
        features['contact_rate'] = df['contacts'] / df['dials'].replace(0, 1)
        features['abandon_rate'] = df['abandonments'] / df['contacts'].replace(0, 1)
        features['utilization'] = df['agents_connected'] / df['agents_available']
        features['avg_call_duration'] = df['call_duration']
        
        # Fill missing values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Store feature columns for prediction
        self.feature_columns = features.columns.tolist()
        
        return features

    def find_optimal_ratios(self, df):
        """Find optimal ratios based on performance"""
        df['dialing_ratio'] = df['dials'] / df['agents_available']
        df['performance_score'] = self.calculate_performance_score(df)
        
        # Group by time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        feature_cols = ['hour', 'day_of_week', 'agents_available']
        optimal_ratios = df.groupby(feature_cols).apply(
            lambda x: pd.Series({
                'dialing_ratio': x.loc[x['performance_score'].idxmax(), 'dialing_ratio'],
                'max_performance': x['performance_score'].max()
            })
        ).reset_index()
        
        return optimal_ratios

    def train(self, df):
        """Train the model"""
        # Ensure minimum data size
        if len(df) < 100:  # If less than 100 samples
            df = pd.concat([df] * (100 // len(df) + 1))  # Replicate data to reach minimum size
        # Resample and prepare data
        resampled_data = self.resample_data(df)
        optimal_ratios = self.find_optimal_ratios(resampled_data)
        features = self.prepare_features(resampled_data)
        
        # Merge features with optimal ratios
        training_data = pd.merge(
            features,
            optimal_ratios[['hour', 'day_of_week', 'agents_available', 'dialing_ratio']],
            on=['hour', 'day_of_week', 'agents_available'],
            how='left'
        )
        
        # Fill missing ratios with median
        training_data['dialing_ratio'] = training_data['dialing_ratio'].fillna(
            resampled_data['dials'].median() / resampled_data['agents_available'].median()
        )
        
        # Train model
        X = training_data[self.feature_columns]
        y = training_data['dialing_ratio']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'feature_importance': dict(zip(self.feature_columns, 
                                        self.model.feature_importances_))
        }
        
        return metrics, resampled_data

    def predict(self, features_df):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if self.feature_columns is None:
            raise ValueError("Feature columns not initialized. Run train() first")
        
        features = features_df[self.feature_columns]
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def update_model(self, new_data):
        """Update model with new data"""
        if not self.is_trained:
            return self.train(new_data)
        
        # Prepare new data
        resampled_data = self.resample_data(new_data)
        optimal_ratios = self.find_optimal_ratios(resampled_data)
        features = self.prepare_features(resampled_data)
        
        # Merge features with optimal ratios
        update_data = pd.merge(
            features,
            optimal_ratios[['hour', 'day_of_week', 'agents_available', 'dialing_ratio']],
            on=['hour', 'day_of_week', 'agents_available'],
            how='left'
        )
        
        # Update model (simple update by retraining)
        X = update_data[self.feature_columns]
        y = update_data['dialing_ratio'].fillna(
            resampled_data['dials'].median() / resampled_data['agents_available'].median()
        )
        
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        
        return {
            'status': 'Model updated successfully',
            'samples_processed': len(new_data)
        }