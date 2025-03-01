#!/usr/bin/env python3
"""
Player Performance Forecasting Module

This module provides functions for forecasting player performance
for the upcoming fantasy baseball season.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib


class PlayerForecaster:
    """
    Class for forecasting player performance for fantasy baseball.
    """
    
    def __init__(self, batting_data=None, pitching_data=None):
        """
        Initialize the PlayerForecaster with batting and pitching data.
        
        Args:
            batting_data (pd.DataFrame, optional): DataFrame with batting statistics
            pitching_data (pd.DataFrame, optional): DataFrame with pitching statistics
        """
        self.batting_data = batting_data
        self.pitching_data = pitching_data
        
        # League settings for H2H categories
        self.batting_categories = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
        self.pitching_categories = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']
        
        # Models
        self.batting_models = {}
        self.pitching_models = {}
    
    def load_data(self, batting_file, pitching_file):
        """
        Load batting and pitching data from CSV files.
        
        Args:
            batting_file (str): Path to batting statistics CSV
            pitching_file (str): Path to pitching statistics CSV
        """
        print(f"Loading batting data from {batting_file}")
        self.batting_data = pd.read_csv(batting_file)
        
        print(f"Loading pitching data from {pitching_file}")
        self.pitching_data = pd.read_csv(pitching_file)
        
        print(f"Loaded {len(self.batting_data)} batting records and {len(self.pitching_data)} pitching records")
    
    def prepare_player_data(self, data, player_id, category, min_seasons=3):
        """
        Prepare time series data for a specific player and category.
        
        Args:
            data (pd.DataFrame): DataFrame with player statistics
            player_id (str): Player ID
            category (str): Statistical category to forecast
            min_seasons (int): Minimum number of seasons required
            
        Returns:
            np.ndarray: Time series data for the player and category
        """
        player_data = data[data['PLAYER_ID'] == player_id].sort_values('SEASON')
        
        if len(player_data) < min_seasons:
            return None
        
        return player_data[category].values
    
    def forecast_arima(self, time_series, forecast_periods=1, order=(1, 0, 0)):
        """
        Forecast using ARIMA model.
        
        Args:
            time_series (np.ndarray): Time series data
            forecast_periods (int): Number of periods to forecast
            order (tuple): ARIMA order (p, d, q)
            
        Returns:
            tuple: (forecast, model)
        """
        if len(time_series) < 3:
            return None, None
        
        try:
            model = ARIMA(time_series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_periods)
            return forecast, model_fit
        except:
            return None, None
    
    def forecast_exponential_smoothing(self, time_series, forecast_periods=1):
        """
        Forecast using Exponential Smoothing.
        
        Args:
            time_series (np.ndarray): Time series data
            forecast_periods (int): Number of periods to forecast
            
        Returns:
            tuple: (forecast, model)
        """
        if len(time_series) < 3:
            return None, None
        
        try:
            model = ExponentialSmoothing(time_series, trend='add', seasonal=None)
            model_fit = model.fit()
            forecast = model_fit.forecast(forecast_periods)
            return forecast, model_fit
        except:
            return None, None
    
    def prepare_regression_features(self, data, player_id_col='PLAYER_ID', season_col='SEASON'):
        """
        Prepare features for regression models.
        
        Args:
            data (pd.DataFrame): DataFrame with player statistics
            player_id_col (str): Column name for player ID
            season_col (str): Column name for season
            
        Returns:
            pd.DataFrame: DataFrame with regression features
        """
        # Create a copy of the data
        df = data.copy()
        
        # Create lagged features (previous season stats)
        players = df[player_id_col].unique()
        all_rows = []
        
        for player in players:
            player_data = df[df[player_id_col] == player].sort_values(season_col)
            
            if len(player_data) < 2:
                continue
            
            for i in range(1, len(player_data)):
                current_season = player_data.iloc[i]
                prev_season = player_data.iloc[i-1]
                
                row = {
                    'PLAYER_ID': player,
                    'SEASON': current_season[season_col],
                    'AGE': current_season.get('AGE', 0)
                }
                
                # Add current season stats (these will be the targets)
                for col in player_data.columns:
                    if col not in [player_id_col, season_col, 'AGE']:
                        row[col] = current_season[col]
                
                # Add previous season stats as features
                for col in player_data.columns:
                    if col not in [player_id_col, season_col, 'AGE']:
                        row[f'PREV_{col}'] = prev_season[col]
                
                # Add age-related features
                if 'AGE' in current_season:
                    age = current_season['AGE']
                    row['AGE_SQUARED'] = age ** 2
                    
                    # Age indicators for different career phases
                    row['EARLY_CAREER'] = 1 if age < 26 else 0
                    row['PRIME'] = 1 if 26 <= age <= 32 else 0
                    row['DECLINE'] = 1 if age > 32 else 0
                
                all_rows.append(row)
        
        return pd.DataFrame(all_rows)
    
    def train_regression_model(self, data, target, features, model_type='ridge'):
        """
        Train a regression model for forecasting.
        
        Args:
            data (pd.DataFrame): DataFrame with features and target
            target (str): Target column name
            features (list): List of feature column names
            model_type (str): Type of regression model
            
        Returns:
            tuple: (trained_model, feature_importance)
        """
        X = data[features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select model
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
            param_grid = {'alpha': [0.1, 1.0, 10.0]}
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model for {target} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Get feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importance = np.abs(best_model.coef_)
        else:
            importance = np.ones(len(features))
        
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return best_model, feature_importance
    
    def train_batting_models(self, model_type='ensemble'):
        """
        Train models for forecasting batting statistics.
        
        Args:
            model_type (str): Type of model to train
            
        Returns:
            dict: Dictionary of trained models
        """
        if self.batting_data is None:
            raise ValueError("Batting data not loaded. Call load_data() first.")
        
        print("Preparing batting data for modeling...")
        regression_data = self.prepare_regression_features(self.batting_data)
        
        # Create output directory for models
        os.makedirs('data/models', exist_ok=True)
        
        models = {}
        
        for category in tqdm(self.batting_categories, desc="Training batting models"):
            print(f"\nTraining model for {category}")
            
            # Define features
            features = [col for col in regression_data.columns if col.startswith('PREV_')]
            
            if 'AGE' in regression_data.columns:
                features.extend(['AGE', 'AGE_SQUARED', 'EARLY_CAREER', 'PRIME', 'DECLINE'])
            
            # Train model
            if model_type == 'ensemble':
                # Train multiple models and create an ensemble
                ridge_model, _ = self.train_regression_model(regression_data, category, features, 'ridge')
                rf_model, _ = self.train_regression_model(regression_data, category, features, 'random_forest')
                gb_model, importance = self.train_regression_model(regression_data, category, features, 'gradient_boosting')
                
                models[category] = {
                    'ridge': ridge_model,
                    'random_forest': rf_model,
                    'gradient_boosting': gb_model,
                    'feature_importance': importance
                }
                
                # Save models
                joblib.dump(models[category], f'data/models/batting_{category}_ensemble.joblib')
            else:
                # Train a single model
                model, importance = self.train_regression_model(regression_data, category, features, model_type)
                
                models[category] = {
                    model_type: model,
                    'feature_importance': importance
                }
                
                # Save model
                joblib.dump(models[category], f'data/models/batting_{category}_{model_type}.joblib')
        
        self.batting_models = models
        return models
    
    def train_pitching_models(self, model_type='ensemble'):
        """
        Train models for forecasting pitching statistics.
        
        Args:
            model_type (str): Type of model to train
            
        Returns:
            dict: Dictionary of trained models
        """
        if self.pitching_data is None:
            raise ValueError("Pitching data not loaded. Call load_data() first.")
        
        print("Preparing pitching data for modeling...")
        regression_data = self.prepare_regression_features(self.pitching_data)
        
        # Create output directory for models
        os.makedirs('data/models', exist_ok=True)
        
        models = {}
        
        for category in tqdm(self.pitching_categories, desc="Training pitching models"):
            print(f"\nTraining model for {category}")
            
            # Define features
            features = [col for col in regression_data.columns if col.startswith('PREV_')]
            
            if 'AGE' in regression_data.columns:
                features.extend(['AGE', 'AGE_SQUARED', 'EARLY_CAREER', 'PRIME', 'DECLINE'])
            
            # Train model
            if model_type == 'ensemble':
                # Train multiple models and create an ensemble
                ridge_model, _ = self.train_regression_model(regression_data, category, features, 'ridge')
                rf_model, _ = self.train_regression_model(regression_data, category, features, 'random_forest')
                gb_model, importance = self.train_regression_model(regression_data, category, features, 'gradient_boosting')
                
                models[category] = {
                    'ridge': ridge_model,
                    'random_forest': rf_model,
                    'gradient_boosting': gb_model,
                    'feature_importance': importance
                }
                
                # Save models
                joblib.dump(models[category], f'data/models/pitching_{category}_ensemble.joblib')
            else:
                # Train a single model
                model, importance = self.train_regression_model(regression_data, category, features, model_type)
                
                models[category] = {
                    model_type: model,
                    'feature_importance': importance
                }
                
                # Save model
                joblib.dump(models[category], f'data/models/pitching_{category}_{model_type}.joblib')
        
        self.pitching_models = models
        return models
    
    def generate_player_forecast(self, player_id, player_data, models, categories, is_pitcher=False):
        """
        Generate forecast for a specific player.
        
        Args:
            player_id (str): Player ID
            player_data (pd.DataFrame): DataFrame with player's historical data
            models (dict): Dictionary of trained models
            categories (list): List of categories to forecast
            is_pitcher (bool): Whether the player is a pitcher
            
        Returns:
            dict: Dictionary with forecasted values
        """
        if len(player_data) < 1:
            return None
        
        # Get the most recent season data
        recent_data = player_data.sort_values('SEASON', ascending=False).iloc[0]
        
        # Prepare features for regression models
        features = {}
        
        for col in player_data.columns:
            if col not in ['PLAYER_ID', 'SEASON', 'AGE']:
                features[f'PREV_{col}'] = recent_data[col]
        
        if 'AGE' in recent_data:
            age = recent_data['AGE'] + 1  # Increment age for next season
            features['AGE'] = age
            features['AGE_SQUARED'] = age ** 2
            features['EARLY_CAREER'] = 1 if age < 26 else 0
            features['PRIME'] = 1 if 26 <= age <= 32 else 0
            features['DECLINE'] = 1 if age > 32 else 0
        
        # Generate forecasts for each category
        forecasts = {'PLAYER_ID': player_id}
        
        for category in categories:
            if category not in models:
                continue
            
            # Get time series data for ARIMA and Exponential Smoothing
            time_series = self.prepare_player_data(
                player_data, player_id, category, min_seasons=3
            )
            
            # Initialize forecast values
            arima_forecast = None
            exp_smooth_forecast = None
            regression_forecasts = {}
            
            # ARIMA forecast
            if time_series is not None and len(time_series) >= 3:
                arima_result, _ = self.forecast_arima(time_series)
                if arima_result is not None:
                    arima_forecast = arima_result[0]
            
            # Exponential Smoothing forecast
            if time_series is not None and len(time_series) >= 3:
                exp_smooth_result, _ = self.forecast_exponential_smoothing(time_series)
                if exp_smooth_result is not None:
                    exp_smooth_forecast = exp_smooth_result[0]
            
            # Regression forecasts
            model_dict = models[category]
            for model_name, model in model_dict.items():
                if model_name != 'feature_importance' and hasattr(model, 'predict'):
                    # Prepare feature vector
                    X = pd.DataFrame([features])
                    
                    # Keep only the features the model was trained on
                    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
                    if model_features is not None:
                        X = X[model_features]
                    
                    # Make prediction
                    try:
                        prediction = model.predict(X)[0]
                        regression_forecasts[model_name] = prediction
                    except:
                        pass
            
            # Combine forecasts (simple average for now)
            valid_forecasts = []
            
            if arima_forecast is not None:
                valid_forecasts.append(arima_forecast)
            
            if exp_smooth_forecast is not None:
                valid_forecasts.append(exp_smooth_forecast)
            
            for _, forecast in regression_forecasts.items():
                valid_forecasts.append(forecast)
            
            if valid_forecasts:
                # For ERA and WHIP, we want to ensure they don't go below a reasonable threshold
                if category in ['ERA', 'WHIP'] and is_pitcher:
                    combined_forecast = max(np.mean(valid_forecasts), 1.0)  # Minimum ERA/WHIP of 1.0
                else:
                    combined_forecast = max(np.mean(valid_forecasts), 0)  # Ensure non-negative
                
                forecasts[category] = combined_forecast
            else:
                # If no valid forecasts, use the most recent value
                forecasts[category] = recent_data[category]
        
        return forecasts
    
    def generate_all_forecasts(self, output_dir='data/projections'):
        """
        Generate forecasts for all players.
        
        Args:
            output_dir (str): Directory to save forecasts
            
        Returns:
            tuple: (batting_forecasts, pitching_forecasts)
        """
        if self.batting_data is None or self.pitching_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not self.batting_models or not self.pitching_models:
            raise ValueError("Models not trained. Call train_batting_models() and train_pitching_models() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate batting forecasts
        print("Generating batting forecasts...")
        batting_forecasts = []
        
        for player_id in tqdm(self.batting_data['PLAYER_ID'].unique(), desc="Forecasting batters"):
            player_data = self.batting_data[self.batting_data['PLAYER_ID'] == player_id]
            
            forecast = self.generate_player_forecast(
                player_id, player_data, self.batting_models, self.batting_categories
            )
            
            if forecast:
                batting_forecasts.append(forecast)
        
        batting_df = pd.DataFrame(batting_forecasts)
        batting_df.to_csv(os.path.join(output_dir, 'batting_forecasts.csv'), index=False)
        
        # Generate pitching forecasts
        print("Generating pitching forecasts...")
        pitching_forecasts = []
        
        for player_id in tqdm(self.pitching_data['PLAYER_ID'].unique(), desc="Forecasting pitchers"):
            player_data = self.pitching_data[self.pitching_data['PLAYER_ID'] == player_id]
            
            forecast = self.generate_player_forecast(
                player_id, player_data, self.pitching_models, self.pitching_categories, is_pitcher=True
            )
            
            if forecast:
                pitching_forecasts.append(forecast)
        
        pitching_df = pd.DataFrame(pitching_forecasts)
        pitching_df.to_csv(os.path.join(output_dir, 'pitching_forecasts.csv'), index=False)
        
        print(f"Forecasts generated for {len(batting_forecasts)} batters and {len(pitching_forecasts)} pitchers")
        print(f"Forecasts saved to {output_dir}")
        
        return batting_df, pitching_df


if __name__ == "__main__":
    # Example usage
    print("Player Performance Forecaster")
    print("============================")
    
    # Initialize forecaster
    forecaster = PlayerForecaster()
    
    # Load data
    forecaster.load_data(
        batting_file='data/processed/batting_stats_all.csv',
        pitching_file='data/processed/pitching_stats_all.csv'
    )
    
    # Train models
    print("\nTraining batting models...")
    forecaster.train_batting_models()
    
    print("\nTraining pitching models...")
    forecaster.train_pitching_models()
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    batting_forecasts, pitching_forecasts = forecaster.generate_all_forecasts()
    
    print("\nForecasting complete!")
