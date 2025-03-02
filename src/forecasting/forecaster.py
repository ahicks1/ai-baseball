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
from joblib import Parallel, delayed


class PlayerForecaster:
    """
    Class for forecasting player performance for fantasy baseball.
    """
    
    def __init__(self, batting_data=None, pitching_data=None, bio_data=None, load_existing_models=False):
        """
        Initialize the PlayerForecaster with batting and pitching data.
        
        Args:
            batting_data (pd.DataFrame, optional): DataFrame with batting statistics
            pitching_data (pd.DataFrame, optional): DataFrame with pitching statistics
            bio_data (pd.DataFrame, optional): DataFrame with player biographical data
            load_existing_models (bool, optional): Whether to load existing models from data/models
        """
        self.batting_data = batting_data
        self.pitching_data = pitching_data
        self.bio_data = bio_data
        self.load_existing_models = load_existing_models
        
        # League settings for H2H categories
        self.batting_categories = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
        self.pitching_categories = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']
        
        # Models
        self.batting_models = {}
        self.pitching_models = {}
        
        # If load_existing_models is True, try to load models
        if load_existing_models:
            self.load_models()
    
    def load_bio_data(self, bio_file='data/raw/biofile0.csv'):
        """
        Load player biographical data from CSV file.
        
        Args:
            bio_file (str): Path to biographical data CSV
        """
        print(f"Loading biographical data from {bio_file}")
        self.bio_data = pd.read_csv(bio_file)
        print(f"Loaded biographical data for {len(self.bio_data)} players")
    
    def calculate_age(self, birthdate, season):
        """
        Calculate player's age as of March 1st for a given season.
        
        Args:
            birthdate (str): Birthdate in YYYYMMDD format
            season (int): Season year
            
        Returns:
            int: Player's age during that season
        """
        if pd.isna(birthdate) or not birthdate:
            return 28  # MLB average age for missing data
            
        birth_year = int(str(birthdate)[:4])
        birth_month = int(str(birthdate)[4:6])
        birth_day = int(str(birthdate)[6:8])
        
        # Calculate age as of March 1st of the season
        age = season - birth_year
        if birth_month > 3 or (birth_month == 3 and birth_day > 1):
            age -= 1
            
        return age
    
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
        
        # Define columns to exclude from features (non-numeric columns)
        exclude_cols = ['TEAM', 'first', 'last']
        
        # Add age data if bio_data is available
        if hasattr(self, 'bio_data') and self.bio_data is not None:
            # Create a mapping of player_id to birthdate
            player_birthdate = dict(zip(self.bio_data['id'], self.bio_data['birthdate']))
            
            # Calculate age for each player-season combination
            df['AGE'] = df.apply(
                lambda row: self.calculate_age(
                    player_birthdate.get(row[player_id_col].lower(), None), 
                    row[season_col]
                ),
                axis=1
            )
        
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
                    'AGE': current_season.get('AGE', 28)  # Use 28 as default if missing
                }
                
                # Add current season stats (these will be the targets)
                for col in player_data.columns:
                    if col not in [player_id_col, season_col, 'AGE'] and col not in exclude_cols:
                        row[col] = current_season[col]
                
                # Add previous season stats as features
                for col in player_data.columns:
                    if col not in [player_id_col, season_col, 'AGE'] and col not in exclude_cols:
                        row[f'PREV_{col}'] = prev_season[col]
                
                # Add age-related features
                age = row['AGE']
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
        
        # Drop rows with NaN values in features or target
        initial_sample_count = len(X)
        combined = pd.concat([X, pd.Series(y, name='target')], axis=1)
        combined = combined.dropna()
        X = combined[features]
        y = combined['target']
        final_sample_count = len(X)
        
        if initial_sample_count > final_sample_count:
            print(f"Dropped {initial_sample_count - final_sample_count} samples with NaN values ({(initial_sample_count - final_sample_count) / initial_sample_count:.2%} of data)")
        
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
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # Use all cores
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Grid search for hyperparameter tuning with parallelization
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)  # Use all cores
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
    
    def train_category_model(self, category, regression_data, features_prefix='batting', model_type='ensemble'):
        """
        Train models for a specific category.
        
        Args:
            category (str): Statistical category to forecast
            regression_data (pd.DataFrame): DataFrame with regression features
            features_prefix (str): Prefix for saving model files ('batting' or 'pitching')
            model_type (str): Type of model to train
            
        Returns:
            tuple: (category, model_dict) - Category name and dictionary of trained models
        """
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
            
            model_dict = {
                'ridge': ridge_model,
                'random_forest': rf_model,
                'gradient_boosting': gb_model,
                'feature_importance': importance
            }
            
            # Save models
            joblib.dump(model_dict, f'data/models/{features_prefix}_{category}_ensemble.joblib')
        else:
            # Train a single model
            model, importance = self.train_regression_model(regression_data, category, features, model_type)
            
            model_dict = {
                model_type: model,
                'feature_importance': importance
            }
            
            # Save model
            joblib.dump(model_dict, f'data/models/{features_prefix}_{category}_{model_type}.joblib')
        
        return category, model_dict
    
    def train_batting_models(self, model_type='ensemble', n_jobs=8):
        """
        Train models for forecasting batting statistics in parallel.
        
        Args:
            model_type (str): Type of model to train
            n_jobs (int): Number of parallel jobs to run (default: 8)
            
        Returns:
            dict: Dictionary of trained models
        """
        if self.batting_data is None:
            raise ValueError("Batting data not loaded. Call load_data() first.")
        
        print("Preparing batting data for modeling...")
        regression_data = self.prepare_regression_features(self.batting_data)
        
        # Create output directory for models
        os.makedirs('data/models', exist_ok=True)
        
        print(f"Training batting models in parallel using {n_jobs} cores...")
        
        # Train models in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.train_category_model)(
                category, regression_data, 'batting', model_type
            ) 
            for category in self.batting_categories
        )
        
        # Collect results
        models = {category: model_dict for category, model_dict in results}
        self.batting_models = models
        
        print(f"Completed training models for {len(models)} batting categories")
        return models
    
    def train_pitching_models(self, model_type='ensemble', n_jobs=8):
        """
        Train models for forecasting pitching statistics in parallel.
        
        Args:
            model_type (str): Type of model to train
            n_jobs (int): Number of parallel jobs to run (default: 8)
            
        Returns:
            dict: Dictionary of trained models
        """
        if self.pitching_data is None:
            raise ValueError("Pitching data not loaded. Call load_data() first.")
        
        print("Preparing pitching data for modeling...")
        regression_data = self.prepare_regression_features(self.pitching_data)
        
        # Create output directory for models
        os.makedirs('data/models', exist_ok=True)
        
        print(f"Training pitching models in parallel using {n_jobs} cores...")
        
        # Train models in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.train_category_model)(
                category, regression_data, 'pitching', model_type
            ) 
            for category in self.pitching_categories
        )
        
        # Collect results
        models = {category: model_dict for category, model_dict in results}
        self.pitching_models = models
        
        print(f"Completed training models for {len(models)} pitching categories")
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
        forecast_season = recent_data['SEASON'] + 1  # Next season
        
        # Define columns to exclude from features (non-numeric columns)
        exclude_cols = ['TEAM', 'first', 'last']
        
        # Prepare features for regression models
        features = {}
        
        for col in player_data.columns:
            if col not in ['PLAYER_ID', 'SEASON', 'AGE'] and col not in exclude_cols:
                features[f'PREV_{col}'] = recent_data[col]
        
        # Handle age calculation
        if 'AGE' in recent_data:
            # If age is already in the data, increment it for next season
            age = recent_data['AGE'] + 1
        elif hasattr(self, 'bio_data') and self.bio_data is not None:
            # If bio data is available, calculate age from birthdate
            player_id_lower = player_id.lower()
            player_bio = self.bio_data[self.bio_data['id'] == player_id_lower]
            
            if not player_bio.empty and not pd.isna(player_bio.iloc[0]['birthdate']):
                birthdate = player_bio.iloc[0]['birthdate']
                age = self.calculate_age(birthdate, forecast_season)
            else:
                age = 28  # MLB average age for missing data
        else:
            # Default age if no bio data available
            age = 28
        
        # Add age-related features
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
    
    def _generate_player_forecast_wrapper(self, player_id, data, models, categories, is_pitcher=False):
        """
        Wrapper for generate_player_forecast to use with joblib.Parallel.
        
        Args:
            player_id (str): Player ID
            data (pd.DataFrame): DataFrame with all player statistics
            models (dict): Dictionary of trained models
            categories (list): List of categories to forecast
            is_pitcher (bool): Whether the player is a pitcher
            
        Returns:
            dict: Dictionary with forecasted values or None if forecast couldn't be generated
        """
        player_data = data[data['PLAYER_ID'] == player_id]
        return self.generate_player_forecast(player_id, player_data, models, categories, is_pitcher)
    
    def load_models(self, models_dir='data/models'):
        """
        Load existing models from the specified directory.
        
        Args:
            models_dir (str): Directory containing saved models
            
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        if not os.path.exists(models_dir):
            print(f"Models directory not found: {models_dir}")
            return False
        
        # Load batting models
        batting_models_loaded = 0
        for category in self.batting_categories:
            model_path = os.path.join(models_dir, f'batting_{category}_ensemble.joblib')
            if os.path.exists(model_path):
                try:
                    self.batting_models[category] = joblib.load(model_path)
                    batting_models_loaded += 1
                except Exception as e:
                    print(f"Error loading batting model for {category}: {e}")
        
        # Load pitching models
        pitching_models_loaded = 0
        for category in self.pitching_categories:
            model_path = os.path.join(models_dir, f'pitching_{category}_ensemble.joblib')
            if os.path.exists(model_path):
                try:
                    self.pitching_models[category] = joblib.load(model_path)
                    pitching_models_loaded += 1
                except Exception as e:
                    print(f"Error loading pitching model for {category}: {e}")
        
        print(f"Loaded {batting_models_loaded}/{len(self.batting_categories)} batting models and "
              f"{pitching_models_loaded}/{len(self.pitching_categories)} pitching models")
        
        return batting_models_loaded > 0 and pitching_models_loaded > 0
    
    def generate_all_forecasts(self, output_dir='data/projections', n_jobs=8):
        """
        Generate forecasts for all players in parallel.
        
        Args:
            output_dir (str): Directory to save forecasts
            n_jobs (int): Number of parallel jobs to run (default: 8)
            
        Returns:
            tuple: (batting_forecasts, pitching_forecasts)
        """
        if self.batting_data is None or self.pitching_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not self.batting_models or not self.pitching_models:
            if self.load_existing_models:
                # Try to load models if not already loaded
                if not self.load_models():
                    raise ValueError("Failed to load existing models and no models are trained. "
                                    "Call train_batting_models() and train_pitching_models() first.")
            else:
                raise ValueError("Models not trained. Call train_batting_models() and train_pitching_models() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate batting forecasts in parallel
        print(f"Generating batting forecasts in parallel using {n_jobs} cores...")
        batting_player_ids = self.batting_data['PLAYER_ID'].unique()
        
        batting_forecasts = Parallel(n_jobs=n_jobs)(
            delayed(self._generate_player_forecast_wrapper)(
                player_id, self.batting_data, self.batting_models, self.batting_categories, False
            )
            for player_id in tqdm(batting_player_ids, desc="Forecasting batters")
        )
        
        # Filter out None values
        batting_forecasts = [f for f in batting_forecasts if f is not None]
        
        batting_df = pd.DataFrame(batting_forecasts)
        batting_df.to_csv(os.path.join(output_dir, 'batting_forecasts.csv'), index=False)
        
        # Generate pitching forecasts in parallel
        print(f"Generating pitching forecasts in parallel using {n_jobs} cores...")
        pitching_player_ids = self.pitching_data['PLAYER_ID'].unique()
        
        pitching_forecasts = Parallel(n_jobs=n_jobs)(
            delayed(self._generate_player_forecast_wrapper)(
                player_id, self.pitching_data, self.pitching_models, self.pitching_categories, True
            )
            for player_id in tqdm(pitching_player_ids, desc="Forecasting pitchers")
        )
        
        # Filter out None values
        pitching_forecasts = [f for f in pitching_forecasts if f is not None]
        
        pitching_df = pd.DataFrame(pitching_forecasts)
        pitching_df.to_csv(os.path.join(output_dir, 'pitching_forecasts.csv'), index=False)
        
        print(f"Forecasts generated for {len(batting_forecasts)} batters and {len(pitching_forecasts)} pitchers")
        print(f"Forecasts saved to {output_dir}")
        
        return batting_df, pitching_df


if __name__ == "__main__":
    # Example usage
    print("Player Performance Forecaster")
    print("============================")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate player performance forecasts')
    parser.add_argument('--load-models', action='store_true', help='Load existing models instead of training new ones')
    parser.add_argument('--models-dir', default='data/models', help='Directory containing saved models (when using --load-models)')
    parser.add_argument('--n-jobs', type=int, default=8, help='Number of parallel jobs to run')
    args = parser.parse_args()
    
    # Number of cores to use for parallel processing
    n_jobs = args.n_jobs
    
    # Initialize forecaster
    forecaster = PlayerForecaster(load_existing_models=args.load_models)
    
    # Load data
    forecaster.load_bio_data(bio_file='data/raw/biofile0.csv')
    
    forecaster.load_data(
        batting_file='data/processed/batting_stats_all.csv',
        pitching_file='data/processed/pitching_stats_all.csv'
    )
    
    if args.load_models:
        print(f"\nLoading existing models from {args.models_dir}...")
        if not forecaster.load_models(args.models_dir):
            print("Failed to load some models. Training may be required.")
    else:
        # Train models with parallel processing
        print("\nTraining batting models...")
        forecaster.train_batting_models(n_jobs=n_jobs)
        
        print("\nTraining pitching models...")
        forecaster.train_pitching_models(n_jobs=n_jobs)
    
    # Generate forecasts with parallel processing
    print("\nGenerating forecasts...")
    batting_forecasts, pitching_forecasts = forecaster.generate_all_forecasts(n_jobs=n_jobs)
    
    print("\nForecasting complete!")
