#!/usr/bin/env python3
"""
Model Training Module for Player Performance Forecasting

This module provides functions for training and saving models for
player performance forecasting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import contextlib

from src.forecasting.data_preparation import BATTING_CATEGORIES, PITCHING_CATEGORIES


@contextlib.contextmanager
def suppress_statsmodels_warnings():
    """Context manager to suppress specific statsmodels warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                               message="divide by zero encountered in log")
        warnings.filterwarnings("ignore", category=RuntimeWarning, 
                               message="invalid value encountered in scalar add")
        warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)
        yield


def validate_time_series(time_series, min_length=3, min_variance=1e-6):
    """
    Validate time series data before fitting complex models.
    
    Args:
        time_series (np.ndarray): Time series data
        min_length (int): Minimum required length for complex models
        min_variance (float): Minimum required variance
        
    Returns:
        str: 'valid' if time series is valid for complex models,
             'simple' if only simple models should be used,
             'invalid' if the data cannot be used at all
    """
    if time_series is None:
        return 'invalid'
    
    # Check for NaN or infinite values
    if np.any(np.isnan(time_series)) or np.any(np.isinf(time_series)):
        return 'invalid'
    
    # For very short time series, use simple models
    if len(time_series) < min_length:
        if len(time_series) >= 2:
            return 'simple'  # Use simple models for 2 data points
        else:
            return 'invalid'  # Can't use time series with just 1 point
    
    # Check for sufficient variance
    if np.var(time_series) < min_variance:
        return 'simple'  # Use simple models for low variance data
    
    return 'valid'  # Data is valid for complex models


def forecast_simple(time_series, forecast_periods=1):
    """
    Simple forecasting for limited data (1-2 seasons).
    
    Args:
        time_series (np.ndarray): Time series data
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        tuple: (forecast, None)
    """
    if len(time_series) == 0:
        return None, None
    
    if len(time_series) == 1:
        # For a single data point, just repeat it
        return np.array([time_series[0]] * forecast_periods), None
    
    # For two data points, use simple trend
    if len(time_series) == 2:
        trend = time_series[1] - time_series[0]
        # Dampen the trend for stability
        damping_factor = 0.5
        forecast = time_series[-1] + damping_factor * trend * np.arange(1, forecast_periods + 1)
        return forecast, None
    
    # For more points but still using simple method, use average of last 2 seasons
    return np.array([np.mean(time_series[-2:])] * forecast_periods), None


def forecast_arima(time_series, forecast_periods=1, order=(1, 0, 0)):
    """
    Forecast using ARIMA model with improved validation and fallback to simple methods.
    
    Args:
        time_series (np.ndarray): Time series data
        forecast_periods (int): Number of periods to forecast
        order (tuple): ARIMA order (p, d, q)
        
    Returns:
        tuple: (forecast, model)
    """
    validation_result = validate_time_series(time_series)
    
    if validation_result == 'invalid':
        return None, None
    
    if validation_result == 'simple':
        # Fall back to simple forecasting for limited data
        return forecast_simple(time_series, forecast_periods)
    
    # Check if all values are identical (no variation)
    if np.all(time_series == time_series[0]):
        # Return the constant value as forecast
        return np.array([time_series[0]] * forecast_periods), None
    
    # Check for stationarity and adjust differencing if needed
    d = order[1]
    if d == 0 and len(time_series) >= 10:
        # Simple stationarity check using rolling mean
        rolling_mean = np.abs(np.diff(np.mean(
            [time_series[i:i+5] for i in range(len(time_series)-5)], 
            axis=1
        )))
        if np.max(rolling_mean) > 0.1 * np.mean(time_series):
            # Data might be non-stationary, use d=1
            order = (order[0], 1, order[2])
    
    try:
        with suppress_statsmodels_warnings():
            # Set convergence options
            model = ARIMA(time_series, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_periods)
        return forecast, model_fit
    except Exception as e:
        # More detailed error logging
        print(f"ARIMA error: {type(e).__name__}: {e}")
        return None, None


def forecast_exponential_smoothing(time_series, forecast_periods=1):
    """
    Forecast using Exponential Smoothing with improved validation and fallback to simple methods.
    
    Args:
        time_series (np.ndarray): Time series data
        forecast_periods (int): Number of periods to forecast
        
    Returns:
        tuple: (forecast, model)
    """
    validation_result = validate_time_series(time_series)
    
    if validation_result == 'invalid':
        return None, None
    
    if validation_result == 'simple':
        # Fall back to simple forecasting for limited data
        return forecast_simple(time_series, forecast_periods)
    
    # Check if all values are identical (no variation)
    if np.all(time_series == time_series[0]):
        # Return the constant value as forecast
        return np.array([time_series[0]] * forecast_periods), None
    
    try:
        with suppress_statsmodels_warnings():
            # Add damped_trend for more stability
            model = ExponentialSmoothing(
                time_series, 
                trend='add', 
                damped_trend=True,  # Add damping to stabilize forecasts
                seasonal=None
            )
            
            # Use more robust optimization settings
            model_fit = model.fit(
                optimized=True,
                method='L-BFGS-B',  # More robust optimization method
                use_brute=False,    # Start with L-BFGS-B directly
                remove_bias=True    # Can help with convergence
            )
            
            forecast = model_fit.forecast(forecast_periods)
        return forecast, model_fit
    except Exception as e:
        # More detailed error logging
        print(f"Exponential smoothing error: {type(e).__name__}: {e}")
        return None, None


def train_regression_model(data, target, features, model_type='ridge'):
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
        model = RandomForestRegressor(n_estimators=100, random_state=42)
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


def train_category_model(category, regression_data, features_prefix='batting', model_type='ensemble'):
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
        ridge_model, _ = train_regression_model(regression_data, category, features, 'ridge')
        rf_model, _ = train_regression_model(regression_data, category, features, 'random_forest')
        gb_model, importance = train_regression_model(regression_data, category, features, 'gradient_boosting')
        
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
        model, importance = train_regression_model(regression_data, category, features, model_type)
        
        model_dict = {
            model_type: model,
            'feature_importance': importance
        }
        
        # Save model
        joblib.dump(model_dict, f'data/models/{features_prefix}_{category}_{model_type}.joblib')
    
    return category, model_dict


def train_batting_models(batting_data, bio_data=None, model_type='ensemble', n_jobs=8):
    """
    Train models for forecasting batting statistics in parallel.
    
    Args:
        batting_data (pd.DataFrame): DataFrame with batting statistics
        bio_data (pd.DataFrame, optional): DataFrame with player biographical data
        model_type (str): Type of model to train
        n_jobs (int): Number of parallel jobs to run (default: 8)
        
    Returns:
        dict: Dictionary of trained models
    """
    from src.forecasting.data_preparation import prepare_regression_features
    
    print("Preparing batting data for modeling...")
    regression_data = prepare_regression_features(batting_data, bio_data)
    
    # Create output directory for models
    os.makedirs('data/models', exist_ok=True)
    
    print(f"Training batting models in parallel using {n_jobs} cores...")
    
    # Train models in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_category_model)(
            category, regression_data, 'batting', model_type
        ) 
        for category in BATTING_CATEGORIES
    )
    
    # Collect results
    models = {category: model_dict for category, model_dict in results}
    
    print(f"Completed training models for {len(models)} batting categories")
    return models


def train_pitching_models(pitching_data, bio_data=None, model_type='ensemble', n_jobs=8):
    """
    Train models for forecasting pitching statistics in parallel.
    
    Args:
        pitching_data (pd.DataFrame): DataFrame with pitching statistics
        bio_data (pd.DataFrame, optional): DataFrame with player biographical data
        model_type (str): Type of model to train
        n_jobs (int): Number of parallel jobs to run (default: 8)
        
    Returns:
        dict: Dictionary of trained models
    """
    from src.forecasting.data_preparation import prepare_regression_features
    
    print("Preparing pitching data for modeling...")
    regression_data = prepare_regression_features(pitching_data, bio_data)
    
    # Create output directory for models
    os.makedirs('data/models', exist_ok=True)
    
    print(f"Training pitching models in parallel using {n_jobs} cores...")
    
    # Train models in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_category_model)(
            category, regression_data, 'pitching', model_type
        ) 
        for category in PITCHING_CATEGORIES
    )
    
    # Collect results
    models = {category: model_dict for category, model_dict in results}
    
    print(f"Completed training models for {len(models)} pitching categories")
    return models


def load_models(batting_categories=BATTING_CATEGORIES, pitching_categories=PITCHING_CATEGORIES, models_dir='data/models'):
    """
    Load existing models from the specified directory.
    
    Args:
        batting_categories (list): List of batting categories
        pitching_categories (list): List of pitching categories
        models_dir (str): Directory containing saved models
        
    Returns:
        tuple: (batting_models, pitching_models) Dictionaries of loaded models
    """
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return {}, {}
    
    # Load batting models
    batting_models = {}
    batting_models_loaded = 0
    for category in batting_categories:
        model_path = os.path.join(models_dir, f'batting_{category}_ensemble.joblib')
        if os.path.exists(model_path):
            try:
                batting_models[category] = joblib.load(model_path)
                batting_models_loaded += 1
            except Exception as e:
                print(f"Error loading batting model for {category}: {e}")
    
    # Load pitching models
    pitching_models = {}
    pitching_models_loaded = 0
    for category in pitching_categories:
        model_path = os.path.join(models_dir, f'pitching_{category}_ensemble.joblib')
        if os.path.exists(model_path):
            try:
                pitching_models[category] = joblib.load(model_path)
                pitching_models_loaded += 1
            except Exception as e:
                print(f"Error loading pitching model for {category}: {e}")
    
    print(f"Loaded {batting_models_loaded}/{len(batting_categories)} batting models and "
          f"{pitching_models_loaded}/{len(pitching_categories)} pitching models")
    
    return batting_models, pitching_models
