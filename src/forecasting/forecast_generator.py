#!/usr/bin/env python3
"""
Forecast Generation Module for Player Performance Forecasting

This module provides functions for generating forecasts for
player performance in fantasy baseball.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from src.forecasting.data_preparation import (
    BATTING_CATEGORIES, 
    PITCHING_CATEGORIES,
    prepare_player_data,
    calculate_age,
    compute_historical_averages
)
from src.forecasting.model_trainer import (
    forecast_arima,
    forecast_exponential_smoothing
)


def generate_player_forecast(player_id, player_data, models, categories, bio_data, is_pitcher=False, forecast_season=None):
    """
    Generate forecast for a specific player.
    
    Args:
        player_id (str): Player ID
        player_data (pd.DataFrame): DataFrame with player's historical data
        models (dict): Dictionary of trained models
        categories (list): List of categories to forecast
        bio_data pd.DataFrame: DataFrame with player biographical data
        is_pitcher (bool): Whether the player is a pitcher
        forecast_season (int, optional): Season to forecast for. If None, uses most recent season + 1
        
    Returns:
        dict: Dictionary with forecasted values
    """
    if len(player_data) < 1:
        return None
    
    # Set forecast season
    if forecast_season is None:
        # Get the most recent season data
        recent_data = player_data.sort_values('SEASON', ascending=False).iloc[0]
        forecast_season = recent_data['SEASON'] + 1  # Next season
    else:
        # For backtesting: filter out data from seasons after the forecast_season
        player_data = player_data[player_data['SEASON'] < forecast_season]
        
        # If no data left after filtering, return None
        if len(player_data) < 1:
            return None
        
        # Get the most recent season data (before the forecast season)
        recent_data = player_data.sort_values('SEASON', ascending=False).iloc[0]
    
    # Check if the most recent season is the season immediately before the forecast season
    if recent_data['SEASON'] != forecast_season - 1:
        return None
    
    # Check if player had non-zero data in the previous season
    has_non_zero_data = False
    if is_pitcher:
        # For pitchers, check if they had meaningful pitching stats
        if any(recent_data.get(cat, 0) > 0 for cat in ['K', 'W', 'QS', 'SV', 'HLD']):
            has_non_zero_data = True
    else:
        # For batters, check if they had meaningful batting stats
        if any(recent_data.get(cat, 0) > 0 for cat in ['HR', 'R', 'RBI', 'SB', 'TB']):
            has_non_zero_data = True
    
    if not has_non_zero_data:
        return None
    
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
    else:
        # If bio data is available, calculate age from birthdate
        player_id_lower = player_id.lower()
        player_bio = bio_data[bio_data['id'] == player_id_lower]
        
        if not player_bio.empty and not pd.isna(player_bio.iloc[0]['birthdate']):
            birthdate = player_bio.iloc[0]['birthdate']
            age = calculate_age(birthdate, forecast_season)
        else:
            print("Encountered missing player bio", player_id)
            age = 28  # MLB average age for missing data
    
    # Compute historical averages (3 and 5 seasons)
    avg_features = compute_historical_averages(player_data)
    
    # Add historical averages to the feature set
    for feat_name, feat_value in avg_features.items():
        features[feat_name] = feat_value
    
    # Add age-related features
    features['AGE'] = age
    features['AGE_SQUARED'] = age ** 2
    features['EARLY_CAREER'] = 1 if age < 26 else 0
    features['PRIME'] = 1 if 26 <= age <= 32 else 0
    features['DECLINE'] = 1 if age > 32 else 0
    
    # Get player name from bio data if available
    player_name = None
    player_id_lower = player_id.lower()
    player_bio = bio_data[bio_data['id'] == player_id_lower]
    if not player_bio.empty:
        player_name = player_bio.iloc[0]['fullname']
    
    # Generate forecasts for each category
    forecasts = {
        'PLAYER_ID': player_id,
        'SEASON': forecast_season
    }
    
    # Add player name if available
    if player_name:
        forecasts['PLAYER_NAME'] = player_name
    
    for category in categories:
        if category not in models:
            continue
        
        # Get time series data for ARIMA and Exponential Smoothing
        time_series = prepare_player_data(
            player_data, player_id, category, min_seasons=3
        )
        
        # Initialize forecast values
        arima_forecast = None
        exp_smooth_forecast = None
        regression_forecasts = {}
        
        # ARIMA forecast
        if time_series is not None and len(time_series) >= 3:
            arima_result, _ = forecast_arima(time_series)
            if arima_result is not None:
                arima_forecast = arima_result[0]
        
        # Exponential Smoothing forecast
        if time_series is not None and len(time_series) >= 3:
            exp_smooth_result, _ = forecast_exponential_smoothing(time_series)
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
        
        # Combine forecasts with improved weighting and outlier handling
        valid_forecasts = []
        forecast_sources = {}
        
        if arima_forecast is not None:
            valid_forecasts.append(arima_forecast)
            forecast_sources['arima'] = arima_forecast
        
        if exp_smooth_forecast is not None:
            valid_forecasts.append(exp_smooth_forecast)
            forecast_sources['exp_smooth'] = exp_smooth_forecast
        
        for model_name, forecast in regression_forecasts.items():
            valid_forecasts.append(forecast)
            forecast_sources[model_name] = forecast
        
        if valid_forecasts:
            # For players with limited history, adjust the weights
            if len(player_data) < 3:
                # Put more weight on regression models for players with limited history
                weights = {
                    'arima': 0.1,
                    'exp_smooth': 0.1,
                    'ridge': 0.25,
                    'random_forest': 0.25,
                    'gradient_boosting': 0.3
                }
                
                # Calculate weighted average
                weighted_sum = 0
                total_weight = 0
                
                for source, forecast in forecast_sources.items():
                    if source in weights:
                        weighted_sum += weights[source] * forecast
                        total_weight += weights[source]
                
                if total_weight > 0:
                    combined_forecast = weighted_sum / total_weight
                else:
                    # Fall back to simple average if no weights match
                    combined_forecast = np.mean(valid_forecasts)
            else:
                # For players with sufficient history
                # Remove extreme outliers (more than 3 std devs from mean) if we have enough forecasts
                if len(valid_forecasts) > 2:
                    mean_forecast = np.mean(valid_forecasts)
                    std_forecast = np.std(valid_forecasts)
                    filtered_forecasts = [f for f in valid_forecasts 
                                        if abs(f - mean_forecast) <= 3 * std_forecast]
                    
                    # Only use filtered forecasts if we didn't filter everything out
                    if filtered_forecasts:
                        valid_forecasts = filtered_forecasts
                
                # Use median for robustness against outliers
                combined_forecast = np.median(valid_forecasts)
            
            # For ERA and WHIP, we want to ensure they don't go below a reasonable threshold
            if category in ['ERA', 'WHIP'] and is_pitcher:
                combined_forecast = max(combined_forecast, 1.0)  # Minimum ERA/WHIP of 1.0
            else:
                combined_forecast = max(combined_forecast, 0)  # Ensure non-negative
            
            # Store individual model forecasts in the output dictionary
            for model_name, forecast in forecast_sources.items():
                # Apply the same non-negative constraint to individual forecasts
                if category in ['ERA', 'WHIP'] and is_pitcher:
                    forecast = max(forecast, 1.0)  # Minimum ERA/WHIP of 1.0
                else:
                    forecast = max(forecast, 0)  # Ensure non-negative
                
                forecasts[f"{category}_{model_name}"] = forecast
            
            # Store the combined forecast
            forecasts[category] = combined_forecast
        else:
            # If no valid forecasts, use the most recent value
            forecasts[category] = recent_data[category]
    
    return forecasts


def _generate_player_forecast_wrapper(player_id, data, models, categories, bio_data, is_pitcher=False, forecast_season=None):
    """
    Wrapper for generate_player_forecast to use with joblib.Parallel.
    
    Args:
        player_id (str): Player ID
        data (pd.DataFrame): DataFrame with all player statistics
        models (dict): Dictionary of trained models
        categories (list): List of categories to forecast
        bio_data (pd.DataFrame, optional): DataFrame with player biographical data
        is_pitcher (bool): Whether the player is a pitcher
        forecast_season (int, optional): Season to forecast for
        
    Returns:
        dict: Dictionary with forecasted values or None if forecast couldn't be generated
    """
    player_data = data[data['PLAYER_ID'] == player_id]
    return generate_player_forecast(player_id, player_data, models, categories, bio_data, is_pitcher, forecast_season)


def generate_all_forecasts(batting_data, pitching_data, batting_models, pitching_models, 
                          bio_data, output_dir='data/projections', n_jobs=15, forecast_season=None):
    """
    Generate forecasts for all players in parallel.
    
    Args:
        batting_data (pd.DataFrame): DataFrame with batting statistics
        pitching_data (pd.DataFrame): DataFrame with pitching statistics
        batting_models (dict): Dictionary of trained batting models
        pitching_models (dict): Dictionary of trained pitching models
        bio_data (pd.DataFrame, optional): DataFrame with player biographical data
        output_dir (str): Directory to save forecasts
        n_jobs (int): Number of parallel jobs to run (default: 15)
        forecast_season (int, optional): Season to forecast for
        
    Returns:
        tuple: (batting_forecasts, pitching_forecasts)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate batting forecasts in parallel
    print(f"Generating batting forecasts in parallel using {n_jobs} cores...")
    batting_player_ids = batting_data['PLAYER_ID'].unique()
    
    batting_forecasts = Parallel(n_jobs=n_jobs)(
        delayed(_generate_player_forecast_wrapper)(
            player_id, batting_data, batting_models, BATTING_CATEGORIES, bio_data, False, forecast_season
        )
        for player_id in tqdm(batting_player_ids, desc="Forecasting batters")
    )
    
    # Filter out None values
    batting_forecasts = [f for f in batting_forecasts if f is not None]
    
    batting_df = pd.DataFrame(batting_forecasts)
    batting_df.to_csv(os.path.join(output_dir, 'batting_forecasts.csv'), index=False)
    
    # Generate pitching forecasts in parallel
    print(f"Generating pitching forecasts in parallel using {n_jobs} cores...")
    pitching_player_ids = pitching_data['PLAYER_ID'].unique()
    
    pitching_forecasts = Parallel(n_jobs=n_jobs)(
        delayed(_generate_player_forecast_wrapper)(
            player_id, pitching_data, pitching_models, PITCHING_CATEGORIES, bio_data, True, forecast_season
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
