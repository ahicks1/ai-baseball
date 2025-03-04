#!/usr/bin/env python3
"""
Data Preparation Module for Player Performance Forecasting

This module provides functions for loading and preparing data for
player performance forecasting.
"""

import pandas as pd
import numpy as np

# League settings for H2H categories
BATTING_CATEGORIES = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
PITCHING_CATEGORIES = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']


def load_bio_data(bio_file='data/raw/biofile0.csv'):
    """
    Load player biographical data from CSV file.
    
    Args:
        bio_file (str): Path to biographical data CSV
        
    Returns:
        pd.DataFrame: DataFrame with player biographical data
    """
    print(f"Loading biographical data from {bio_file}")
    bio_data = pd.read_csv(bio_file)
    print(f"Loaded biographical data for {len(bio_data)} players")
    return bio_data


def calculate_age(birthdate, season):
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


def load_data(batting_file, pitching_file):
    """
    Load batting and pitching data from CSV files.
    
    Args:
        batting_file (str): Path to batting statistics CSV
        pitching_file (str): Path to pitching statistics CSV
        
    Returns:
        tuple: (batting_data, pitching_data) DataFrames with player statistics
    """
    print(f"Loading batting data from {batting_file}")
    batting_data = pd.read_csv(batting_file)
    batting_data = batting_data[batting_data['SEASON'] != 2020]
    
    print(f"Loading pitching data from {pitching_file}")
    pitching_data = pd.read_csv(pitching_file)
    pitching_data = pitching_data[pitching_data['SEASON'] != 2020]
    
    print(f"Loaded {len(batting_data)} batting records and {len(pitching_data)} pitching records")
    return batting_data, pitching_data


def prepare_player_data(data, player_id, category, min_seasons=3):
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


def compute_historical_averages(player_data, exclude_cols=None, season_col='SEASON'):
    """
    Compute historical averages of player statistics over the past 3 and 5 seasons.
    
    Args:
        player_data (pd.DataFrame): DataFrame with player statistics sorted by season
        exclude_cols (list, optional): Columns to exclude from averaging
        season_col (str): Column name for season
        
    Returns:
        dict: Dictionary with averaged features for past 3 and 5 seasons
    """
    if exclude_cols is None:
        exclude_cols = ['PLAYER_ID', 'SEASON', 'TEAM', 'AGE', 'first', 'last']
    
    # Create a copy of the data sorted by season
    player_data = player_data.sort_values(season_col)
    
    # Initialize the result dictionary
    avg_features = {}
    
    # Get the most recent season
    if len(player_data) == 0:
        return avg_features
    
    # For each column that's not in exclude_cols
    for col in player_data.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(player_data[col]):
            # Calculate 3-season average
            if len(player_data) >= 3:
                avg_3 = player_data.iloc[-3:][col].mean()
                avg_features[f'AVG3_{col}'] = avg_3
            elif len(player_data) > 0:
                # Use whatever data is available
                avg_3 = player_data[col].mean()
                avg_features[f'AVG3_{col}'] = avg_3
            
            # Calculate 5-season average
            if len(player_data) >= 5:
                avg_5 = player_data.iloc[-5:][col].mean()
                avg_features[f'AVG5_{col}'] = avg_5
            elif len(player_data) > 0:
                # Use whatever data is available
                avg_5 = player_data[col].mean()
                avg_features[f'AVG5_{col}'] = avg_5
    
    return avg_features


def prepare_regression_features(data: pd.DataFrame, bio_data, player_id_col='PLAYER_ID', season_col='SEASON'):
    """
    Prepare features for regression models.
    
    Args:
        data (pd.DataFrame): DataFrame with player statistics
        bio_data (pd.DataFrame, optional): DataFrame with player biographical data
        player_id_col (str): Column name for player ID
        season_col (str): Column name for season
        
    Returns:
        pd.DataFrame: DataFrame with regression features
    """
    # Create a copy of the data
    df = data.copy()
    
    # Define columns to exclude from features (non-numeric columns)
    exclude_cols = ['TEAM', 'first', 'last']
    
    # Create a mapping of player_id to birthdate
    player_birthdate = dict(zip(bio_data['id'], bio_data['birthdate']))
    
    # Calculate age for each player-season combination
    df['AGE'] = df.apply(
        lambda row: calculate_age(
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
            
            # Compute historical averages (3 and 5 seasons)
            # Use data up to the previous season to avoid data leakage
            historical_data = player_data.iloc[:i]
            avg_features = compute_historical_averages(historical_data, exclude_cols=[player_id_col, season_col, 'AGE'] + exclude_cols)
            
            # Add historical averages to the feature set
            for feat_name, feat_value in avg_features.items():
                row[feat_name] = feat_value
            
            # Add age-related features
            age = row['AGE']
            row['AGE_SQUARED'] = age ** 2
            
            # Age indicators for different career phases
            row['EARLY_CAREER'] = 1 if age < 26 else 0
            row['PRIME'] = 1 if 26 <= age <= 32 else 0
            row['DECLINE'] = 1 if age > 32 else 0
            
            all_rows.append(row)
    
    return pd.DataFrame(all_rows)
