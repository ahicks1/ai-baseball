#!/usr/bin/env python3
"""
Retrosheet Data Processing Module

This module handles the processing of Retrosheet CSV data files,
converting them into player statistics for fantasy baseball analysis.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import patch
from src.data_processing.pitching_stats import integrate_advanced_stats


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


def load_players_file(file_path: str) -> pd.DataFrame:
    """
    Load player information from allplayers.csv.
    
    Args:
        file_path (str): Path to the allplayers.csv file
        
    Returns:
        pd.DataFrame: DataFrame containing player information
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    print(f"Loading players file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Players file not found: {file_path}")
    
    # Load the CSV file
    players_df = pd.read_csv(file_path)
    
    # Convert date columns to datetime if they exist
    if 'first_g' in players_df.columns:
        players_df['first_g'] = pd.to_datetime(players_df['first_g'], format='%Y%m%d', errors='coerce')
    if 'last_g' in players_df.columns:
        players_df['last_g'] = pd.to_datetime(players_df['last_g'], format='%Y%m%d', errors='coerce')
    
    return players_df


def load_batting_file(file_path: str) -> pd.DataFrame:
    """
    Load batting statistics from batting.csv.
    
    Args:
        file_path (str): Path to the batting.csv file
        
    Returns:
        pd.DataFrame: DataFrame containing batting statistics
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    print(f"Loading batting file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Batting file not found: {file_path}")
    
    # Load the CSV file
    batting_df = pd.read_csv(file_path)
    
    # Convert date column to datetime if it exists
    if 'date' in batting_df.columns:
        batting_df['date'] = pd.to_datetime(batting_df['date'], format='%Y%m%d', errors='coerce')
    
    return batting_df


def load_pitching_file(file_path: str) -> pd.DataFrame:
    """
    Load pitching statistics from pitching.csv.
    
    Args:
        file_path (str): Path to the pitching.csv file
        
    Returns:
        pd.DataFrame: DataFrame containing pitching statistics
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    print(f"Loading pitching file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pitching file not found: {file_path}")
    
    # Load the CSV file
    pitching_df = pd.read_csv(file_path)
    
    # Convert date column to datetime if it exists
    if 'date' in pitching_df.columns:
        pitching_df['date'] = pd.to_datetime(pitching_df['date'], format='%Y%m%d', errors='coerce')
    
    return pitching_df


def load_fielding_file(file_path: str) -> pd.DataFrame:
    """
    Load fielding statistics from fielding.csv.
    
    Args:
        file_path (str): Path to the fielding.csv file
        
    Returns:
        pd.DataFrame: DataFrame containing fielding statistics
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    print(f"Loading fielding file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fielding file not found: {file_path}")
    
    # Load the CSV file
    fielding_df = pd.read_csv(file_path)
    
    # Convert date column to datetime if it exists
    if 'date' in fielding_df.columns:
        fielding_df['date'] = pd.to_datetime(fielding_df['date'], format='%Y%m%d', errors='coerce')
    
    return fielding_df


def load_plays_file(file_path: str) -> pd.DataFrame:
    """
    Load play-by-play data from plays.csv.
    
    Args:
        file_path (str): Path to the plays.csv file
        
    Returns:
        pd.DataFrame: DataFrame containing play-by-play data
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    print(f"Loading plays file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plays file not found: {file_path}")
    
    # Load the CSV file
    plays_df = pd.read_csv(file_path)
    
    # Convert date column to datetime if it exists
    if 'date' in plays_df.columns:
        plays_df['date'] = pd.to_datetime(plays_df['date'], format='%Y%m%d', errors='coerce')
    
    return plays_df


def load_plays_file_chunked(file_path: str, start_year: Optional[int] = None, end_year: Optional[int] = None, chunksize: int = 500000) -> pd.DataFrame:
    """
    Load play-by-play data from plays.csv in chunks to reduce memory usage.
    
    Args:
        file_path (str): Path to the plays.csv file
        start_year (int, optional): Start year for filtering data
        end_year (int, optional): End year for filtering data
        chunksize (int): Number of rows to read at a time
        
    Returns:
        pd.DataFrame: DataFrame containing filtered play-by-play data
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    print(f"Loading plays file in chunks: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Plays file not found: {file_path}")
    
    import gc  # Import garbage collector for memory management
    
    filtered_chunks = []
    total_rows_processed = 0
    total_rows_kept = 0
    
    # Read the file in chunks
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
        total_rows_processed += len(chunk)
        
        # Convert date column to datetime if it exists
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'], format='%Y%m%d', errors='coerce')
            
            # Filter by date range if specified
            if start_year is not None or end_year is not None:
                if start_year is not None:
                    chunk = chunk[chunk['date'].dt.year >= start_year]
                if end_year is not None:
                    chunk = chunk[chunk['date'].dt.year <= end_year]
        
        # Only keep the chunk if it has data after filtering
        if not chunk.empty:
            filtered_chunks.append(chunk)
            total_rows_kept += len(chunk)
        
        # Print progress
        if (chunk_num + 1) % 10 == 0:
            print(f"Processed {chunk_num + 1} chunks ({total_rows_processed:,} rows), kept {total_rows_kept:,} rows")
            
        # Force garbage collection after processing each chunk to free memory
        gc.collect()
    
    # Combine all filtered chunks
    if filtered_chunks:
        print(f"Combining {len(filtered_chunks)} filtered chunks with {total_rows_kept:,} total rows")
        plays_df = pd.concat(filtered_chunks, ignore_index=True)
        
        # Force garbage collection after concatenation
        gc.collect()
        
        print(f"Final plays DataFrame shape: {plays_df.shape}")
        return plays_df
    else:
        print("No data matched the filter criteria")
        return pd.DataFrame()


def aggregate_batting_stats(batting_df: pd.DataFrame, players_df: Optional[pd.DataFrame] = None, season: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate game-by-game batting stats to season totals.
    
    Args:
        batting_df (pd.DataFrame): DataFrame containing batting statistics
        players_df (pd.DataFrame, optional): DataFrame containing player information
        season (int, optional): Filter for a specific season
        
    Returns:
        pd.DataFrame: DataFrame with aggregated batting statistics
    """
    print("Aggregating batting statistics")
    
    # Filter by season if provided
    if season is not None and 'date' in batting_df.columns:
        batting_df = batting_df[batting_df['date'].dt.year == season]
    
    # Only include rows where stattype is 'value'
    batting_df = batting_df[batting_df['stattype'] == 'value']
    
    # Group by player ID and team
    grouped = batting_df.groupby(['id', 'team'])
    
    # Aggregate statistics
    batting_stats = grouped.agg({
        'b_pa': 'sum',
        'b_ab': 'sum',
        'b_r': 'sum',
        'b_h': 'sum',
        'b_d': 'sum',
        'b_t': 'sum',
        'b_hr': 'sum',
        'b_rbi': 'sum',
        'b_sh': 'sum',
        'b_sf': 'sum',
        'b_hbp': 'sum',
        'b_w': 'sum',
        'b_iw': 'sum',
        'b_k': 'sum',
        'b_sb': 'sum',
        'b_cs': 'sum',
        'b_gdp': 'sum',
        'gid': 'count'  # Count of games
    }).reset_index()
    
    # Rename columns
    batting_stats = batting_stats.rename(columns={
        'id': 'PLAYER_ID',
        'team': 'TEAM',
        'gid': 'G',
        'b_pa': 'PA',
        'b_ab': 'AB',
        'b_r': 'R',
        'b_h': 'H',
        'b_d': 'D',
        'b_t': 'T',
        'b_hr': 'HR',
        'b_rbi': 'RBI',
        'b_sh': 'SH',
        'b_sf': 'SF',
        'b_hbp': 'HBP',
        'b_w': 'BB',
        'b_iw': 'IBB',
        'b_k': 'SO',
        'b_sb': 'SB',
        'b_cs': 'CS',
        'b_gdp': 'GDP'
    })
    
    # Add season column
    if season is not None:
        batting_stats['SEASON'] = season
    elif 'date' in batting_df.columns:
        # Extract year from the first date for each player
        season_data = batting_df.groupby('id')['date'].min().dt.year
        batting_stats['SEASON'] = batting_stats['PLAYER_ID'].map(season_data)
    else:
        batting_stats['SEASON'] = np.nan
    
    # Add player names if players_df is provided
    if players_df is not None:
        player_names = players_df[['id', 'first', 'last']].rename(columns={'id': 'PLAYER_ID'})
        batting_stats = pd.merge(batting_stats, player_names, on='PLAYER_ID', how='left')
        
        # Add age data if players_df has birthdate information
        if 'birthdate' in players_df.columns:
            # Create a mapping of player_id to birthdate
            player_birthdate = dict(zip(players_df['id'], players_df['birthdate']))
            
            # Calculate age for each player based on the season
            batting_stats['AGE'] = batting_stats.apply(
                lambda row: calculate_age(
                    player_birthdate.get(row['PLAYER_ID'].lower(), None), 
                    row['SEASON']
                ),
                axis=1
            )
    
    return batting_stats


def calculate_advanced_batting_stats(batting_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate advanced batting metrics like OBP, SLG, OPS, and TB.
    
    Args:
        batting_stats (pd.DataFrame): DataFrame with basic batting statistics
        
    Returns:
        pd.DataFrame: DataFrame with additional advanced statistics
    """
    print("Calculating advanced batting statistics")
    
    # Make a copy to avoid modifying the original DataFrame
    stats = batting_stats.copy()
    
    # Calculate Total Bases (TB)
    stats['TB'] = stats['H'] - stats['D'] - stats['T'] - stats['HR'] + (2 * stats['D']) + (3 * stats['T']) + (4 * stats['HR'])
    
    # Calculate Batting Average (AVG)
    stats['AVG'] = stats['H'] / stats['AB']
    
    # Calculate On-Base Percentage (OBP)
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    stats['OBP'] = (stats['H'] + stats['BB'] + stats['HBP']) / (stats['AB'] + stats['BB'] + stats['HBP'] + stats['SF'])
    
    # Calculate Slugging Percentage (SLG)
    stats['SLG'] = stats['TB'] / stats['AB']
    
    # Calculate On-Base Plus Slugging (OPS)
    stats['OPS'] = stats['OBP'] + stats['SLG']
    
    # Handle division by zero
    stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return stats


def aggregate_pitching_stats(pitching_df: pd.DataFrame, players_df: Optional[pd.DataFrame] = None, season: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate game-by-game pitching stats to season totals.
    
    Args:
        pitching_df (pd.DataFrame): DataFrame containing pitching statistics
        players_df (pd.DataFrame, optional): DataFrame containing player information
        season (int, optional): Filter for a specific season
        
    Returns:
        pd.DataFrame: DataFrame with aggregated pitching statistics
    """
    print("Aggregating pitching statistics")
    
    # Filter by season if provided
    if season is not None and 'date' in pitching_df.columns:
        pitching_df = pitching_df[pitching_df['date'].dt.year == season]
    
    # Only include rows where stattype is 'value'
    pitching_df = pitching_df[pitching_df['stattype'] == 'value']
    
    # Group by player ID and team
    grouped = pitching_df.groupby(['id', 'team'])
    
    # Aggregate statistics
    pitching_stats = grouped.agg({
        'p_ipouts': 'sum',
        'p_bfp': 'sum',
        'p_h': 'sum',
        'p_d': 'sum',
        'p_t': 'sum',
        'p_hr': 'sum',
        'p_r': 'sum',
        'p_er': 'sum',
        'p_w': 'sum',
        'p_iw': 'sum',
        'p_k': 'sum',
        'p_hbp': 'sum',
        'p_wp': 'sum',
        'p_bk': 'sum',
        'p_gs': 'sum',
        'p_gf': 'sum',
        'p_cg': 'sum',
        'wp': 'sum',
        'lp': 'sum',
        'save': 'sum',
        'gid': 'count'  # Count of games
    }).reset_index()
    
    # Rename columns
    pitching_stats = pitching_stats.rename(columns={
        'id': 'PLAYER_ID',
        'team': 'TEAM',
        'gid': 'G',
        'p_gs': 'GS',
        'p_gf': 'GF',
        'p_cg': 'CG',
        'wp': 'W',
        'lp': 'L',
        'save': 'SV',
        'p_ipouts': 'IPouts',
        'p_bfp': 'BFP',
        'p_h': 'H',
        'p_d': 'D',
        'p_t': 'T',
        'p_hr': 'HR',
        'p_r': 'R',
        'p_er': 'ER',
        'p_w': 'BB',
        'p_iw': 'IBB',
        'p_k': 'K',
        'p_hbp': 'HBP',
        'p_wp': 'WP',
        'p_bk': 'BK'
    })
    
    # Add season column
    if season is not None:
        pitching_stats['SEASON'] = season
    elif 'date' in pitching_df.columns:
        # Extract year from the first date for each player
        season_data = pitching_df.groupby('id')['date'].min().dt.year
        pitching_stats['SEASON'] = pitching_stats['PLAYER_ID'].map(season_data)
    else:
        pitching_stats['SEASON'] = np.nan
    
    # Add player names if players_df is provided
    if players_df is not None:
        player_names = players_df[['id', 'first', 'last']].rename(columns={'id': 'PLAYER_ID'})
        pitching_stats = pd.merge(pitching_stats, player_names, on='PLAYER_ID', how='left')
        
        # Add age data if players_df has birthdate information
        if 'birthdate' in players_df.columns:
            # Create a mapping of player_id to birthdate
            player_birthdate = dict(zip(players_df['id'], players_df['birthdate']))
            
            # Calculate age for each player based on the season
            pitching_stats['AGE'] = pitching_stats.apply(
                lambda row: calculate_age(
                    player_birthdate.get(row['PLAYER_ID'].lower(), None), 
                    row['SEASON']
                ),
                axis=1
            )
    
    return pitching_stats


def calculate_advanced_pitching_stats(pitching_stats: pd.DataFrame, plays_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate advanced pitching metrics like ERA, WHIP, and fantasy-relevant stats.
    
    Args:
        pitching_stats (pd.DataFrame): DataFrame with basic pitching statistics
        plays_df (pd.DataFrame, optional): DataFrame with play-by-play data for calculating holds and quality starts
        
    Returns:
        pd.DataFrame: DataFrame with additional advanced statistics
    """
    print("Calculating advanced pitching statistics")
    
    # Make a copy to avoid modifying the original DataFrame
    stats = pitching_stats.copy()
    
    # Calculate Innings Pitched (IP)
    stats['IP'] = stats['IPouts'] / 3
    
    # Calculate Earned Run Average (ERA)
    # ERA = (ER * 9) / IP
    stats['ERA'] = (stats['ER'] * 9) / stats['IP']
    
    # Calculate Walks plus Hits per Inning Pitched (WHIP)
    # WHIP = (BB + H) / IP
    stats['WHIP'] = (stats['BB'] + stats['H']) / stats['IP']
    
    # Calculate Quality Starts (QS) and Holds (HLD)
    if plays_df is not None:
        # Use the specialized module to calculate QS and HLD from play-by-play data
        stats = integrate_advanced_stats(stats, plays_df)
    else:
        # If play-by-play data is not available, use estimates
        # Estimate QS based on overall performance
        stats['QS'] = np.floor(stats['GS'] * 0.65)  # Rough estimate: 65% of starts are quality starts
        # Set HLD to 0 as we can't estimate without play-by-play data
        stats['HLD'] = 0
        # Calculate combined fantasy stats
        stats['SV+HLD'] = stats['SV'] + stats['HLD']
        stats['W+QS'] = stats['W'] + stats['QS']
    
    # Handle division by zero
    stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return stats


def process_season_data_with_dataframes(
    season: int, 
    players_df: pd.DataFrame, 
    batting_df: pd.DataFrame, 
    pitching_df: pd.DataFrame, 
    fielding_df: Optional[pd.DataFrame] = None, 
    plays_df: Optional[pd.DataFrame] = None, 
    output_dir: str = 'data/processed'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process data for a given MLB season using pre-loaded DataFrames.
    
    Args:
        season (int): The MLB season year to process
        players_df (pd.DataFrame): DataFrame containing player information
        batting_df (pd.DataFrame): DataFrame containing batting statistics
        pitching_df (pd.DataFrame): DataFrame containing pitching statistics
        fielding_df (pd.DataFrame, optional): DataFrame containing fielding statistics
        plays_df (pd.DataFrame, optional): DataFrame containing play-by-play data
        output_dir (str): Directory to save processed data
        
    Returns:
        tuple: (batting_stats_df, pitching_stats_df)
    """
    print(f"Processing data for {season} season")
    
    # Process batting statistics
    batting_stats = aggregate_batting_stats(batting_df, players_df, season)
    batting_stats = calculate_advanced_batting_stats(batting_stats)
    
    # Process pitching statistics
    pitching_stats = aggregate_pitching_stats(pitching_df, players_df, season)
    pitching_stats = calculate_advanced_pitching_stats(pitching_stats, plays_df)
    
    # Remove duplicate rows
    batting_stats = batting_stats.drop_duplicates()
    pitching_stats = pitching_stats.drop_duplicates()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data
    batting_file = os.path.join(output_dir, f'batting_stats_{season}.csv')
    pitching_file = os.path.join(output_dir, f'pitching_stats_{season}.csv')
    
    batting_stats.to_csv(batting_file, index=False)
    pitching_stats.to_csv(pitching_file, index=False)
    
    print(f"Saved batting stats to {batting_file}")
    print(f"Saved pitching stats to {pitching_file}")
    
    return batting_stats, pitching_stats


def process_season_data(season: int, data_dir: str = 'data/raw', output_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all data for a given MLB season.
    
    Args:
        season (int): The MLB season year to process
        data_dir (str): Directory containing raw Retrosheet data
        output_dir (str): Directory to save processed data
        
    Returns:
        tuple: (batting_stats_df, pitching_stats_df)
    """
    # Define file paths
    players_file = os.path.join(data_dir, 'allplayers.csv')
    batting_file = os.path.join(data_dir, 'batting.csv')
    pitching_file = os.path.join(data_dir, 'pitching.csv')
    fielding_file = os.path.join(data_dir, 'fielding.csv')
    plays_file = os.path.join(data_dir, 'plays.csv')
    
    # Load data files
    try:
        players_df = load_players_file(players_file)
        batting_df = load_batting_file(batting_file)
        pitching_df = load_pitching_file(pitching_file)
        
        # Optional files - we'll try to load them but continue if they don't exist
        try:
            fielding_df = load_fielding_file(fielding_file)
        except FileNotFoundError:
            fielding_df = None
            print(f"Warning: Fielding file not found: {fielding_file}")
        
        # Load plays file in chunks with date filtering to reduce memory usage
        try:
            print(f"Loading plays file in chunks with date filtering for {season}...")
            plays_df = load_plays_file_chunked(plays_file, season, season)
        except FileNotFoundError:
            plays_df = None
            print(f"Warning: Plays file not found: {plays_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Return empty DataFrames if files are not found
        return pd.DataFrame(), pd.DataFrame()
    
    # Use the new function that accepts DataFrames
    return process_season_data_with_dataframes(
        season, players_df, batting_df, pitching_df, fielding_df, plays_df, output_dir
    )


def process_multiple_seasons(start_year: int, end_year: int, data_dir: str = 'data/raw', output_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process data for multiple MLB seasons.
    
    Args:
        start_year (int): First season to process
        end_year (int): Last season to process
        data_dir (str): Directory containing raw Retrosheet data
        output_dir (str): Directory to save processed data
        
    Returns:
        tuple: (batting_stats_df, pitching_stats_df) with data for all seasons
    """
    all_batting = []
    all_pitching = []
    
    # Define file paths
    players_file = os.path.join(data_dir, 'allplayers.csv')
    batting_file = os.path.join(data_dir, 'batting.csv')
    pitching_file = os.path.join(data_dir, 'pitching.csv')
    fielding_file = os.path.join(data_dir, 'fielding.csv')
    plays_file = os.path.join(data_dir, 'plays.csv')
    
    # Load data files once
    try:
        print("Loading data files once for all seasons...")
        players_df = load_players_file(players_file)
        batting_df = load_batting_file(batting_file)
        pitching_df = load_pitching_file(pitching_file)
        
        # Optional files - we'll try to load them but continue if they don't exist
        try:
            fielding_df = load_fielding_file(fielding_file)
        except FileNotFoundError:
            fielding_df = None
            print(f"Warning: Fielding file not found: {fielding_file}")
        
        # Load plays file in chunks with date filtering to reduce memory usage
        try:
            print(f"Loading plays file in chunks with date filtering ({start_year}-{end_year})...")
            plays_df = load_plays_file_chunked(plays_file, start_year, end_year)
        except FileNotFoundError:
            plays_df = None
            print(f"Warning: Plays file not found: {plays_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # Return empty DataFrames if files are not found
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter data by year range immediately after loading to avoid processing unnecessary data
    print(f"Filtering data for years {start_year} to {end_year}...")
    
    if 'date' in batting_df.columns:
        batting_df = batting_df[
            (batting_df['date'].dt.year >= start_year) & 
            (batting_df['date'].dt.year <= end_year)
        ]
        print(f"Filtered batting data: {batting_df.shape[0]} rows")
    
    if 'date' in pitching_df.columns:
        pitching_df = pitching_df[
            (pitching_df['date'].dt.year >= start_year) & 
            (pitching_df['date'].dt.year <= end_year)
        ]
        print(f"Filtered pitching data: {pitching_df.shape[0]} rows")
    
    if fielding_df is not None and 'date' in fielding_df.columns:
        fielding_df = fielding_df[
            (fielding_df['date'].dt.year >= start_year) & 
            (fielding_df['date'].dt.year <= end_year)
        ]
        print(f"Filtered fielding data: {fielding_df.shape[0]} rows")
    
    # Note: plays_df is already filtered by date range in load_plays_file_chunked
    
    import gc  # Import garbage collector for memory management
    
    # Process each year with the filtered data
    for year in tqdm(range(start_year, end_year + 1), desc="Processing seasons"):
        print(f"\nProcessing year {year}...")
        
        # Filter the already-filtered data for just this year
        year_batting_df = batting_df.copy() if batting_df.empty else batting_df[batting_df['date'].dt.year == year].copy()
        year_pitching_df = pitching_df.copy() if pitching_df.empty else pitching_df[pitching_df['date'].dt.year == year].copy()
        
        year_fielding_df = None
        if fielding_df is not None and not fielding_df.empty:
            year_fielding_df = fielding_df[fielding_df['date'].dt.year == year].copy()
        
        # For plays data, we have two options:
        # 1. If we loaded all plays data at once, filter it for this year
        # 2. If the plays data is too large, load just this year's data in chunks
        year_plays_df = None
        if plays_df is not None and not plays_df.empty:
            # Option 1: Filter from already loaded data
            print(f"Filtering plays data for {year}...")
            year_plays_df = plays_df[plays_df['date'].dt.year == year].copy()
            print(f"Filtered plays data for {year}: {year_plays_df.shape[0]} rows")
        else:
            # Option 2: Load just this year's data in chunks
            # This is a fallback in case we couldn't load all plays data at once
            try:
                print(f"Loading plays data for {year} in chunks...")
                year_plays_df = load_plays_file_chunked(plays_file, year, year)
            except Exception as e:
                print(f"Error loading plays data for {year}: {e}")
                year_plays_df = None
        
        # Process this year's data directly with the dataframes
        batting_stats, pitching_stats = process_season_data_with_dataframes(
            year, players_df, year_batting_df, year_pitching_df, 
            year_fielding_df, year_plays_df, output_dir
        )
        
        # Clean up year-specific dataframes to free memory
        del year_batting_df, year_pitching_df, year_fielding_df, year_plays_df
        gc.collect()
        
        # Only append if data was successfully processed
        if not batting_stats.empty:
            all_batting.append(batting_stats)
        if not pitching_stats.empty:
            all_pitching.append(pitching_stats)
    
    # Combine all seasons if we have data
    if all_batting:
        combined_batting = pd.concat(all_batting, ignore_index=True)
        # Remove duplicate rows in the combined file
        combined_batting = combined_batting.drop_duplicates()
        combined_batting.to_csv(os.path.join(output_dir, 'batting_stats_all.csv'), index=False)
    else:
        combined_batting = pd.DataFrame()
        print("Warning: No batting data was processed")
    
    if all_pitching:
        combined_pitching = pd.concat(all_pitching, ignore_index=True)
        # Remove duplicate rows in the combined file
        combined_pitching = combined_pitching.drop_duplicates()
        combined_pitching.to_csv(os.path.join(output_dir, 'pitching_stats_all.csv'), index=False)
    else:
        combined_pitching = pd.DataFrame()
        print("Warning: No pitching data was processed")
    
    return combined_batting, combined_pitching


if __name__ == "__main__":
    # Example usage
    print("Retrosheet Data Processing")
    print("=========================")
    
    # Process the last 30 seasons
    current_year = 2025  # Update this to the current year
    start_year = current_year - 30
    
    print(f"Processing data from {start_year} to {current_year-1}")
    
    # Check if raw data directory exists and has files
    if not os.path.exists('data/raw'):
        print("Error: Raw data directory not found.")
        print("Please download Retrosheet data to the 'data/raw' directory.")
        exit(1)
    
    # Process the data
    batting_stats, pitching_stats = process_multiple_seasons(start_year, current_year-1)
    
    print("\nProcessing complete!")
    print(f"Batting stats shape: {batting_stats.shape}")
    print(f"Pitching stats shape: {pitching_stats.shape}")
