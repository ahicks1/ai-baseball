#!/usr/bin/env python3
"""
Backtesting Module for Player Performance Forecasting

This module provides functionality for backtesting forecasting models
against historical data to evaluate their performance.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from src.forecasting.data_preparation import (
    BATTING_CATEGORIES, 
    PITCHING_CATEGORIES,
    load_bio_data,
    load_data
)
from src.forecasting.model_trainer import (
    train_batting_models,
    train_pitching_models,
    train_category_model
)
from src.forecasting.forecast_generator import (
    generate_all_forecasts,
    generate_player_forecast
)


class BacktestEvaluator:
    """
    Class for backtesting forecasting models against historical data.
    """
    
    def __init__(self, start_season=2019, end_season=2024, output_dir='data/backtests',
                 category=None, model_type=None):
        """
        Initialize the BacktestEvaluator.
        
        Args:
            start_season (int): First season to backtest
            end_season (int): Last season to backtest
            output_dir (str): Directory to save backtest results
            category (str, optional): Specific category to backtest (e.g., 'HR', 'ERA')
            model_type (str, optional): Specific model type to backtest (e.g., 'ridge', 'random_forest', 'gradient_boosting', 'arima', 'exp_smooth')
        """
        self.start_season = start_season
        self.end_season = end_season
        self.output_dir = output_dir
        self.category = category
        self.model_type = model_type
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.batting_data = None
        self.pitching_data = None
        self.bio_data = None
        
        # Results storage
        self.batting_results = []
        self.pitching_results = []
        
        # League settings for H2H categories
        # If a specific category is provided, filter the categories
        if self.category:
            if self.category in BATTING_CATEGORIES:
                self.batting_categories = [self.category]
                self.pitching_categories = []
            elif self.category in PITCHING_CATEGORIES:
                self.pitching_categories = [self.category]
                self.batting_categories = []
            else:
                print(f"Warning: Category '{self.category}' not found in batting or pitching categories")
                self.batting_categories = BATTING_CATEGORIES
                self.pitching_categories = PITCHING_CATEGORIES
        else:
            self.batting_categories = BATTING_CATEGORIES
            self.pitching_categories = PITCHING_CATEGORIES
        
        # Print information about the backtest configuration
        if self.category:
            print(f"Backtesting for specific category: {self.category}")
        if self.model_type:
            print(f"Using specific model type: {self.model_type}")
    
    def load_data(self, batting_file='data/processed/batting_stats_all.csv', 
                 pitching_file='data/processed/pitching_stats_all.csv',
                 bio_file='data/raw/biofile0.csv'):
        """
        Load historical data for backtesting.
        
        Args:
            batting_file (str): Path to batting statistics CSV
            pitching_file (str): Path to pitching statistics CSV
            bio_file (str): Path to biographical data CSV
        """
        print("Loading data for backtesting...")
        
        # Load bio data
        if os.path.exists(bio_file):
            self.bio_data = load_bio_data(bio_file)
        else:
            print(f"Bio data file not found: {bio_file}")
            self.bio_data = None
        
        # Load batting and pitching data
        self.batting_data, self.pitching_data = load_data(batting_file, pitching_file)
        
        # Verify data contains the seasons we want to backtest
        batting_seasons = self.batting_data['SEASON'].unique()
        pitching_seasons = self.pitching_data['SEASON'].unique()
        
        print(f"Available batting seasons: {sorted(batting_seasons)}")
        print(f"Available pitching seasons: {sorted(pitching_seasons)}")
        
        # Check if we have the seasons we need
        missing_batting = [s for s in range(self.start_season, self.end_season + 1) 
                          if s not in batting_seasons]
        missing_pitching = [s for s in range(self.start_season, self.end_season + 1) 
                           if s not in pitching_seasons]
        
        if missing_batting:
            print(f"Warning: Missing batting data for seasons: {missing_batting}")
        
        if missing_pitching:
            print(f"Warning: Missing pitching data for seasons: {missing_pitching}")
    
    def run_backtest_for_season(self, season, n_jobs=8):
        """
        Run backtest for a specific season.
        
        Args:
            season (int): Season to backtest
            n_jobs (int): Number of parallel jobs to run
            
        Returns:
            tuple: (batting_forecasts, pitching_forecasts)
        """
        print(f"\n=== Backtesting for season {season} ===")
        
        # Filter data to include only seasons before the target season
        batting_train = self.batting_data[self.batting_data['SEASON'] < season]
        pitching_train = self.pitching_data[self.pitching_data['SEASON'] < season]
        
        # Get actual results for the target season
        batting_actual = self.batting_data[self.batting_data['SEASON'] == season]
        pitching_actual = self.pitching_data[self.pitching_data['SEASON'] == season]

        # Get test dataset to train against for the target season
        batting_test = self.batting_data[self.batting_data['SEASON'] == season-1]
        pitching_test = self.pitching_data[self.pitching_data['SEASON'] == season-1]
        
        print(f"Training on {len(batting_train)} batting records and {len(pitching_train)} pitching records")
        print(f"Testing against {len(batting_actual)} batting records and {len(pitching_actual)} pitching records")
        
        # Initialize forecaster
        batting_models, pitching_models = {}, {}
        
        # If a specific category is provided, train only that category
        if self.category:
            from src.forecasting.data_preparation import prepare_regression_features
            
            if self.category in self.batting_categories:
                print(f"Training batting model for category: {self.category}...")
                # Prepare regression data
                regression_data = prepare_regression_features(batting_train, self.bio_data)
                
                # Train the specific category model
                category, model_dict = train_category_model(
                    self.category, 
                    regression_data, 
                    'batting', 
                    self.model_type or 'ensemble'
                )
                
                # Add the model to the forecaster
                batting_models = {category: model_dict}
                
            elif self.category in self.pitching_categories:
                print(f"Training pitching model for category: {self.category}...")
                # Prepare regression data
                regression_data = prepare_regression_features(pitching_train, self.bio_data)
                
                # Train the specific category model
                category, model_dict = train_category_model(
                    self.category, 
                    regression_data, 
                    'pitching', 
                    self.model_type or 'ensemble'
                )
                
                # Add the model to the forecaster
                pitching_models = {category: model_dict}
        
        # Generate forecasts for the target season
        print(f"Generating forecasts for season {season}...")
        
        # Create a directory name that includes category and model type if specified
        output_subdir = f"season_{season}"
        if self.category:
            output_subdir += f"_{self.category}"
        if self.model_type:
            output_subdir += f"_{self.model_type}"
        
        batting_forecasts, pitching_forecasts = generate_all_forecasts(
            batting_data=batting_test,
            pitching_data=pitching_test,
            batting_models=batting_models,
            pitching_models=pitching_models,
            bio_data=self.bio_data,
            output_dir=os.path.join(self.output_dir, output_subdir),
            n_jobs=n_jobs,
            forecast_season=season
        )
        
        # Add actual values to the forecasts
        self._add_actual_values(batting_forecasts, batting_actual, is_pitcher=False)
        self._add_actual_values(pitching_forecasts, pitching_actual, is_pitcher=True)
        
        # Calculate performance metrics
        batting_metrics = self._calculate_metrics(batting_forecasts, is_pitcher=False)
        pitching_metrics = self._calculate_metrics(pitching_forecasts, is_pitcher=True)
        
        # Save metrics
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        batting_metrics.to_csv(os.path.join(metrics_dir, f"batting_metrics_{season}.csv"), index=False)
        pitching_metrics.to_csv(os.path.join(metrics_dir, f"pitching_metrics_{season}.csv"), index=False)
        
        return batting_forecasts, pitching_forecasts
    
    def _add_actual_values(self, forecasts, actual_data, is_pitcher=False):
        """
        Add actual values to the forecast DataFrame.
        
        Args:
            forecasts (pd.DataFrame): DataFrame with forecasts
            actual_data (pd.DataFrame): DataFrame with actual values
            is_pitcher (bool): Whether the data is for pitchers
        """
        if forecasts is None or len(forecasts) == 0:
            return
        
        # Create a mapping of player_id to actual values
        categories = self.pitching_categories if is_pitcher else self.batting_categories
        
        for player_id in forecasts['PLAYER_ID'].unique():
            player_actual = actual_data[actual_data['PLAYER_ID'] == player_id]
            
            if len(player_actual) > 0:
                # Get the index of this player in the forecasts DataFrame
                idx = forecasts.index[forecasts['PLAYER_ID'] == player_id].tolist()
                
                if idx:
                    # Add actual values for each category
                    for category in categories:
                        if category in player_actual.columns:
                            forecasts.loc[idx, f"{category}_actual"] = player_actual[category].values[0]
    
    def _calculate_metrics(self, forecasts, is_pitcher=False):
        """
        Calculate performance metrics for forecasts.
        
        Args:
            forecasts (pd.DataFrame): DataFrame with forecasts and actual values
            is_pitcher (bool): Whether the data is for pitchers
            
        Returns:
            pd.DataFrame: DataFrame with performance metrics
        """
        if forecasts is None or len(forecasts) == 0:
            return pd.DataFrame()
        
        categories = self.pitching_categories if is_pitcher else self.batting_categories
        
        # If a specific model type is provided, only calculate metrics for that model
        if self.model_type and self.model_type in ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting']:
            models = [self.model_type, 'combined']
        else:
            models = ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting', 'combined']
        
        metrics = []
        
        for category in categories:
            actual_col = f"{category}_actual"
            
            # Skip if we don't have actual values
            if actual_col not in forecasts.columns:
                continue
            
            for model in models:
                model_col = category if model == 'combined' else f"{category}_{model}"
                
                # Skip if we don't have predictions for this model
                if model_col not in forecasts.columns:
                    continue
                
                # Get actual and predicted values (dropping NaN)
                data = forecasts[[actual_col, model_col]].dropna()
                
                if len(data) == 0:
                    continue
                
                actual = data[actual_col]
                predicted = data[model_col]
                
                # Calculate metrics
                mse = np.mean((predicted - actual) ** 2)
                mae = np.mean(np.abs(predicted - actual))
                
                # Calculate R² if possible
                if len(actual) > 1 and np.var(actual) > 0:
                    r2 = 1 - (np.sum((predicted - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
                else:
                    r2 = np.nan
                
                metrics.append({
                    'category': category,
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'n_players': len(data)
                })
        
        return pd.DataFrame(metrics)
    
    def run_all_backtests(self, n_jobs=8):
        """
        Run backtests for all seasons in the range.
        
        Args:
            n_jobs (int): Number of parallel jobs to run
            
        Returns:
            tuple: (batting_results, pitching_results)
        """
        # Clear previous results
        self.batting_results = []
        self.pitching_results = []
        
        # Run backtests for each season
        for season in range(self.start_season, self.end_season + 1):
            batting_forecasts, pitching_forecasts = self.run_backtest_for_season(season, n_jobs)
            
            if batting_forecasts is not None and len(batting_forecasts) > 0:
                # Add season column if not already present
                if 'SEASON' not in batting_forecasts.columns:
                    batting_forecasts['SEASON'] = season
                
                self.batting_results.append(batting_forecasts)
            
            if pitching_forecasts is not None and len(pitching_forecasts) > 0:
                # Add season column if not already present
                if 'SEASON' not in pitching_forecasts.columns:
                    pitching_forecasts['SEASON'] = season
                
                self.pitching_results.append(pitching_forecasts)
        
        # Combine results
        combined_batting = pd.concat(self.batting_results, ignore_index=True) if self.batting_results else None
        combined_pitching = pd.concat(self.pitching_results, ignore_index=True) if self.pitching_results else None
        
        # Save combined results
        if combined_batting is not None:
            combined_batting.to_csv(os.path.join(self.output_dir, "batting_backtests.csv"), index=False)
        
        if combined_pitching is not None:
            combined_pitching.to_csv(os.path.join(self.output_dir, "pitching_backtests.csv"), index=False)
        
        # Combine metrics
        self._combine_metrics()
        
        return combined_batting, combined_pitching
    
    def _combine_metrics(self):
        """
        Combine metrics from all seasons into a single file.
        """
        metrics_dir = os.path.join(self.output_dir, "metrics")
        
        # Combine batting metrics
        batting_metrics = []
        if self.batting_results:
            for season in range(self.start_season, self.end_season + 1):
                file_path = os.path.join(metrics_dir, f"batting_metrics_{season}.csv")
                if os.path.exists(file_path):
                    metrics = pd.read_csv(file_path)
                    metrics['season'] = season
                    batting_metrics.append(metrics)
        
        if batting_metrics:
            combined_batting_metrics = pd.concat(batting_metrics, ignore_index=True)
            combined_batting_metrics.to_csv(os.path.join(metrics_dir, "batting_metrics_all.csv"), index=False)
        
        # Combine pitching metrics
        pitching_metrics = []
        if self.pitching_results:
            for season in range(self.start_season, self.end_season + 1):
                file_path = os.path.join(metrics_dir, f"pitching_metrics_{season}.csv")
                if os.path.exists(file_path):
                    metrics = pd.read_csv(file_path)
                    metrics['season'] = season
                    pitching_metrics.append(metrics)
        
        if pitching_metrics:
            combined_pitching_metrics = pd.concat(pitching_metrics, ignore_index=True)
            combined_pitching_metrics.to_csv(os.path.join(metrics_dir, "pitching_metrics_all.csv"), index=False)
    
    def prepare_for_visualization(self, output_dir='data/projections'):
        """
        Prepare backtest results for visualization with ModelVisualizer.
        
        Args:
            output_dir (str): Directory to save visualization-ready data
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load combined results
        batting_file = os.path.join(self.output_dir, "batting_backtests.csv")
        pitching_file = os.path.join(self.output_dir, "pitching_backtests.csv")
        
        if os.path.exists(batting_file):
            batting_backtests = pd.read_csv(batting_file)
            
            # Rename actual columns to match the format expected by ModelVisualizer
            for category in self.batting_categories:
                if f"{category}_actual" in batting_backtests.columns:
                    # Add the actual value to the historical data
                    self.batting_data = self.batting_data.copy()
                    
                    # For each player and season in the backtests
                    for _, row in batting_backtests.iterrows():
                        player_id = row['PLAYER_ID']
                        season = row['SEASON']
                        actual_value = row.get(f"{category}_actual")
                        
                        if not pd.isna(actual_value):
                            # Update or add the actual value in the historical data
                            mask = ((self.batting_data['PLAYER_ID'] == player_id) & 
                                   (self.batting_data['SEASON'] == season))
                            
                            if mask.any():
                                self.batting_data.loc[mask, category] = actual_value
            
            # Save the updated historical data
            self.batting_data.to_csv(os.path.join(output_dir, "batting_stats_with_backtests.csv"), index=False)
            
            # Save the forecasts in the format expected by ModelVisualizer
            batting_backtests.to_csv(os.path.join(output_dir, "batting_forecasts.csv"), index=False)
        
        if os.path.exists(pitching_file):
            pitching_backtests = pd.read_csv(pitching_file)
            
            # Rename actual columns to match the format expected by ModelVisualizer
            for category in self.pitching_categories:
                if f"{category}_actual" in pitching_backtests.columns:
                    # Add the actual value to the historical data
                    self.pitching_data = self.pitching_data.copy()
                    
                    # For each player and season in the backtests
                    for _, row in pitching_backtests.iterrows():
                        player_id = row['PLAYER_ID']
                        season = row['SEASON']
                        actual_value = row.get(f"{category}_actual")
                        
                        if not pd.isna(actual_value):
                            # Update or add the actual value in the historical data
                            mask = ((self.pitching_data['PLAYER_ID'] == player_id) & 
                                   (self.pitching_data['SEASON'] == season))
                            
                            if mask.any():
                                self.pitching_data.loc[mask, category] = actual_value
            
            # Save the updated historical data
            self.pitching_data.to_csv(os.path.join(output_dir, "pitching_stats_with_backtests.csv"), index=False)
            
            # Save the forecasts in the format expected by ModelVisualizer
            pitching_backtests.to_csv(os.path.join(output_dir, "pitching_forecasts.csv"), index=False)
        
        print(f"Backtest results prepared for visualization in {output_dir}")
        print("You can now use ModelVisualizer to visualize the results.")


def main():
    """
    Main entry point for the backtesting module.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest forecasting models')
    parser.add_argument('--start-season', type=int, default=2018, help='First season to backtest')
    parser.add_argument('--end-season', type=int, default=2023, help='Last season to backtest')
    parser.add_argument('--output-dir', default='data/backtests', help='Directory to save backtest results')
    parser.add_argument('--n-jobs', type=int, default=8, help='Number of parallel jobs to run')
    parser.add_argument('--batting-file', default='data/processed/batting_stats_all.csv', 
                        help='Path to historical batting statistics CSV')
    parser.add_argument('--pitching-file', default='data/processed/pitching_stats_all.csv', 
                        help='Path to historical pitching statistics CSV')
    parser.add_argument('--bio-file', default='data/raw/biofile0.csv', 
                        help='Path to biographical data CSV')
    parser.add_argument('--visualization-dir', default='data/projections', 
                        help='Directory to save visualization-ready data')
    parser.add_argument('--category', 
                        help='Specific category to backtest (e.g., HR, ERA)')
    parser.add_argument('--model-type', 
                        help='Specific model type to backtest (ridge, random_forest, gradient_boosting, arima, exp_smooth)')
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = BacktestEvaluator(
        start_season=args.start_season,
        end_season=args.end_season,
        output_dir=args.output_dir,
        category=args.category,
        model_type=args.model_type
    )
    
    # Load data
    backtester.load_data(
        batting_file=args.batting_file,
        pitching_file=args.pitching_file,
        bio_file=args.bio_file
    )
    
    # Run backtests
    backtester.run_all_backtests(n_jobs=args.n_jobs)
    
    # Prepare for visualization
    backtester.prepare_for_visualization(output_dir=args.visualization_dir)


if __name__ == "__main__":
    print("Forecasting Model Backtester")
    print("===========================")
    main()
