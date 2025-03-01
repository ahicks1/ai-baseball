#!/usr/bin/env python3
"""
Statistical Analysis Module

This module provides functions for analyzing baseball statistics
and identifying trends relevant to fantasy baseball.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class StatsAnalyzer:
    """
    Class for analyzing baseball statistics for fantasy baseball purposes.
    """
    
    def __init__(self, batting_data=None, pitching_data=None):
        """
        Initialize the StatsAnalyzer with batting and pitching data.
        
        Args:
            batting_data (pd.DataFrame, optional): DataFrame with batting statistics
            pitching_data (pd.DataFrame, optional): DataFrame with pitching statistics
        """
        self.batting_data = batting_data
        self.pitching_data = pitching_data
        
        # League settings for H2H categories
        self.batting_categories = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
        self.pitching_categories = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']
    
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
    
    def calculate_category_correlations(self, data, categories):
        """
        Calculate correlations between statistical categories.
        
        Args:
            data (pd.DataFrame): DataFrame with player statistics
            categories (list): List of categories to analyze
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Filter to only include the categories of interest
        category_data = data[categories].copy()
        
        # Calculate correlation matrix
        corr_matrix = category_data.corr()
        
        return corr_matrix
    
    def analyze_batting_correlations(self):
        """
        Analyze correlations between batting categories.
        
        Returns:
            pd.DataFrame: Correlation matrix for batting categories
        """
        if self.batting_data is None:
            raise ValueError("Batting data not loaded. Call load_data() first.")
        
        return self.calculate_category_correlations(self.batting_data, self.batting_categories)
    
    def analyze_pitching_correlations(self):
        """
        Analyze correlations between pitching categories.
        
        Returns:
            pd.DataFrame: Correlation matrix for pitching categories
        """
        if self.pitching_data is None:
            raise ValueError("Pitching data not loaded. Call load_data() first.")
        
        # Create a copy of the data
        pitching_data = self.pitching_data.copy()
        
        # For ERA and WHIP, lower is better, so we'll negate them for correlation analysis
        pitching_data['ERA'] = -pitching_data['ERA']
        pitching_data['WHIP'] = -pitching_data['WHIP']
        
        # Calculate SV+HLD and W+QS if they don't exist
        if 'SV+HLD' not in pitching_data.columns and 'SV' in pitching_data.columns and 'HLD' in pitching_data.columns:
            pitching_data['SV+HLD'] = pitching_data['SV'] + pitching_data['HLD']
            print(pitching_data.columns)
        
        if 'W+QS' not in pitching_data.columns and 'W' in pitching_data.columns and 'QS' in pitching_data.columns:
            pitching_data['W+QS'] = pitching_data['W'] + pitching_data['QS']
        
        return self.calculate_category_correlations(pitching_data, self.pitching_categories)
    
    def plot_category_correlations(self, corr_matrix, title):
        """
        Plot a heatmap of category correlations.
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def analyze_player_consistency(self, data, player_id_col='PLAYER_ID', season_col='SEASON', 
                                  categories=None, min_seasons=3):
        """
        Analyze player consistency across seasons.
        
        Args:
            data (pd.DataFrame): DataFrame with player statistics
            player_id_col (str): Column name for player ID
            season_col (str): Column name for season
            categories (list, optional): List of categories to analyze
            min_seasons (int): Minimum number of seasons required
            
        Returns:
            pd.DataFrame: DataFrame with consistency metrics
        """
        if categories is None:
            if data is self.batting_data:
                categories = self.batting_categories
            else:
                categories = self.pitching_categories
        
        # Group by player and calculate stats
        results = []
        
        # Get unique players
        players = data[player_id_col].unique()
        
        for player in players:
            player_data = data[data[player_id_col] == player]
            
            # Skip players with too few seasons
            if len(player_data) < min_seasons:
                continue
            
            player_result = {'PLAYER_ID': player, 'SEASONS': len(player_data)}
            
            # Calculate mean, std, and coefficient of variation for each category
            for category in categories:
                # Handle blank/NaN values by filling them with 0.0
                values = player_data[category].fillna(0.0).values
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean if mean != 0 else np.nan
                
                player_result[f'{category}_MEAN'] = mean
                player_result[f'{category}_STD'] = std
                player_result[f'{category}_CV'] = cv
            
            results.append(player_result)
        
        return pd.DataFrame(results)
    
    def analyze_batting_consistency(self, min_seasons=3):
        """
        Analyze batter consistency across seasons.
        
        Args:
            min_seasons (int): Minimum number of seasons required
            
        Returns:
            pd.DataFrame: DataFrame with consistency metrics
        """
        if self.batting_data is None:
            raise ValueError("Batting data not loaded. Call load_data() first.")
        
        return self.analyze_player_consistency(self.batting_data, categories=self.batting_categories, 
                                              min_seasons=min_seasons)
    
    def analyze_pitching_consistency(self, min_seasons=3):
        """
        Analyze pitcher consistency across seasons.
        
        Args:
            min_seasons (int): Minimum number of seasons required
            
        Returns:
            pd.DataFrame: DataFrame with consistency metrics
        """
        if self.pitching_data is None:
            raise ValueError("Pitching data not loaded. Call load_data() first.")
        
        return self.analyze_player_consistency(self.pitching_data, categories=self.pitching_categories, 
                                              min_seasons=min_seasons)
    
    def identify_category_scarcity(self, data, categories=None, percentile_cutoffs=[90, 75, 50, 25]):
        """
        Identify scarcity in statistical categories.
        
        Args:
            data (pd.DataFrame): DataFrame with player statistics
            categories (list, optional): List of categories to analyze
            percentile_cutoffs (list): Percentiles to calculate
            
        Returns:
            pd.DataFrame: DataFrame with scarcity metrics
        """
        if categories is None:
            if data is self.batting_data:
                categories = self.batting_categories
            else:
                categories = self.pitching_categories
        
        results = {}
        
        for category in categories:
            # Handle blank/NaN values by filling them with 0.0
            category_data = data[category].fillna(0.0)
            
            # For ERA and WHIP, lower is better
            if category in ['ERA', 'WHIP']:
                values = -category_data.values
            else:
                values = category_data.values
            
            percentiles = np.percentile(values, percentile_cutoffs)
            
            results[category] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
            for cutoff, value in zip(percentile_cutoffs, percentiles):
                results[category][f'p{cutoff}'] = value
        
        return pd.DataFrame(results).T
    
    def analyze_batting_scarcity(self):
        """
        Analyze scarcity in batting categories.
        
        Returns:
            pd.DataFrame: DataFrame with scarcity metrics
        """
        if self.batting_data is None:
            raise ValueError("Batting data not loaded. Call load_data() first.")
        
        return self.identify_category_scarcity(self.batting_data, self.batting_categories)
    
    def analyze_pitching_scarcity(self):
        """
        Analyze scarcity in pitching categories.
        
        Returns:
            pd.DataFrame: DataFrame with scarcity metrics
        """
        if self.pitching_data is None:
            raise ValueError("Pitching data not loaded. Call load_data() first.")
        
        return self.identify_category_scarcity(self.pitching_data, self.pitching_categories)
    
    def run_full_analysis(self, output_dir='data/analysis'):
        """
        Run a full analysis and save results to files.
        
        Args:
            output_dir (str): Directory to save analysis results
            
        Returns:
            dict: Dictionary with analysis results
        """
        if self.batting_data is None or self.pitching_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Analyze correlations
        print("Analyzing category correlations...")
        batting_corr = self.analyze_batting_correlations()
        pitching_corr = self.analyze_pitching_correlations()
        
        # Save correlation matrices
        batting_corr.to_csv(os.path.join(output_dir, 'batting_correlations.csv'))
        pitching_corr.to_csv(os.path.join(output_dir, 'pitching_correlations.csv'))
        
        # Plot and save correlation heatmaps
        batting_corr_fig = self.plot_category_correlations(batting_corr, "Batting Category Correlations")
        pitching_corr_fig = self.plot_category_correlations(pitching_corr, "Pitching Category Correlations")
        
        batting_corr_fig.savefig(os.path.join(output_dir, 'batting_correlations.png'))
        pitching_corr_fig.savefig(os.path.join(output_dir, 'pitching_correlations.png'))
        
        results['correlations'] = {
            'batting': batting_corr,
            'pitching': pitching_corr
        }
        
        # Analyze player consistency
        print("Analyzing player consistency...")
        batting_consistency = self.analyze_batting_consistency()
        pitching_consistency = self.analyze_pitching_consistency()
        
        # Save consistency analysis
        batting_consistency.to_csv(os.path.join(output_dir, 'batting_consistency.csv'), index=False)
        pitching_consistency.to_csv(os.path.join(output_dir, 'pitching_consistency.csv'), index=False)
        
        results['consistency'] = {
            'batting': batting_consistency,
            'pitching': pitching_consistency
        }
        
        # Analyze category scarcity
        print("Analyzing category scarcity...")
        batting_scarcity = self.analyze_batting_scarcity()
        pitching_scarcity = self.analyze_pitching_scarcity()
        
        # Save scarcity analysis
        batting_scarcity.to_csv(os.path.join(output_dir, 'batting_scarcity.csv'))
        pitching_scarcity.to_csv(os.path.join(output_dir, 'pitching_scarcity.csv'))
        
        results['scarcity'] = {
            'batting': batting_scarcity,
            'pitching': pitching_scarcity
        }
        
        print(f"Analysis complete. Results saved to {output_dir}")
        
        return results


if __name__ == "__main__":
    # Example usage
    print("Baseball Statistics Analyzer")
    print("===========================")
    
    # Initialize analyzer
    analyzer = StatsAnalyzer()
    
    # Load data
    analyzer.load_data(
        batting_file='data/processed/batting_stats_all.csv',
        pitching_file='data/processed/pitching_stats_all.csv'
    )
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    print("\nAnalysis complete!")
