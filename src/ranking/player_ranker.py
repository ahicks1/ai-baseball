#!/usr/bin/env python3
"""
Player Ranking Module

This module provides functions for ranking players based on
projected fantasy value for the upcoming season.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class PlayerRanker:
    """
    Class for ranking players based on projected fantasy value.
    """
    
    def __init__(self, batting_projections=None, pitching_projections=None):
        """
        Initialize the PlayerRanker with player projections.
        
        Args:
            batting_projections (pd.DataFrame, optional): DataFrame with batting projections
            pitching_projections (pd.DataFrame, optional): DataFrame with pitching projections
        """
        self.batting_projections = batting_projections
        self.pitching_projections = pitching_projections
        
        # League settings for H2H categories
        self.batting_categories = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
        self.pitching_categories = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']
        
        # Category weights (default to equal weighting)
        self.batting_weights = {cat: 1.0 for cat in self.batting_categories}
        self.pitching_weights = {cat: 1.0 for cat in self.pitching_categories}
        
        # Player position data
        self.player_positions = {}
        
        # Player rankings
        self.batting_rankings = None
        self.pitching_rankings = None
        self.overall_rankings = None
    
    def load_projections(self, batting_file, pitching_file):
        """
        Load player projections from CSV files.
        
        Args:
            batting_file (str): Path to batting projections CSV
            pitching_file (str): Path to pitching projections CSV
        """
        print(f"Loading batting projections from {batting_file}")
        self.batting_projections = pd.read_csv(batting_file)
        
        print(f"Loading pitching projections from {pitching_file}")
        self.pitching_projections = pd.read_csv(pitching_file)
        
        print(f"Loaded projections for {len(self.batting_projections)} batters and {len(self.pitching_projections)} pitchers")
    
    def load_player_positions(self, positions_file):
        """
        Load player position data from a CSV file.
        
        Args:
            positions_file (str): Path to player positions CSV
        """
        print(f"Loading player positions from {positions_file}")
        positions_df = pd.read_csv(positions_file)
        
        # Create a dictionary mapping player IDs to positions
        self.player_positions = positions_df.set_index('PLAYER_ID')['POSITIONS'].to_dict()
        
        print(f"Loaded positions for {len(self.player_positions)} players")
    
    def set_category_weights(self, batting_weights=None, pitching_weights=None):
        """
        Set custom weights for fantasy categories.
        
        Args:
            batting_weights (dict, optional): Dictionary mapping batting categories to weights
            pitching_weights (dict, optional): Dictionary mapping pitching categories to weights
        """
        if batting_weights:
            for cat, weight in batting_weights.items():
                if cat in self.batting_categories:
                    self.batting_weights[cat] = weight
        
        if pitching_weights:
            for cat, weight in pitching_weights.items():
                if cat in self.pitching_categories:
                    self.pitching_weights[cat] = weight
        
        print("Category weights updated")
    
    def calculate_z_scores(self, data, categories, reverse_categories=None):
        """
        Calculate z-scores for each category.
        
        Args:
            data (pd.DataFrame): DataFrame with player projections
            categories (list): List of categories to calculate z-scores for
            reverse_categories (list, optional): List of categories where lower is better
            
        Returns:
            pd.DataFrame: DataFrame with z-scores for each category
        """
        if reverse_categories is None:
            reverse_categories = []
        
        # Create a copy of the data
        df = data.copy()
        
        # Calculate z-scores for each category
        for category in categories:
            # Get the mean and standard deviation
            mean = df[category].mean()
            std = df[category].std()
            
            # Calculate z-score
            if std > 0:
                if category in reverse_categories:
                    # For categories where lower is better (e.g., ERA, WHIP)
                    df[f'{category}_Z'] = -1 * (df[category] - mean) / std
                else:
                    # For categories where higher is better
                    df[f'{category}_Z'] = (df[category] - mean) / std
            else:
                df[f'{category}_Z'] = 0
        
        return df
    
    def calculate_weighted_z_scores(self, data, categories, weights, z_score_suffix='_Z'):
        """
        Calculate weighted z-scores for each category.
        
        Args:
            data (pd.DataFrame): DataFrame with z-scores
            categories (list): List of categories
            weights (dict): Dictionary mapping categories to weights
            z_score_suffix (str): Suffix for z-score columns
            
        Returns:
            pd.DataFrame: DataFrame with weighted z-scores
        """
        # Create a copy of the data
        df = data.copy()
        
        # Calculate weighted z-scores
        for category in categories:
            z_score_col = f'{category}{z_score_suffix}'
            if z_score_col in df.columns:
                weight = weights.get(category, 1.0)
                df[f'{category}_WZ'] = df[z_score_col] * weight
        
        return df
    
    def calculate_total_value(self, data, categories, value_suffix='_WZ'):
        """
        Calculate total fantasy value based on weighted z-scores.
        
        Args:
            data (pd.DataFrame): DataFrame with weighted z-scores
            categories (list): List of categories
            value_suffix (str): Suffix for value columns
            
        Returns:
            pd.DataFrame: DataFrame with total fantasy value
        """
        # Create a copy of the data
        df = data.copy()
        
        # Calculate total value
        value_columns = [f'{category}{value_suffix}' for category in categories]
        df['TOTAL_VALUE'] = df[value_columns].sum(axis=1)
        
        return df
    
    def calculate_positional_adjustments(self, data, position_col='POSITIONS'):
        """
        Calculate positional adjustments based on scarcity.
        
        Args:
            data (pd.DataFrame): DataFrame with player rankings
            position_col (str): Column name for player positions
            
        Returns:
            pd.DataFrame: DataFrame with positional adjustments
        """
        # This is a placeholder implementation
        # In a real implementation, we would calculate positional adjustments
        # based on the scarcity of each position
        
        # Create a copy of the data
        df = data.copy()
        
        # Define position scarcity adjustments
        # These are arbitrary values for demonstration
        position_adjustments = {
            'C': 1.5,    # Catchers are scarce
            '1B': 0.8,   # First basemen are abundant
            '2B': 1.2,   # Second basemen are somewhat scarce
            '3B': 1.0,   # Third basemen are average
            'SS': 1.3,   # Shortstops are scarce
            'OF': 0.9,   # Outfielders are abundant
            'DH': 0.7,   # Designated hitters are very abundant
            'SP': 1.0,   # Starting pitchers are average
            'RP': 1.2    # Relief pitchers are somewhat scarce
        }
        
        # Apply positional adjustments
        if position_col in df.columns:
            # Initialize positional adjustment column
            df['POS_ADJ'] = 1.0
            
            # Apply adjustments based on positions
            for idx, row in df.iterrows():
                positions = str(row[position_col]).split(',')
                
                # Get the maximum adjustment for the player's positions
                max_adj = max([position_adjustments.get(pos.strip(), 1.0) for pos in positions])
                df.at[idx, 'POS_ADJ'] = max_adj
            
            # Apply the adjustment to the total value
            df['ADJ_VALUE'] = df['TOTAL_VALUE'] * df['POS_ADJ']
        else:
            # If no position data, just copy the total value
            df['ADJ_VALUE'] = df['TOTAL_VALUE']
        
        return df
    
    def rank_players(self, data, value_col='ADJ_VALUE', rank_col='RANK'):
        """
        Rank players based on adjusted fantasy value.
        
        Args:
            data (pd.DataFrame): DataFrame with player values
            value_col (str): Column name for value to rank by
            rank_col (str): Column name for rank
            
        Returns:
            pd.DataFrame: DataFrame with player rankings
        """
        # Create a copy of the data
        df = data.copy()
        
        # Sort by value in descending order
        df = df.sort_values(value_col, ascending=False)
        
        # Add rank column
        df[rank_col] = range(1, len(df) + 1)
        
        return df
    
    def rank_batters(self):
        """
        Rank batters based on projected fantasy value.
        
        Returns:
            pd.DataFrame: DataFrame with batter rankings
        """
        if self.batting_projections is None:
            raise ValueError("Batting projections not loaded. Call load_projections() first.")
        
        print("Ranking batters...")
        
        # Calculate z-scores
        df = self.calculate_z_scores(self.batting_projections, self.batting_categories)
        
        # Calculate weighted z-scores
        df = self.calculate_weighted_z_scores(df, self.batting_categories, self.batting_weights)
        
        # Calculate total value
        df = self.calculate_total_value(df, self.batting_categories)
        
        # Add position data if available
        if self.player_positions:
            df['POSITIONS'] = df['PLAYER_ID'].map(self.player_positions)
        
        # Calculate positional adjustments
        df = self.calculate_positional_adjustments(df)
        
        # Rank players
        df = self.rank_players(df)
        
        # Add player type
        df['PLAYER_TYPE'] = 'BATTER'
        
        self.batting_rankings = df
        return df
    
    def rank_pitchers(self):
        """
        Rank pitchers based on projected fantasy value.
        
        Returns:
            pd.DataFrame: DataFrame with pitcher rankings
        """
        if self.pitching_projections is None:
            raise ValueError("Pitching projections not loaded. Call load_projections() first.")
        
        print("Ranking pitchers...")
        
        # Calculate z-scores (ERA and WHIP are reverse categories)
        df = self.calculate_z_scores(self.pitching_projections, self.pitching_categories, 
                                    reverse_categories=['ERA', 'WHIP'])
        
        # Calculate weighted z-scores
        df = self.calculate_weighted_z_scores(df, self.pitching_categories, self.pitching_weights)
        
        # Calculate total value
        df = self.calculate_total_value(df, self.pitching_categories)
        
        # Add position data if available
        if self.player_positions:
            df['POSITIONS'] = df['PLAYER_ID'].map(self.player_positions)
        
        # Calculate positional adjustments
        df = self.calculate_positional_adjustments(df)
        
        # Rank players
        df = self.rank_players(df)
        
        # Add player type
        df['PLAYER_TYPE'] = 'PITCHER'
        
        self.pitching_rankings = df
        return df
    
    def create_overall_rankings(self):
        """
        Create overall rankings combining batters and pitchers.
        
        Returns:
            pd.DataFrame: DataFrame with overall player rankings
        """
        if self.batting_rankings is None:
            self.rank_batters()
        
        if self.pitching_rankings is None:
            self.rank_pitchers()
        
        print("Creating overall rankings...")
        
        # Combine batting and pitching rankings
        combined = pd.concat([self.batting_rankings, self.pitching_rankings])
        
        # Rank players
        overall = self.rank_players(combined)
        
        self.overall_rankings = overall
        return overall
    
    def get_top_players(self, n=100, player_type=None):
        """
        Get the top N players from the rankings.
        
        Args:
            n (int): Number of players to return
            player_type (str, optional): Filter by player type ('BATTER' or 'PITCHER')
            
        Returns:
            pd.DataFrame: DataFrame with top N players
        """
        if self.overall_rankings is None:
            self.create_overall_rankings()
        
        # Filter by player type if specified
        if player_type:
            filtered = self.overall_rankings[self.overall_rankings['PLAYER_TYPE'] == player_type]
        else:
            filtered = self.overall_rankings
        
        # Get top N players
        top_players = filtered.head(n)
        
        return top_players
    
    def get_players_by_position(self, position, n=None):
        """
        Get players for a specific position from the rankings.
        
        Args:
            position (str): Position to filter by
            n (int, optional): Number of players to return
            
        Returns:
            pd.DataFrame: DataFrame with players for the position
        """
        if self.overall_rankings is None:
            self.create_overall_rankings()
        
        # Filter by position
        filtered = self.overall_rankings[self.overall_rankings['POSITIONS'].str.contains(position, na=False)]
        
        # Get top N players if specified
        if n:
            filtered = filtered.head(n)
        
        return filtered
    
    def plot_value_distribution(self, player_type=None, n=100):
        """
        Plot the distribution of player values.
        
        Args:
            player_type (str, optional): Filter by player type ('BATTER' or 'PITCHER')
            n (int): Number of top players to include
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.overall_rankings is None:
            self.create_overall_rankings()
        
        # Filter by player type if specified
        if player_type:
            filtered = self.overall_rankings[self.overall_rankings['PLAYER_TYPE'] == player_type]
        else:
            filtered = self.overall_rankings
        
        # Get top N players
        top_players = filtered.head(n)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot value distribution
        sns.barplot(x='RANK', y='ADJ_VALUE', hue='PLAYER_TYPE', data=top_players)
        
        plt.title(f'Value Distribution of Top {n} Players')
        plt.xlabel('Rank')
        plt.ylabel('Adjusted Value')
        plt.xticks(range(0, n, 10))
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_position_scarcity(self):
        """
        Plot position scarcity based on player values.
        
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if self.overall_rankings is None:
            self.create_overall_rankings()
        
        # Check if position data is available
        if 'POSITIONS' not in self.overall_rankings.columns:
            raise ValueError("Position data not available. Call load_player_positions() first.")
        
        # Define common positions
        positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
        
        # Calculate average value for top 10 players at each position
        position_values = []
        
        for position in positions:
            pos_players = self.get_players_by_position(position, n=10)
            if len(pos_players) > 0:
                avg_value = pos_players['ADJ_VALUE'].mean()
                position_values.append({'Position': position, 'Average Value': avg_value})
        
        # Create DataFrame
        pos_df = pd.DataFrame(position_values)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot position scarcity
        sns.barplot(x='Position', y='Average Value', data=pos_df)
        
        plt.title('Position Scarcity (Average Value of Top 10 Players)')
        plt.xlabel('Position')
        plt.ylabel('Average Adjusted Value')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_rankings(self, output_dir='data/rankings'):
        """
        Save player rankings to CSV files.
        
        Args:
            output_dir (str): Directory to save rankings
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save batting rankings
        if self.batting_rankings is not None:
            batting_file = os.path.join(output_dir, 'batting_rankings.csv')
            self.batting_rankings.to_csv(batting_file, index=False)
            print(f"Saved batting rankings to {batting_file}")
        
        # Save pitching rankings
        if self.pitching_rankings is not None:
            pitching_file = os.path.join(output_dir, 'pitching_rankings.csv')
            self.pitching_rankings.to_csv(pitching_file, index=False)
            print(f"Saved pitching rankings to {pitching_file}")
        
        # Save overall rankings
        if self.overall_rankings is not None:
            overall_file = os.path.join(output_dir, 'overall_rankings.csv')
            self.overall_rankings.to_csv(overall_file, index=False)
            print(f"Saved overall rankings to {overall_file}")
        
        # Save position-specific rankings
        if self.overall_rankings is not None and 'POSITIONS' in self.overall_rankings.columns:
            positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
            
            for position in positions:
                pos_players = self.get_players_by_position(position)
                if len(pos_players) > 0:
                    pos_file = os.path.join(output_dir, f'{position}_rankings.csv')
                    pos_players.to_csv(pos_file, index=False)
                    print(f"Saved {position} rankings to {pos_file}")
    
    def run_full_ranking(self, output_dir='data/rankings'):
        """
        Run a full ranking process and save results.
        
        Args:
            output_dir (str): Directory to save rankings
            
        Returns:
            pd.DataFrame: Overall player rankings
        """
        # Rank batters
        self.rank_batters()
        
        # Rank pitchers
        self.rank_pitchers()
        
        # Create overall rankings
        self.create_overall_rankings()
        
        # Save rankings
        self.save_rankings(output_dir)
        
        # Create and save plots
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Value distribution plot
        value_plot = self.plot_value_distribution()
        value_plot.savefig(os.path.join(output_dir, 'plots', 'value_distribution.png'))
        
        # Position scarcity plot (if position data is available)
        if 'POSITIONS' in self.overall_rankings.columns:
            try:
                scarcity_plot = self.plot_position_scarcity()
                scarcity_plot.savefig(os.path.join(output_dir, 'plots', 'position_scarcity.png'))
            except:
                print("Could not create position scarcity plot")
        
        print(f"Ranking complete. Results saved to {output_dir}")
        
        return self.overall_rankings


if __name__ == "__main__":
    # Example usage
    print("Player Ranker")
    print("============")
    
    # Initialize ranker
    ranker = PlayerRanker()
    
    # Load projections
    ranker.load_projections(
        batting_file='data/projections/batting_forecasts.csv',
        pitching_file='data/projections/pitching_forecasts.csv'
    )
    
    # Run ranking
    rankings = ranker.run_full_ranking()
    
    # Display top 10 overall players
    print("\nTop 10 Overall Players:")
    print(rankings.head(10)[['RANK', 'PLAYER_ID', 'PLAYER_TYPE', 'ADJ_VALUE']])
    
    print("\nRanking complete!")
