#!/usr/bin/env python3
"""
Fantasy Baseball Draft Tool CLI

This module provides a command-line interface for the fantasy baseball draft tool.
"""

import os
import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
import click
from tqdm import tqdm

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing.process_retrosheet import process_multiple_seasons
from src.analysis.stats_analyzer import StatsAnalyzer
from src.ranking.player_ranker import PlayerRanker
from src.forecasting.forecast_generator import generate_all_forecasts
from src.forecasting.data_preparation import load_data, load_bio_data
from src.forecasting.model_trainer import load_models, train_batting_models, train_pitching_models


class DraftTool:
    """
    Command-line interface for the fantasy baseball draft tool.
    """
    
    def __init__(self):
        """
        Initialize the DraftTool.
        """
        self.rankings = None
        self.drafted_players = set()
        self.my_team = []
        self.other_teams = {}
        self.current_pick = 1
        self.total_teams = 12  # Default to 12 teams
        self.roster_size = 23  # Default roster size
        self.positions = {
            'C': 1,
            '1B': 1,
            '2B': 1,
            '3B': 1,
            'SS': 1,
            'OF': 3,
            'UTIL': 1,
            'SP': 5,
            'RP': 3,
            'P': 2,
            'BN': 4
        }
    
    def load_rankings(self, rankings_file):
        """
        Load player rankings from a CSV file.
        
        Args:
            rankings_file (str): Path to rankings CSV file
        """
        print(f"Loading rankings from {rankings_file}")
        self.rankings = pd.read_csv(rankings_file)
        
        # Ensure we have the necessary columns
        required_columns = ['RANK', 'PLAYER_ID', 'PLAYER_TYPE']
        for col in required_columns:
            if col not in self.rankings.columns:
                raise ValueError(f"Rankings file missing required column: {col}")
        
        print(f"Loaded rankings for {len(self.rankings)} players")
    
    def setup_draft(self, teams=12, roster_size=23, positions=None):
        """
        Set up the draft parameters.
        
        Args:
            teams (int): Number of teams in the draft
            roster_size (int): Number of players per team
            positions (dict, optional): Dictionary mapping positions to roster spots
        """
        self.total_teams = teams
        self.roster_size = roster_size
        
        if positions:
            self.positions = positions
        
        # Initialize other teams
        self.other_teams = {f"Team {i+1}": [] for i in range(self.total_teams - 1)}
        
        print(f"Draft set up with {self.total_teams} teams, {self.roster_size} roster spots per team")
    
    def get_pick_number(self, round_num, team_num):
        """
        Calculate the overall pick number for a given round and team.
        
        Args:
            round_num (int): Draft round number
            team_num (int): Team number
            
        Returns:
            int: Overall pick number
        """
        if round_num % 2 == 1:
            # Odd rounds go 1 to N
            return (round_num - 1) * self.total_teams + team_num
        else:
            # Even rounds go N to 1 (snake draft)
            return round_num * self.total_teams - team_num + 1
    
    def get_team_and_round(self, pick_num):
        """
        Calculate the team and round for a given overall pick number.
        
        Args:
            pick_num (int): Overall pick number
            
        Returns:
            tuple: (team_num, round_num)
        """
        round_num = (pick_num - 1) // self.total_teams + 1
        
        if round_num % 2 == 1:
            # Odd rounds go 1 to N
            team_num = (pick_num - 1) % self.total_teams + 1
        else:
            # Even rounds go N to 1 (snake draft)
            team_num = self.total_teams - (pick_num - 1) % self.total_teams
        
        return team_num, round_num
    
    def get_available_players(self, position=None, n=10):
        """
        Get the top N available players, optionally filtered by position.
        
        Args:
            position (str, optional): Filter by position
            n (int): Number of players to return
            
        Returns:
            pd.DataFrame: DataFrame with available players
        """
        if self.rankings is None:
            raise ValueError("Rankings not loaded. Call load_rankings() first.")
        
        # Filter out drafted players
        available = self.rankings[~self.rankings['PLAYER_ID'].isin(self.drafted_players)]
        
        # Filter by position if specified
        if position and 'POSITIONS' in available.columns:
            available = available[available['POSITIONS'].str.contains(position, na=False)]
        
        # Get top N available players
        top_available = available.head(n)
        
        return top_available
    
    def display_available_players(self, position=None, n=10):
        """
        Display the top N available players, optionally filtered by position.
        
        Args:
            position (str, optional): Filter by position
            n (int): Number of players to display
        """
        top_available = self.get_available_players(position, n)
        
        # Select columns to display
        display_cols = ['RANK', 'PLAYER_ID', 'PLAYER_TYPE']
        
        # Add position column if available
        if 'POSITIONS' in top_available.columns:
            display_cols.append('POSITIONS')
        
        # Add value columns if available
        if 'ADJ_VALUE' in top_available.columns:
            display_cols.append('ADJ_VALUE')
        
        # Add category columns if available
        for cat in ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB', 'ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']:
            if cat in top_available.columns:
                display_cols.append(cat)
        
        # Display the players
        print("\nTop Available Players:")
        if position:
            print(f"Position: {position}")
        
        print(tabulate(top_available[display_cols], headers='keys', tablefmt='psql', showindex=False))
    
    def draft_player(self, player_id, team_name="My Team"):
        """
        Draft a player to a team.
        
        Args:
            player_id (str): Player ID to draft
            team_name (str): Team drafting the player
            
        Returns:
            bool: True if the player was drafted successfully, False otherwise
        """
        if self.rankings is None:
            raise ValueError("Rankings not loaded. Call load_rankings() first.")
        
        # Check if player is already drafted
        if player_id in self.drafted_players:
            print(f"Player {player_id} is already drafted.")
            return False
        
        # Find the player in the rankings
        player = self.rankings[self.rankings['PLAYER_ID'] == player_id]
        
        if len(player) == 0:
            print(f"Player {player_id} not found in rankings.")
            return False
        
        # Add player to drafted players
        self.drafted_players.add(player_id)
        
        # Add player to the appropriate team
        if team_name == "My Team":
            self.my_team.append(player_id)
        elif team_name in self.other_teams:
            self.other_teams[team_name].append(player_id)
        else:
            print(f"Team {team_name} not found.")
            return False
        
        # Increment the current pick
        self.current_pick += 1
        
        print(f"Drafted {player_id} to {team_name}")
        return True
    
    def display_team(self, team_name="My Team"):
        """
        Display the players on a team.
        
        Args:
            team_name (str): Team to display
        """
        if self.rankings is None:
            raise ValueError("Rankings not loaded. Call load_rankings() first.")
        
        # Get the team's players
        if team_name == "My Team":
            team_players = self.my_team
        elif team_name in self.other_teams:
            team_players = self.other_teams[team_name]
        else:
            print(f"Team {team_name} not found.")
            return
        
        if not team_players:
            print(f"{team_name} has no players.")
            return
        
        # Get player details from rankings
        team_df = self.rankings[self.rankings['PLAYER_ID'].isin(team_players)]
        
        # Select columns to display
        display_cols = ['RANK', 'PLAYER_ID', 'PLAYER_TYPE']
        
        # Add position column if available
        if 'POSITIONS' in team_df.columns:
            display_cols.append('POSITIONS')
        
        # Add value columns if available
        if 'ADJ_VALUE' in team_df.columns:
            display_cols.append('ADJ_VALUE')
        
        # Display the team
        print(f"\n{team_name} Roster:")
        print(tabulate(team_df[display_cols], headers='keys', tablefmt='psql', showindex=False))
    
    def analyze_team_balance(self, team_name="My Team"):
        """
        Analyze the category balance of a team.
        
        Args:
            team_name (str): Team to analyze
        """
        if self.rankings is None:
            raise ValueError("Rankings not loaded. Call load_rankings() first.")
        
        # Get the team's players
        if team_name == "My Team":
            team_players = self.my_team
        elif team_name in self.other_teams:
            team_players = self.other_teams[team_name]
        else:
            print(f"Team {team_name} not found.")
            return
        
        if not team_players:
            print(f"{team_name} has no players.")
            return
        
        # Get player details from rankings
        team_df = self.rankings[self.rankings['PLAYER_ID'].isin(team_players)]
        
        # Check if we have the necessary category columns
        batting_categories = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
        pitching_categories = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']
        
        missing_categories = []
        for cat in batting_categories + pitching_categories:
            if cat not in team_df.columns:
                missing_categories.append(cat)
        
        if missing_categories:
            print(f"Missing category data for: {', '.join(missing_categories)}")
            print("Cannot perform full team balance analysis.")
            return
        
        # Separate batters and pitchers
        batters = team_df[team_df['PLAYER_TYPE'] == 'BATTER']
        pitchers = team_df[team_df['PLAYER_TYPE'] == 'PITCHER']
        
        # Calculate team totals for each category
        team_totals = {}
        
        # Batting categories (sum)
        for cat in ['HR', 'R', 'RBI', 'SB', 'TB']:
            if cat in batters.columns:
                team_totals[cat] = batters[cat].sum()
        
        # OBP (weighted average by plate appearances)
        if 'OBP' in batters.columns and 'PA' in batters.columns:
            team_totals['OBP'] = (batters['OBP'] * batters['PA']).sum() / batters['PA'].sum()
        elif 'OBP' in batters.columns:
            team_totals['OBP'] = batters['OBP'].mean()
        
        # Pitching categories
        # ERA and WHIP (weighted average by innings pitched)
        for cat in ['ERA', 'WHIP']:
            if cat in pitchers.columns and 'IP' in pitchers.columns:
                team_totals[cat] = (pitchers[cat] * pitchers['IP']).sum() / pitchers['IP'].sum()
            elif cat in pitchers.columns:
                team_totals[cat] = pitchers[cat].mean()
        
        # K, SV+HLD, W+QS (sum)
        for cat in ['K', 'SV+HLD', 'W+QS']:
            if cat in pitchers.columns:
                team_totals[cat] = pitchers[cat].sum()
        
        # Display team totals
        print(f"\n{team_name} Category Projections:")
        
        # Create a DataFrame for display
        totals_df = pd.DataFrame([team_totals])
        
        # Display batting categories
        print("\nBatting Categories:")
        batting_cols = [cat for cat in batting_categories if cat in team_totals]
        if batting_cols:
            print(tabulate(totals_df[batting_cols], headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No batting category data available.")
        
        # Display pitching categories
        print("\nPitching Categories:")
        pitching_cols = [cat for cat in pitching_categories if cat in team_totals]
        if pitching_cols:
            print(tabulate(totals_df[pitching_cols], headers='keys', tablefmt='psql', showindex=False))
        else:
            print("No pitching category data available.")
    
    def recommend_player(self, team_name="My Team"):
        """
        Recommend the best available player for a team.
        
        Args:
            team_name (str): Team to recommend for
            
        Returns:
            str: Recommended player ID
        """
        if self.rankings is None:
            raise ValueError("Rankings not loaded. Call load_rankings() first.")
        
        # Get available players
        available = self.rankings[~self.rankings['PLAYER_ID'].isin(self.drafted_players)]
        
        # Get the team's players
        if team_name == "My Team":
            team_players = self.my_team
        elif team_name in self.other_teams:
            team_players = self.other_teams[team_name]
        else:
            print(f"Team {team_name} not found.")
            return None
        
        # If the team has no players, recommend the best available player
        if not team_players:
            best_player = available.iloc[0]['PLAYER_ID']
            return best_player
        
        # Get player details from rankings
        team_df = self.rankings[self.rankings['PLAYER_ID'].isin(team_players)]
        
        # Check if we have position data
        if 'POSITIONS' in team_df.columns and 'POSITIONS' in available.columns:
            # Count positions on the team
            team_positions = []
            for positions in team_df['POSITIONS']:
                if isinstance(positions, str):
                    team_positions.extend([pos.strip() for pos in positions.split(',')])
            
            position_counts = {}
            for pos in team_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            # Find positions that need to be filled
            needed_positions = []
            for pos, count in self.positions.items():
                if pos in ['UTIL', 'P', 'BN']:
                    continue  # Skip flexible positions
                
                current_count = position_counts.get(pos, 0)
                if current_count < count:
                    needed_positions.append(pos)
            
            # If we need to fill positions, recommend the best player at a needed position
            if needed_positions:
                for pos in needed_positions:
                    pos_players = available[available['POSITIONS'].str.contains(pos, na=False)]
                    if not pos_players.empty:
                        return pos_players.iloc[0]['PLAYER_ID']
        
        # If no position needs or no position data, recommend based on value
        if 'ADJ_VALUE' in available.columns:
            # Get the best available player by adjusted value
            best_player = available.iloc[0]['PLAYER_ID']
            return best_player
        else:
            # If no value data, recommend based on rank
            best_player = available.iloc[0]['PLAYER_ID']
            return best_player
    
    def run_draft_simulation(self):
        """
        Run a simulated draft where other teams draft automatically.
        """
        if self.rankings is None:
            raise ValueError("Rankings not loaded. Call load_rankings() first.")
        
        total_picks = self.total_teams * self.roster_size
        
        print(f"\nStarting draft simulation ({total_picks} total picks)...")
        
        while self.current_pick <= total_picks:
            team_num, round_num = self.get_team_and_round(self.current_pick)
            team_name = "My Team" if team_num == 1 else f"Team {team_num}"
            
            print(f"\nPick {self.current_pick} (Round {round_num}, {team_name}):")
            
            if team_name == "My Team":
                # Display available players
                self.display_available_players(n=10)
                
                # Get user input for the pick
                while True:
                    player_input = input("\nEnter player ID to draft (or 'recommend' for a recommendation): ")
                    
                    if player_input.lower() == 'recommend':
                        recommended = self.recommend_player()
                        print(f"Recommended player: {recommended}")
                        continue
                    
                    if player_input.lower() == 'available':
                        position = input("Enter position to filter by (or leave blank for all): ")
                        position = position if position else None
                        self.display_available_players(position=position, n=10)
                        continue
                    
                    if player_input.lower() == 'team':
                        self.display_team()
                        continue
                    
                    if player_input.lower() == 'analyze':
                        self.analyze_team_balance()
                        continue
                    
                    # Try to draft the player
                    if self.draft_player(player_input):
                        break
            else:
                # Simulate other team's pick
                recommended = self.recommend_player(team_name)
                self.draft_player(recommended, team_name)
        
        print("\nDraft complete!")
        
        # Display final team
        self.display_team()
        
        # Analyze team balance
        self.analyze_team_balance()
    
    def save_draft_results(self, output_dir='data/draft'):
        """
        Save the draft results to CSV files.
        
        Args:
            output_dir (str): Directory to save draft results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save my team
        if self.my_team:
            my_team_df = self.rankings[self.rankings['PLAYER_ID'].isin(self.my_team)]
            my_team_file = os.path.join(output_dir, 'my_team.csv')
            my_team_df.to_csv(my_team_file, index=False)
            print(f"Saved my team to {my_team_file}")
        
        # Save all draft results
        all_picks = []
        
        # Add my team's picks
        for player_id in self.my_team:
            pick_num = list(self.drafted_players).index(player_id) + 1
            team_num, round_num = self.get_team_and_round(pick_num)
            all_picks.append({
                'PICK': pick_num,
                'ROUND': round_num,
                'TEAM': "My Team",
                'PLAYER_ID': player_id
            })
        
        # Add other teams' picks
        for team_name, players in self.other_teams.items():
            for player_id in players:
                pick_num = list(self.drafted_players).index(player_id) + 1
                team_num, round_num = self.get_team_and_round(pick_num)
                all_picks.append({
                    'PICK': pick_num,
                    'ROUND': round_num,
                    'TEAM': team_name,
                    'PLAYER_ID': player_id
                })
        
        # Create DataFrame and sort by pick number
        draft_df = pd.DataFrame(all_picks)
        draft_df = draft_df.sort_values('PICK')
        
        # Save draft results
        draft_file = os.path.join(output_dir, 'draft_results.csv')
        draft_df.to_csv(draft_file, index=False)
        print(f"Saved draft results to {draft_file}")


@click.group()
def cli():
    """Fantasy Baseball Draft Tool CLI"""
    pass


@cli.command()
@click.option('--start-year', type=int, default=2014, help='First season to process')
@click.option('--end-year', type=int, default=2023, help='Last season to process')
@click.option('--data-dir', default='data/raw', help='Directory containing raw Retrosheet data')
@click.option('--output-dir', default='data/processed', help='Directory to save processed data')
def process_data(start_year, end_year, data_dir, output_dir):
    """Process Retrosheet data into player statistics"""
    click.echo(f"Processing data from {start_year} to {end_year}")
    
    # Check if raw data directory exists and has files
    if not os.path.exists(data_dir):
        click.echo(f"Error: Raw data directory not found: {data_dir}")
        click.echo("Please download Retrosheet data first.")
        return
    
    # Process the data
    batting_stats, pitching_stats = process_multiple_seasons(start_year, end_year, data_dir, output_dir)
    
    click.echo("\nProcessing complete!")
    click.echo(f"Batting stats shape: {batting_stats.shape}")
    click.echo(f"Pitching stats shape: {pitching_stats.shape}")


@cli.command()
@click.option('--batting-file', default='data/processed/batting_stats_all.csv', help='Path to batting statistics CSV')
@click.option('--pitching-file', default='data/processed/pitching_stats_all.csv', help='Path to pitching statistics CSV')
@click.option('--output-dir', default='data/analysis', help='Directory to save analysis results')
def analyze_stats(batting_file, pitching_file, output_dir):
    """Analyze player statistics"""
    click.echo("Analyzing player statistics")
    
    # Check if input files exist
    if not os.path.exists(batting_file):
        click.echo(f"Error: Batting statistics file not found: {batting_file}")
        return
    
    if not os.path.exists(pitching_file):
        click.echo(f"Error: Pitching statistics file not found: {pitching_file}")
        return
    
    # Initialize analyzer
    analyzer = StatsAnalyzer()
    
    # Load data
    analyzer.load_data(batting_file, pitching_file)
    
    # Run analysis
    results = analyzer.run_full_analysis(output_dir)
    
    click.echo("\nAnalysis complete!")


@cli.command()
@click.option('--batting-file', default='data/processed/batting_stats_all.csv', help='Path to batting statistics CSV')
@click.option('--pitching-file', default='data/processed/pitching_stats_all.csv', help='Path to pitching statistics CSV')
@click.option('--bio-file', default='data/raw/biofile0.csv', help='Path to player biographical data CSV (optional)')
@click.option('--models-dir', default='data/models', help='Directory to save trained models')
@click.option('--model-type', default='ensemble', help='Type of model to train (ensemble, ridge, random_forest, gradient_boosting)')
@click.option('--n-jobs', type=int, default=8, help='Number of parallel jobs to run')
def train_models(batting_file, pitching_file, bio_file, models_dir, model_type, n_jobs):
    """Train forecasting models for player performance"""
    click.echo("Training player forecasting models")
    
    # Check if input files exist
    if not os.path.exists(batting_file):
        click.echo(f"Error: Batting statistics file not found: {batting_file}")
        return
    
    if not os.path.exists(pitching_file):
        click.echo(f"Error: Pitching statistics file not found: {pitching_file}")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    batting_data, pitching_data = load_data(batting_file, pitching_file)
    
    # Load biographical data if file exists
    bio_file = load_bio_data(bio_file)
    
    # Train models
    click.echo("\nTraining batting models...")
    train_batting_models(
        batting_data=batting_data,
        model_type=model_type, 
        bio_data=bio_file,
        n_jobs=n_jobs)
    
    click.echo("\nTraining pitching models...")
    train_pitching_models(
        pitching_data=pitching_data,
        model_type=model_type, 
        bio_data=bio_file,
        n_jobs=n_jobs)
    
    click.echo("\nModel training complete!")
    click.echo(f"Models saved to {models_dir}")


@cli.command()
@click.option('--batting-file', default='data/processed/batting_stats_all.csv', help='Path to batting statistics CSV')
@click.option('--pitching-file', default='data/processed/pitching_stats_all.csv', help='Path to pitching statistics CSV')
@click.option('--bio-file', default='data/raw/biofile0.csv', help='Path to player biographical data CSV')
@click.option('--models-dir', default='data/models', help='Directory containing saved models')
@click.option('--output-dir', default='data/projections', help='Directory to save projections')
@click.option('--n-jobs', type=int, default=8, help='Number of parallel jobs to run')
@click.option('--forecast-season', type=int, default=None, help='Season to forecast (default: most recent season + 1)')
def generate_forecasts(batting_file, pitching_file, bio_file, models_dir, output_dir, n_jobs, forecast_season):
    """Generate player forecasts using trained models"""
    click.echo("Generating player forecasts")
    
    # Check if input files exist
    if not os.path.exists(batting_file):
        click.echo(f"Error: Batting statistics file not found: {batting_file}")
        return
    
    if not os.path.exists(pitching_file):
        click.echo(f"Error: Pitching statistics file not found: {pitching_file}")
        return
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        click.echo(f"Error: Models directory not found: {models_dir}")
        click.echo("Please train models first or specify a different models directory.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize forecaster with existing models
    batting_models, pitching_models = load_models(models_dir=models_dir)
    
    # Load data
    batting_data, pitching_data = load_data(batting_file, pitching_file)
    
    # Load biographical data if file exists
    bio_file = load_bio_data(bio_file)
    
    # Generate forecasts
    click.echo("\nGenerating forecasts...")
    batting_forecasts, pitching_forecasts = generate_all_forecasts(
        batting_data=batting_data,
        pitching_data=pitching_data,
        batting_models=batting_models,
        pitching_models=pitching_models,
        bio_data=bio_file,
        output_dir=output_dir, 
        n_jobs=n_jobs, 
        forecast_season=forecast_season)
    
    click.echo("\nForecasting complete!")
    click.echo(f"Generated forecasts for {len(batting_forecasts)} batters and {len(pitching_forecasts)} pitchers")
    click.echo(f"Forecasts saved to {output_dir}")


@cli.command()
@click.option('--batting-file', default='data/projections/batting_forecasts.csv', help='Path to batting projections CSV')
@click.option('--pitching-file', default='data/projections/pitching_forecasts.csv', help='Path to pitching projections CSV')
@click.option('--positions-file', default=None, help='Path to player positions CSV (optional)')
@click.option('--output-dir', default='data/rankings', help='Directory to save rankings')
def rank_players(batting_file, pitching_file, positions_file, output_dir):
    """Rank players based on projected fantasy value"""
    click.echo("Ranking players")
    
    # Check if input files exist
    if not os.path.exists(batting_file):
        click.echo(f"Error: Batting projections file not found: {batting_file}")
        return
    
    if not os.path.exists(pitching_file):
        click.echo(f"Error: Pitching projections file not found: {pitching_file}")
        return
    
    # Initialize ranker
    ranker = PlayerRanker()
    
    # Load projections
    ranker.load_projections(batting_file, pitching_file)
    
    # Load player positions if provided
    if positions_file and os.path.exists(positions_file):
        ranker.load_player_positions(positions_file)
    
    # Run ranking
    rankings = ranker.run_full_ranking(output_dir)
    
    click.echo("\nRanking complete!")
    click.echo(f"Ranked {len(rankings)} players")
    
    # Display top 20 overall players
    click.echo("\nTop 20 Overall Players:")
    click.echo(tabulate(rankings.head(20)[['RANK', 'PLAYER_ID', 'PLAYER_NAME', 'PLAYER_TYPE', 'ADJ_VALUE']], 
                       headers='keys', tablefmt='psql', showindex=False))


@cli.command()
@click.option('--rankings-file', default='data/rankings/overall_rankings.csv', help='Path to player rankings CSV')
@click.option('--teams', type=int, default=12, help='Number of teams in the draft')
@click.option('--roster-size', type=int, default=23, help='Number of players per team')
@click.option('--output-dir', default='data/draft', help='Directory to save draft results')
def draft(rankings_file, teams, roster_size, output_dir):
    """Run a fantasy baseball draft simulation"""
    click.echo("Fantasy Baseball Draft Tool")
    
    # Check if rankings file exists
    if not os.path.exists(rankings_file):
        click.echo(f"Error: Rankings file not found: {rankings_file}")
        return
    
    # Initialize draft tool
    draft_tool = DraftTool()
    
    # Load rankings
    draft_tool.load_rankings(rankings_file)
    
    # Set up draft
    draft_tool.setup_draft(teams=teams, roster_size=roster_size)
    
    # Run draft simulation
    draft_tool.run_draft_simulation()
    
    # Save draft results
    draft_tool.save_draft_results(output_dir)


@cli.command()
@click.option('--player-id', type=str, help='Player ID to visualize')
@click.option('--category', type=str, help='Statistical category to visualize')
@click.option('--is-pitcher', is_flag=True, help='Whether the player is a pitcher')
@click.option('--output-dir', default='data/visualizations', help='Directory to save visualizations')
@click.option('--historical-batting', default='data/processed/batting_stats_all.csv', 
              help='Path to historical batting statistics CSV')
@click.option('--historical-pitching', default='data/processed/pitching_stats_all.csv', 
              help='Path to historical pitching statistics CSV')
@click.option('--forecast-batting', default='data/projections/batting_forecasts.csv', 
              help='Path to batting forecasts CSV')
@click.option('--forecast-pitching', default='data/projections/pitching_forecasts.csv', 
              help='Path to pitching forecasts CSV')
@click.option('--top-n', type=int, default=5, help='Number of top players to visualize')
@click.option('--all-categories', is_flag=True, help='Visualize all categories')
def visualize_models(player_id, category, is_pitcher, output_dir, historical_batting, historical_pitching, 
                    forecast_batting, forecast_pitching, top_n, all_categories):
    """Visualize model performance and forecasts"""
    click.echo("Visualizing model performance")
    
    # Check if input files exist
    if not os.path.exists(historical_batting):
        click.echo(f"Error: Historical batting file not found: {historical_batting}")
        return
    
    if not os.path.exists(historical_pitching):
        click.echo(f"Error: Historical pitching file not found: {historical_pitching}")
        return
    
    # Check if forecast files exist
    if not os.path.exists(forecast_batting):
        click.echo(f"Warning: Forecast batting file not found: {forecast_batting}")
        click.echo("Will proceed without forecast data.")
        forecast_batting = None
    
    if not os.path.exists(forecast_pitching):
        click.echo(f"Warning: Forecast pitching file not found: {forecast_pitching}")
        click.echo("Will proceed without forecast data.")
        forecast_pitching = None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Import the visualizer
    from src.analysis.model_visualizer import ModelVisualizer
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Load data
    click.echo("Loading data...")
    visualizer.load_data(
        historical_batting_file=historical_batting,
        historical_pitching_file=historical_pitching,
        forecast_batting_file=forecast_batting,
        forecast_pitching_file=forecast_pitching
    )
    
    # If player ID and category are specified, visualize that player
    if player_id and category:
        click.echo(f"Visualizing {category} for player {player_id}...")
        visualizer.run_model_comparison(
            player_id, category, is_pitcher, output_dir
        )
    # Otherwise, visualize top players for each category
    else:
        # Get categories based on player type
        if is_pitcher:
            categories = visualizer.pitching_categories
            data = visualizer.historical_pitching
        else:
            categories = visualizer.batting_categories
            data = visualizer.historical_batting
        
        # If all_categories is False, just use the first category
        if not all_categories:
            categories = [categories[0]]
        
        # For each category, visualize top players
        for category in categories:
            click.echo(f"Visualizing {category} for top {top_n} players...")
            
            # Get top players by category value
            player_stats = data.groupby('PLAYER_ID')[category].mean().sort_values(ascending=False)
            top_players = player_stats.head(top_n).index.tolist()
            
            # Visualize each player
            for player_id in top_players:
                try:
                    click.echo(f"  Visualizing {category} for player {player_id}...")
                    visualizer.run_model_comparison(
                        player_id, category, is_pitcher, output_dir
                    )
                except Exception as e:
                    click.echo(f"  Error visualizing {category} for player {player_id}: {e}")
    
    click.echo(f"Visualizations saved to {output_dir}")


@cli.command()
@click.option('--load-models', is_flag=True, help='Load existing models instead of training new ones')
@click.option('--models-dir', default='data/models', help='Directory containing saved models (when using --load-models)')
@click.option('--forecast-season', type=int, default=None, help='Season to forecast (default: most recent season + 1)')
def run_pipeline(load_models, models_dir, forecast_season):
    """Run the complete fantasy baseball analysis pipeline"""
    click.echo("Running complete fantasy baseball analysis pipeline")
    
    # Process data
    click.echo("\n1. Processing data...")
    process_data.callback(2014, 2023, 'data/raw', 'data/processed')
    
    # Analyze statistics
    click.echo("\n2. Analyzing statistics...")
    analyze_stats.callback('data/processed/batting_stats_all.csv', 'data/processed/pitching_stats_all.csv', 'data/analysis')
    
    # Train models or load existing ones
    click.echo("\n3. Training models or loading existing ones...")
    if load_models:
        click.echo("Using existing models...")
    else:
        click.echo("Training new models...")
        train_models.callback(
            'data/processed/batting_stats_all.csv',
            'data/processed/pitching_stats_all.csv',
            'data/raw/biofile0.csv',
            models_dir,
            'ensemble',
            8
        )
    
    # Generate forecasts
    click.echo("\n4. Generating forecasts...")
    generate_forecasts.callback(
        'data/processed/batting_stats_all.csv',
        'data/processed/pitching_stats_all.csv',
        'data/raw/biofile0.csv',
        models_dir,
        'data/projections',
        8,
        forecast_season
    )
    
    # Rank players
    click.echo("\n5. Ranking players...")
    rank_players.callback('data/projections/batting_forecasts.csv', 'data/projections/pitching_forecasts.csv', None, 'data/rankings')
    
    # Visualize model performance
    click.echo("\n6. Visualizing model performance...")
    visualize_models.callback(
        None, None, False, 'data/visualizations',
        'data/processed/batting_stats_all.csv',
        'data/processed/pitching_stats_all.csv',
        'data/projections/batting_forecasts.csv',
        'data/projections/pitching_forecasts.csv',
        3, True
    )
    
    click.echo("\nPipeline complete!")


if __name__ == "__main__":
    cli()
