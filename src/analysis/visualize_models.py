#!/usr/bin/env python3
"""
Model Visualization Script

This script demonstrates how to use the ModelVisualizer class to
visualize and compare forecasting model performance.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.model_visualizer import ModelVisualizer


def main():
    """
    Main entry point for the model visualization script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize model performance')
    parser.add_argument('--player-id', type=str, help='Player ID to visualize')
    parser.add_argument('--category', type=str, help='Statistical category to visualize')
    parser.add_argument('--is-pitcher', action='store_true', help='Whether the player is a pitcher')
    parser.add_argument('--output-dir', default='data/visualizations', help='Directory to save visualizations')
    parser.add_argument('--historical-batting', default='data/processed/batting_stats_all.csv', 
                        help='Path to historical batting statistics CSV')
    parser.add_argument('--historical-pitching', default='data/processed/pitching_stats_all.csv', 
                        help='Path to historical pitching statistics CSV')
    parser.add_argument('--forecast-batting', default='data/projections/batting_forecasts.csv', 
                        help='Path to batting forecasts CSV')
    parser.add_argument('--forecast-pitching', default='data/projections/pitching_forecasts.csv', 
                        help='Path to pitching forecasts CSV')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top players to visualize')
    parser.add_argument('--all-categories', action='store_true', help='Visualize all categories')
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Load data
    print("Loading data...")
    visualizer.load_data(
        historical_batting_file=args.historical_batting,
        historical_pitching_file=args.historical_pitching,
        forecast_batting_file=args.forecast_batting,
        forecast_pitching_file=args.forecast_pitching
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If player ID and category are specified, visualize that player
    if args.player_id and args.category:
        print(f"Visualizing {args.category} for player {args.player_id}...")
        visualizer.run_model_comparison(
            args.player_id, args.category, args.is_pitcher, args.output_dir
        )
    # Otherwise, visualize top players for each category
    else:
        # Get categories based on player type
        if args.is_pitcher:
            categories = visualizer.pitching_categories
            data = visualizer.historical_pitching
        else:
            categories = visualizer.batting_categories
            data = visualizer.historical_batting
        
        # If all_categories is False, just use the first category
        if not args.all_categories:
            categories = [categories[0]]
        
        # For each category, visualize top players
        for category in categories:
            print(f"Visualizing {category} for top {args.top_n} players...")
            
            # Get top players by category value
            player_stats = data.groupby('PLAYER_ID')[category].mean().sort_values(ascending=False)
            top_players = player_stats.head(args.top_n).index.tolist()
            
            # Visualize each player
            for player_id in top_players:
                try:
                    print(f"  Visualizing {category} for player {player_id}...")
                    visualizer.run_model_comparison(
                        player_id, category, args.is_pitcher, args.output_dir
                    )
                except Exception as e:
                    print(f"  Error visualizing {category} for player {player_id}: {e}")
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    print("Model Performance Visualizer")
    print("===========================")
    main()
