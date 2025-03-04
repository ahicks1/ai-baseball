#!/usr/bin/env python3
"""
Backtesting Script for Player Performance Forecasting

This script provides a command-line interface for running backtests
to evaluate forecasting model performance over historical seasons.
"""

import os
import argparse
from src.analysis.backtest_models import BacktestEvaluator


def main():
    """
    Main entry point for the backtesting script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Backtest forecasting models')
    parser.add_argument('--start-season', type=int, default=2018, help='First season to backtest')
    parser.add_argument('--end-season', type=int, default=2023, help='Last season to backtest')
    parser.add_argument('--output-dir', default='data/backtests', help='Directory to save backtest results')
    parser.add_argument('--n-jobs', type=int, default=8, help='Number of parallel jobs to run')
    parser.add_argument('--historical-batting', default='data/processed/batting_stats_all.csv', 
                        help='Path to historical batting statistics CSV')
    parser.add_argument('--historical-pitching', default='data/processed/pitching_stats_all.csv', 
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
    
    # Print configuration information
    config_info = [f"Running backtests for seasons {args.start_season} to {args.end_season}"]
    if args.category:
        config_info.append(f"Category: {args.category}")
    if args.model_type:
        config_info.append(f"Model type: {args.model_type}")
    print(", ".join(config_info) + "...")
    
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
        batting_file=args.historical_batting,
        pitching_file=args.historical_pitching,
        bio_file=args.bio_file
    )
    
    # Run backtests
    backtester.run_all_backtests(n_jobs=args.n_jobs)
    
    # Prepare for visualization
    backtester.prepare_for_visualization(output_dir=args.visualization_dir)
    
    print(f"Backtest results saved to {args.output_dir}")
    print(f"Visualization-ready data saved to {args.visualization_dir}")
    print("\nTo visualize the results, run:")
    print(f"python -m src.analysis.visualize_models --historical-batting {args.visualization_dir}/batting_stats_with_backtests.csv " +
          f"--historical-pitching {args.visualization_dir}/pitching_stats_with_backtests.csv " +
          f"--forecast-batting {args.visualization_dir}/batting_forecasts.csv " +
          f"--forecast-pitching {args.visualization_dir}/pitching_forecasts.csv " +
          f"--output-dir {args.output_dir}/visualizations")


if __name__ == "__main__":
    print("Forecasting Model Backtester")
    print("===========================")
    main()
