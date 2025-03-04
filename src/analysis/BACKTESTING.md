# Backtesting Forecasting Models

This document explains how to use the backtesting functionality to evaluate forecasting model performance over historical seasons.

## Overview

The backtesting system allows you to:

1. Train models using only data available before a specific season
2. Generate forecasts for that season
3. Compare forecasts against actual results
4. Analyze model performance metrics
5. Visualize the results using the existing ModelVisualizer

## Files

- `backtest_models.py`: Contains the `BacktestEvaluator` class that implements the backtesting functionality
- `run_backtests.py`: Command-line interface for running backtests

## Running Backtests

You can run backtests using the command-line interface:

```bash
python -m src.analysis.run_backtests --start-season 2018 --end-season 2023
```

### Command-line Options

- `--start-season`: First season to backtest (default: 2018)
- `--end-season`: Last season to backtest (default: 2023)
- `--output-dir`: Directory to save backtest results (default: 'data/backtests')
- `--n-jobs`: Number of parallel jobs to run (default: 8)
- `--historical-batting`: Path to historical batting statistics CSV (default: 'data/processed/batting_stats_all.csv')
- `--historical-pitching`: Path to historical pitching statistics CSV (default: 'data/processed/pitching_stats_all.csv')
- `--bio-file`: Path to biographical data CSV (default: 'data/raw/biofile0.csv')
- `--visualization-dir`: Directory to save visualization-ready data (default: 'data/projections')
- `--category`: Specific category to backtest (e.g., HR, ERA)
- `--model-type`: Specific model type to backtest (ridge, random_forest, gradient_boosting, arima, exp_smooth)

### Single Category and Model Type Backtesting

You can now run backtests for a specific category and/or model type to evaluate their performance individually:

```bash
# Test only the HR category
python -m src.analysis.run_backtests --category HR

# Test only the gradient_boosting model
python -m src.analysis.run_backtests --model-type gradient_boosting

# Test the ERA category with the random_forest model
python -m src.analysis.run_backtests --category ERA --model-type random_forest
```

This allows you to:
1. Identify which models perform best for each category
2. Reduce computational resources needed for experimentation
3. Build your ensemble model incrementally based on empirical results

## Visualizing Backtest Results

After running backtests, you can visualize the results using the existing `ModelVisualizer` class:

```bash
python -m src.analysis.visualize_models \
  --historical-batting data/projections/batting_stats_with_backtests.csv \
  --historical-pitching data/projections/pitching_stats_with_backtests.csv \
  --forecast-batting data/projections/batting_forecasts.csv \
  --forecast-pitching data/projections/pitching_forecasts.csv \
  --output-dir data/backtests/visualizations
```

## Programmatic Usage

You can also use the `BacktestEvaluator` class directly in your code:

```python
from src.analysis.backtest_models import BacktestEvaluator

# Initialize backtester
backtester = BacktestEvaluator(
    start_season=2018,
    end_season=2023,
    output_dir='data/backtests',
    category='HR',                # Optional: specific category to test
    model_type='gradient_boosting' # Optional: specific model type to test
)

# Load data
backtester.load_data(
    batting_file='data/processed/batting_stats_all.csv',
    pitching_file='data/processed/pitching_stats_all.csv',
    bio_file='data/raw/biofile0.csv'
)

# Run backtests
backtester.run_all_backtests(n_jobs=8)

# Prepare for visualization
backtester.prepare_for_visualization(output_dir='data/projections')
```

## How It Works

The backtesting process works as follows:

1. For each season in the backtest range (e.g., 2018-2023):
   - Filter the historical data to include only seasons before the target season
   - Train models on this filtered data
   - Generate forecasts for the target season
   - Compare forecasts against actual results
   - Calculate performance metrics (MSE, MAE, R²)

2. Results are saved in the following formats:
   - Individual season forecasts in `{output_dir}/season_{season}/`
   - Combined backtest results in `{output_dir}/batting_backtests.csv` and `{output_dir}/pitching_backtests.csv`
   - Performance metrics in `{output_dir}/metrics/`
   - Visualization-ready data in `{visualization_dir}/`

3. The visualization-ready data is formatted to work with the existing `ModelVisualizer` class, allowing you to use all the existing visualization functionality to analyze backtest results.
