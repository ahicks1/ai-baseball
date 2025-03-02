#!/usr/bin/env python3
"""
Model Visualization Module

This module provides functions for visualizing and comparing
forecasting model performance for baseball player statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelVisualizer:
    """
    Class for visualizing and comparing forecasting model performance.
    """
    
    def __init__(self):
        """
        Initialize the ModelVisualizer.
        """
        # League settings for H2H categories
        self.batting_categories = ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB']
        self.pitching_categories = ['ERA', 'WHIP', 'K', 'SV+HLD', 'W+QS']
        
        # Data storage
        self.historical_batting = None
        self.historical_pitching = None
        self.forecast_batting = None
        self.forecast_pitching = None
        
        # Model names
        self.model_names = ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting']
        
        # Set up plotting style
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """
        Set up the plotting style for visualizations.
        """
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Define a color palette for models
        self.model_colors = {
            'arima': '#1f77b4',  # blue
            'exp_smooth': '#ff7f0e',  # orange
            'ridge': '#2ca02c',  # green
            'random_forest': '#d62728',  # red
            'gradient_boosting': '#9467bd',  # purple
            'combined': '#8c564b',  # brown
            'actual': '#000000'  # black
        }
    
    def load_data(self, historical_batting_file=None, historical_pitching_file=None, 
                 forecast_batting_file=None, forecast_pitching_file=None):
        """
        Load historical and forecast data from CSV files.
        
        Args:
            historical_batting_file (str, optional): Path to historical batting statistics CSV
            historical_pitching_file (str, optional): Path to historical pitching statistics CSV
            forecast_batting_file (str, optional): Path to batting forecasts CSV
            forecast_pitching_file (str, optional): Path to pitching forecasts CSV
        """
        # Load historical batting data
        if historical_batting_file:
            print(f"Loading historical batting data from {historical_batting_file}")
            self.historical_batting = pd.read_csv(historical_batting_file)
            print(f"Loaded {len(self.historical_batting)} historical batting records")
        
        # Load historical pitching data
        if historical_pitching_file:
            print(f"Loading historical pitching data from {historical_pitching_file}")
            self.historical_pitching = pd.read_csv(historical_pitching_file)
            print(f"Loaded {len(self.historical_pitching)} historical pitching records")
        
        # Load forecast batting data
        if forecast_batting_file:
            print(f"Loading batting forecasts from {forecast_batting_file}")
            self.forecast_batting = pd.read_csv(forecast_batting_file)
            print(f"Loaded {len(self.forecast_batting)} batting forecasts")
        
        # Load forecast pitching data
        if forecast_pitching_file:
            print(f"Loading pitching forecasts from {forecast_pitching_file}")
            self.forecast_pitching = pd.read_csv(forecast_pitching_file)
            print(f"Loaded {len(self.forecast_pitching)} pitching forecasts")
    
    def load_model_predictions(self, player_id, category, is_pitcher=False):
        """
        Load model predictions for a specific player and category.
        
        Args:
            player_id (str): Player ID
            category (str): Statistical category
            is_pitcher (bool): Whether the player is a pitcher
            
        Returns:
            dict: Dictionary with model predictions
        """
        # This is a placeholder function. In a real implementation, you would
        # extract model-specific predictions from your forecast data.
        # For now, we'll simulate this with random data.
        
        # In a real implementation, you would:
        # 1. Load the model files for the specific category
        # 2. Get the player's historical data
        # 3. Use each model to make predictions
        # 4. Return the predictions from each model
        
        # Get the player's historical data
        if is_pitcher:
            if self.historical_pitching is None:
                raise ValueError("Historical pitching data not loaded. Call load_data() first.")
            player_data = self.historical_pitching[self.historical_pitching['PLAYER_ID'] == player_id]
        else:
            if self.historical_batting is None:
                raise ValueError("Historical batting data not loaded. Call load_data() first.")
            player_data = self.historical_batting[self.historical_batting['PLAYER_ID'] == player_id]
        
        if len(player_data) == 0:
            raise ValueError(f"No historical data found for player {player_id}")
        
        # Sort by season
        player_data = player_data.sort_values('SEASON')
        
        # Get the actual values for the category
        actual_values = player_data[category].values
        
        # Initialize predictions with actual values only
        predictions = {
            'actual': actual_values
        }
        
        # We'll only add model predictions if they're actually available
        # No simulated values will be added
        
        # Add a forecast for the next season
        seasons = player_data['SEASON'].values
        next_season = seasons[-1] + 1
        
        # Initialize an empty forecast dictionary
        forecast = {}
        
        # Get the forecast for the next season
        if is_pitcher:
            if self.forecast_pitching is not None:
                # Get the forecast from the loaded data
                player_forecast = self.forecast_pitching[self.forecast_pitching['PLAYER_ID'] == player_id]
                if len(player_forecast) > 0:
                    # Get the combined forecast if available
                    if category in player_forecast.columns:
                        forecast['combined'] = player_forecast[category].values[0]
                    
                    # Extract model-specific predictions if available
                    for model_name in ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting']:
                        model_column = f"{category}_{model_name}"
                        if model_column in player_forecast.columns:
                            forecast[model_name] = player_forecast[model_column].values[0]
        else:
            if self.forecast_batting is not None:
                # Get the forecast from the loaded data
                player_forecast = self.forecast_batting[self.forecast_batting['PLAYER_ID'] == player_id]
                if len(player_forecast) > 0:
                    # Get the combined forecast if available
                    if category in player_forecast.columns:
                        forecast['combined'] = player_forecast[category].values[0]
                    
                    # Extract model-specific predictions if available
                    for model_name in ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting']:
                        model_column = f"{category}_{model_name}"
                        if model_column in player_forecast.columns:
                            forecast[model_name] = player_forecast[model_column].values[0]
        
        return {
            'player_id': player_id,
            'category': category,
            'seasons': np.append(seasons, next_season),
            'predictions': predictions,
            'forecast': forecast
        }
    
    def compare_model_predictions(self, player_id, category, is_pitcher=False, models=None, save_path=None):
        """
        Compare predictions from different models for a specific player and category.
        
        Args:
            player_id (str): Player ID
            category (str): Statistical category
            is_pitcher (bool): Whether the player is a pitcher
            models (list, optional): List of models to compare
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Load model predictions
        data = self.load_model_predictions(player_id, category, is_pitcher)
        
        # Get player name
        if is_pitcher:
            if self.historical_pitching is None:
                player_name = player_id
            else:
                player_data = self.historical_pitching[self.historical_pitching['PLAYER_ID'] == player_id]
                if len(player_data) > 0 and 'first' in player_data.columns and 'last' in player_data.columns:
                    player_name = f"{player_data['first'].iloc[0]} {player_data['last'].iloc[0]}"
                else:
                    player_name = player_id
        else:
            if self.historical_batting is None:
                player_name = player_id
            else:
                player_data = self.historical_batting[self.historical_batting['PLAYER_ID'] == player_id]
                if len(player_data) > 0 and 'first' in player_data.columns and 'last' in player_data.columns:
                    player_name = f"{player_data['first'].iloc[0]} {player_data['last'].iloc[0]}"
                else:
                    player_name = player_id
        
        # Filter models if specified
        if models is None:
            # Only include models that have predictions
            models = [model for model in ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting', 'combined'] 
                     if model in data['predictions']]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot historical data
        seasons = data['seasons'][:-1]  # Exclude forecast season
        ax.plot(seasons, data['predictions']['actual'], marker='o', linestyle='-', 
                color=self.model_colors['actual'], linewidth=2, markersize=8, label='Actual')
        
        # Plot model predictions
        for model in models:
            if model in data['predictions']:
                ax.plot(seasons, data['predictions'][model], marker='x', linestyle='--', 
                        color=self.model_colors.get(model, 'gray'), linewidth=1.5, markersize=6, label=model.title())
        
        # Plot forecast for next season
        next_season = data['seasons'][-1]
        for model in models:
            if model in data['forecast']:
                ax.scatter(next_season, data['forecast'][model], marker='s', 
                          color=self.model_colors.get(model, 'gray'), s=100, label=f"{model.title()} Forecast")
        
        # Add labels and title
        ax.set_xlabel('Season', fontsize=14)
        ax.set_ylabel(category, fontsize=14)
        ax.set_title(f"{player_name} - {category} Predictions by Model", fontsize=16)
        
        # Add legend
        ax.legend(loc='best', fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def model_accuracy_comparison(self, category, is_pitcher=False, metric='mse', top_n=10, save_path=None):
        """
        Compare accuracy metrics across models for a specific category.
        
        Args:
            category (str): Statistical category
            is_pitcher (bool): Whether to use pitching data
            metric (str): Metric to use for comparison ('mse', 'mae', or 'r2')
            top_n (int): Number of top players to include
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get the appropriate data
        if is_pitcher:
            if self.historical_pitching is None:
                raise ValueError("Historical pitching data not loaded. Call load_data() first.")
            data = self.historical_pitching
        else:
            if self.historical_batting is None:
                raise ValueError("Historical batting data not loaded. Call load_data() first.")
            data = self.historical_batting
        
        # Get top players by category value
        player_stats = data.groupby('PLAYER_ID')[category].mean().sort_values(ascending=False)
        top_players = player_stats.head(top_n).index.tolist()
        
        # Calculate accuracy metrics for each model and player
        all_models = ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting', 'combined']
        metrics = {model: [] for model in all_models}
        available_models = set()
        
        for player_id in top_players:
            try:
                # Load model predictions
                pred_data = self.load_model_predictions(player_id, category, is_pitcher)
                
                # Calculate metrics
                actual = pred_data['predictions']['actual']
                
                for model in all_models:
                    if model in pred_data['predictions']:
                        available_models.add(model)
                        predicted = pred_data['predictions'][model]
                        
                        if metric == 'mse':
                            # Mean Squared Error (lower is better)
                            value = mean_squared_error(actual, predicted)
                        elif metric == 'mae':
                            # Mean Absolute Error (lower is better)
                            value = mean_absolute_error(actual, predicted)
                        elif metric == 'r2':
                            # R-squared (higher is better)
                            value = r2_score(actual, predicted)
                        else:
                            raise ValueError(f"Unknown metric: {metric}")
                        
                        metrics[model].append(value)
            except Exception as e:
                print(f"Error calculating metrics for player {player_id}: {e}")
        
        # Calculate average metrics
        avg_metrics = {model: np.mean(values) if values else np.nan for model, values in metrics.items()}
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bar chart - only include models that have data
        models_to_plot = [model for model in available_models if not np.isnan(avg_metrics[model])]
        x = np.arange(len(models_to_plot))
        
        # For R-squared, higher is better; for MSE and MAE, lower is better
        if metric == 'r2':
            values = [avg_metrics[model] for model in models_to_plot]
            title_suffix = "Higher is Better"
        else:
            values = [avg_metrics[model] for model in models_to_plot]
            title_suffix = "Lower is Better"
        
        # Plot bars
        bars = ax.bar(x, values, width=0.6, alpha=0.8)
        
        # Color bars by model
        for i, model in enumerate(models_to_plot):
            bars[i].set_color(self.model_colors.get(model, 'gray'))
        
        # Add labels and title
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel(f"{metric.upper()}", fontsize=14)
        ax.set_title(f"Model Accuracy Comparison - {category} ({metric.upper()}) - {title_suffix}", fontsize=16)
        
        # Set x-tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([model.title() for model in models_to_plot], rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v * 1.05, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def actual_vs_predicted(self, model_name, category, is_pitcher=False, save_path=None):
        """
        Create a scatter plot of actual vs. predicted values for a specific model.
        
        Args:
            model_name (str): Name of the model
            category (str): Statistical category
            is_pitcher (bool): Whether to use pitching data
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get the appropriate data
        if is_pitcher:
            if self.historical_pitching is None:
                raise ValueError("Historical pitching data not loaded. Call load_data() first.")
            data = self.historical_pitching
        else:
            if self.historical_batting is None:
                raise ValueError("Historical batting data not loaded. Call load_data() first.")
            data = self.historical_batting
        
        # Get all players
        player_ids = data['PLAYER_ID'].unique()
        
        # Collect actual and predicted values
        actual_values = []
        predicted_values = []
        
        for player_id in player_ids:
            try:
                # Load model predictions
                pred_data = self.load_model_predictions(player_id, category, is_pitcher)
                
                # Get actual and predicted values
                actual = pred_data['predictions']['actual']
                
                if model_name in pred_data['predictions']:
                    predicted = pred_data['predictions'][model_name]
                    
                    # Add to lists
                    actual_values.extend(actual)
                    predicted_values.extend(predicted)
            except Exception as e:
                # Skip players with errors
                continue
                
        # If no predictions are available for this model, raise an error
        if len(actual_values) == 0:
            raise ValueError(f"No predictions available for model {model_name}")
        
        # Convert to numpy arrays
        actual_values = np.array(actual_values)
        predicted_values = np.array(predicted_values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot
        ax.scatter(actual_values, predicted_values, alpha=0.5, 
                  color=self.model_colors.get(model_name, 'blue'), s=50)
        
        # Add perfect prediction line
        min_val = min(np.min(actual_values), np.min(predicted_values))
        max_val = max(np.max(actual_values), np.max(predicted_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect Prediction')
        
        # Add regression line
        if len(actual_values) > 1:
            z = np.polyfit(actual_values, predicted_values, 1)
            p = np.poly1d(z)
            ax.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])), 
                   'r-', linewidth=1.5, label=f'Regression Line (y = {z[0]:.3f}x + {z[1]:.3f})')
        
        # Add labels and title
        ax.set_xlabel('Actual Values', fontsize=14)
        ax.set_ylabel('Predicted Values', fontsize=14)
        ax.set_title(f"{model_name.title()} - Actual vs. Predicted {category}", fontsize=16)
        
        # Add legend
        ax.legend(loc='best', fontsize=12)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def error_distribution(self, category, is_pitcher=False, save_path=None):
        """
        Create box plots showing error distributions across models.
        
        Args:
            category (str): Statistical category
            is_pitcher (bool): Whether to use pitching data
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Get the appropriate data
        if is_pitcher:
            if self.historical_pitching is None:
                raise ValueError("Historical pitching data not loaded. Call load_data() first.")
            data = self.historical_pitching
        else:
            if self.historical_batting is None:
                raise ValueError("Historical batting data not loaded. Call load_data() first.")
            data = self.historical_batting
        
        # Get all players
        player_ids = data['PLAYER_ID'].unique()
        
        # Collect errors for each model
        all_models = ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting', 'combined']
        errors = {model: [] for model in all_models}
        
        for player_id in player_ids:
            try:
                # Load model predictions
                pred_data = self.load_model_predictions(player_id, category, is_pitcher)
                
                # Get actual values
                actual = pred_data['predictions']['actual']
                
                # Calculate errors for each model
                for model in all_models:
                    if model in pred_data['predictions']:
                        predicted = pred_data['predictions'][model]
                        
                        # Calculate errors (predicted - actual)
                        model_errors = predicted - actual
                        
                        # Add to list
                        errors[model].extend(model_errors)
            except Exception as e:
                # Skip players with errors
                continue
        
        # Filter out models with no errors
        models_to_plot = [model for model in all_models if errors[model]]
        
        # If no models have errors, raise an error
        if len(models_to_plot) == 0:
            raise ValueError(f"No model predictions available for category {category}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create box plots
        box_data = [errors[model] for model in models_to_plot]
        box_colors = [self.model_colors.get(model, 'gray') for model in models_to_plot]
        
        # Plot box plots
        box = ax.boxplot(box_data, patch_artist=True, showfliers=True, 
                        flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3})
        
        # Color boxes
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add labels and title
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Error (Predicted - Actual)', fontsize=14)
        ax.set_title(f"Error Distribution Across Models - {category}", fontsize=16)
        
        # Set x-tick labels
        ax.set_xticks(np.arange(1, len(models_to_plot) + 1))
        ax.set_xticklabels([model.title() for model in models_to_plot], rotation=45, ha='right')
        
        # Add horizontal line at y=0 (no error)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_visualization(self, fig, filename, directory='data/visualizations'):
        """
        Save a visualization to a file.
        
        Args:
            fig (matplotlib.figure.Figure): The figure to save
            filename (str): Filename for the saved visualization
            directory (str): Directory to save the visualization
            
        Returns:
            str: Path to the saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Create full path
        path = os.path.join(directory, filename)
        
        # Save figure
        fig.savefig(path, dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to {path}")
        return path
    
    def run_model_comparison(self, player_id, category, is_pitcher=False, output_dir='data/visualizations'):
        """
        Run a comprehensive model comparison for a specific player and category.
        
        Args:
            player_id (str): Player ID
            category (str): Statistical category
            is_pitcher (bool): Whether the player is a pitcher
            output_dir (str): Directory to save visualizations
            
        Returns:
            dict: Dictionary with paths to saved visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get player name
        if is_pitcher:
            if self.historical_pitching is None:
                player_name = player_id
            else:
                player_data = self.historical_pitching[self.historical_pitching['PLAYER_ID'] == player_id]
                if len(player_data) > 0 and 'first' in player_data.columns and 'last' in player_data.columns:
                    player_name = f"{player_data['first'].iloc[0]} {player_data['last'].iloc[0]}"
                else:
                    player_name = player_id
        else:
            if self.historical_batting is None:
                player_name = player_id
            else:
                player_data = self.historical_batting[self.historical_batting['PLAYER_ID'] == player_id]
                if len(player_data) > 0 and 'first' in player_data.columns and 'last' in player_data.columns:
                    player_name = f"{player_data['first'].iloc[0]} {player_data['last'].iloc[0]}"
                else:
                    player_name = player_id
        
        # Generate visualizations
        results = {}
        
        # 1. Model predictions comparison
        print(f"Generating model predictions comparison for {player_name} - {category}...")
        pred_fig = self.compare_model_predictions(
            player_id, category, is_pitcher,
            save_path=os.path.join(output_dir, f"{player_id}_{category}_predictions.png")
        )
        results['predictions'] = os.path.join(output_dir, f"{player_id}_{category}_predictions.png")
        
        # 2. Model accuracy comparison
        print(f"Generating model accuracy comparison for {category}...")
        acc_fig = self.model_accuracy_comparison(
            category, is_pitcher, metric='mse',
            save_path=os.path.join(output_dir, f"{category}_model_accuracy_mse.png")
        )
        results['accuracy_mse'] = os.path.join(output_dir, f"{category}_model_accuracy_mse.png")
        
        # 3. Actual vs. predicted for each model
        for model in ['arima', 'exp_smooth', 'ridge', 'random_forest', 'gradient_boosting', 'combined']:
            print(f"Generating actual vs. predicted for {model} - {category}...")
            try:
                avp_fig = self.actual_vs_predicted(
                    model, category, is_pitcher,
                    save_path=os.path.join(output_dir, f"{category}_{model}_actual_vs_predicted.png")
                )
                results[f'actual_vs_predicted_{model}'] = os.path.join(output_dir, f"{category}_{model}_actual_vs_predicted.png")
            except Exception as e:
                print(f"Error generating actual vs. predicted for {model}: {e}")
        
        # 4. Error distribution
        print(f"Generating error distribution for {category}...")
        err_fig = self.error_distribution(
            category, is_pitcher,
            save_path=os.path.join(output_dir, f"{category}_error_distribution.png")
        )
        results['error_distribution'] = os.path.join(output_dir, f"{category}_error_distribution.png")
        
        print(f"Model comparison complete. Results saved to {output_dir}")
        return results


if __name__ == "__main__":
    # Example usage
    print("Model Visualizer")
    print("===============")
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Load data
    visualizer.load_data(
        historical_batting_file='data/processed/batting_stats_all.csv',
        historical_pitching_file='data/processed/pitching_stats_all.csv',
        forecast_batting_file='data/projections/batting_forecasts.csv',
        forecast_pitching_file='data/projections/pitching_forecasts.csv'
    )
    
    # Run model comparison for a player
    visualizer.run_model_comparison('player1', 'HR', is_pitcher=False)
