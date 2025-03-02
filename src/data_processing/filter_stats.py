#!/usr/bin/env python3
"""
Data Filtering Script for Baseball Statistics

This script filters CSV data based on a specified column and threshold value.
It can be used to remove rows with values below a certain threshold, such as
filtering out batters with fewer than 10 plate appearances.

Usage:
    python filter_stats.py input_file output_file column_name threshold [--less-than]

Example:
    python filter_stats.py data/processed/batting_stats_all_sea.csv data/processed/batting_stats_filtered.csv PA 10
"""

import argparse
import pandas as pd
import os
import sys


def filter_data(input_file, output_file, column_name, threshold, greater_than=True):
    """
    Filter CSV data based on a column value threshold.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        column_name (str): Name of column to filter on
        threshold (float): Threshold value for filtering
        greater_than (bool): If True, keep rows where column value >= threshold
                            If False, keep rows where column value <= threshold
    
    Returns:
        tuple: (total_rows, filtered_rows, remaining_rows)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the data
    print(f"Loading data from {input_file}")
    try:
        data = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    total_rows = len(data)
    print(f"Loaded {total_rows} rows")
    
    # Check if column exists
    if column_name not in data.columns:
        print(f"Error: Column '{column_name}' not found in the data")
        print(f"Available columns: {', '.join(data.columns)}")
        sys.exit(1)
    
    # Apply filter
    if greater_than:
        filtered_data = data[data[column_name] >= threshold]
        filter_description = f"greater than or equal to {threshold}"
    else:
        filtered_data = data[data[column_name] <= threshold]
        filter_description = f"less than or equal to {threshold}"
    
    remaining_rows = len(filtered_data)
    filtered_rows = total_rows - remaining_rows
    
    print(f"Filtering rows where {column_name} is {filter_description}")
    print(f"Removed {filtered_rows} rows ({filtered_rows/total_rows:.1%} of data)")
    print(f"Keeping {remaining_rows} rows ({remaining_rows/total_rows:.1%} of data)")
    
    # Save filtered data
    filtered_data.to_csv(output_file, index=False)
    print(f"Saved filtered data to {output_file}")
    
    return total_rows, filtered_rows, remaining_rows


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Filter CSV data based on column threshold')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output CSV file')
    parser.add_argument('column_name', help='Name of column to filter on')
    parser.add_argument('threshold', type=float, help='Threshold value for filtering')
    parser.add_argument('--less-than', action='store_true', 
                        help='Keep rows where column value <= threshold (default is >=)')
    
    args = parser.parse_args()
    
    # Run the filter
    filter_data(
        args.input_file, 
        args.output_file, 
        args.column_name, 
        args.threshold, 
        greater_than=not args.less_than
    )


if __name__ == "__main__":
    main()
