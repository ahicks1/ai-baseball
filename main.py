#!/usr/bin/env python3
"""
Fantasy Baseball Draft Tool

Main entry point for the fantasy baseball draft tool.
"""

import os
import sys
import click

from src.draft_tool.cli import cli


if __name__ == "__main__":
    # Print welcome message
    print("=" * 80)
    print("Fantasy Baseball Draft Tool")
    print("=" * 80)
    print("A tool for fantasy baseball draft analysis and player forecasting.")
    print()
    print("This tool helps you:")
    print("- Process historical baseball data from Retrosheet")
    print("- Analyze player performance across relevant fantasy categories")
    print("- Forecast player performance for the upcoming season")
    print("- Rank players based on projected value in H2H categories leagues")
    print("- Run a draft simulation to help you make optimal draft decisions")
    print()
    print("For help, run: python main.py --help")
    print("=" * 80)
    print()
    
    # Run the CLI
    cli()
