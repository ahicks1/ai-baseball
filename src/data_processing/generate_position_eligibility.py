#!/usr/bin/env python3
"""
Position Eligibility Generator

This script generates a CSV file mapping player IDs to their eligible positions
for a specified season, based on their playing time in the previous season.
"""

import os
import pandas as pd
import argparse
from collections import defaultdict

# Position code mapping
POSITION_MAP = {
    1: "P",
    2: "C",
    3: "1B",
    4: "2B",
    5: "3B",
    6: "SS",
    7: "OF",  # Left field
    8: "OF",  # Center field
    9: "OF"   # Right field
}

def generate_position_eligibility(fielding_file, season, output_file, games_threshold=20):
    """
    Generate position eligibility data for a specified season.
    
    Args:
        fielding_file (str): Path to the fielding.csv file
        season (int): Season to generate eligibility for
        output_file (str): Path to output CSV file
        games_threshold (int): Number of games required for eligibility
    """
    # Previous season is used for eligibility
    prev_season = season - 1
    
    # Read fielding data
    print(f"Reading fielding data from {fielding_file}...")
    fielding_df = pd.read_csv(fielding_file, low_memory=False)
    
    # Extract season from date (first 4 characters)
    fielding_df['season'] = fielding_df['date'].astype(str).str[:4].astype(int)
    
    # Filter to previous season and only regular games
    prev_season_df = fielding_df[(fielding_df['season'] == prev_season) & 
                                (fielding_df['gametype'] == 'regular')]
    
    print(f"Found {len(prev_season_df)} fielding records for {prev_season} regular season")
    
    # Count games by player and position
    # Group by player ID, game ID, and position to avoid counting multiple positions in same game
    player_games = defaultdict(lambda: defaultdict(set))
    
    for _, row in prev_season_df.iterrows():
        player_id = row['id']
        position = int(row['d_pos'])
        game_id = row['gid']
        
        # Add this game to the player's position
        player_games[player_id][position].add(game_id)
    
    # Determine eligibility
    eligibility_data = []
    
    for player_id, positions in player_games.items():
        eligible_positions = set()
        
        for position, games in positions.items():
            if len(games) >= games_threshold:
                position_name = POSITION_MAP.get(position)
                if position_name:
                    eligible_positions.add(position_name)
        
        # Format positions as dash-separated string
        if eligible_positions:
            positions_str = "-".join(sorted(eligible_positions))
        else:
            positions_str = "DH"  # Default to DH if no eligible positions
        
        eligibility_data.append({
            "PLAYER_ID": player_id,
            "SEASON": season,
            "POSITIONS": positions_str
        })
    
    # Create DataFrame and save to CSV
    eligibility_df = pd.DataFrame(eligibility_data)
    print(f"Generated eligibility data for {len(eligibility_df)} players")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    eligibility_df.to_csv(output_file, index=False)
    print(f"Saved position eligibility data to {output_file}")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description="Generate position eligibility data")
    parser.add_argument("--season", type=int, required=True, help="Season to generate eligibility for")
    parser.add_argument("--fielding-file", type=str, default="data/raw/fielding.csv", help="Path to fielding.csv file")
    parser.add_argument("--output-file", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--games-threshold", type=int, default=20, help="Number of games required for eligibility")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = f"data/processed/position_eligibility_{args.season}.csv"
    
    generate_position_eligibility(
        args.fielding_file,
        args.season,
        args.output_file,
        args.games_threshold
    )

if __name__ == "__main__":
    main()
