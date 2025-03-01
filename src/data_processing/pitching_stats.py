#!/usr/bin/env python3
"""
Pitching Statistics Module

This module handles the calculation of advanced pitching statistics from play-by-play data,
specifically Quality Starts (QS) and Holds (HLD).

Quality Start (QS): A starting pitcher must pitch at least 6 innings and allow 3 or fewer earned runs
Hold (HLD): A relief pitcher enters in a save situation (lead of 3 runs or fewer), records at least
            one out, leaves without giving up the lead, and doesn't finish the game
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional


def process_pitcher_appearances(plays_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Process play-by-play data to track pitcher appearances, innings pitched, runs allowed, and game scores.
    
    Args:
        plays_df (pd.DataFrame): DataFrame containing play-by-play data
        
    Returns:
        Tuple containing:
        - Dict mapping (game_id, pitcher_id) to innings pitched (outs/3)
        - Dict mapping (game_id, pitcher_id) to earned runs allowed
        - Dict mapping game_id to a list of pitchers in order of appearance for each team
        - Dict mapping (game_id, pitcher_id) to a tuple of (entry_score, exit_score),
          where each score is a tuple of (team_score, opponent_score)
    """
    print("Processing pitcher appearances from play-by-play data")
    
    # Initialize dictionaries to store results
    innings_pitched = {}  # (game_id, pitcher_id) -> innings pitched (outs/3)
    earned_runs = {}      # (game_id, pitcher_id) -> earned runs allowed
    pitcher_order = {}    # game_id -> {0: [vis_pitchers], 1: [home_pitchers]}
    entry_exit_scores = {}  # (game_id, pitcher_id) -> (entry_score, exit_score)
    
    # Group by game
    for game_id, game_plays in plays_df.groupby('gid'):
        # Initialize pitcher order for this game
        pitcher_order[game_id] = {0: [], 1: []}  # 0 for visiting team, 1 for home team
        
        # Initialize game scores
        game_score = {0: 0, 1: 0}  # 0 for visiting team, 1 for home team
        
        # Track current pitcher for each team
        current_pitcher = {0: None, 1: None}  # 0 for visiting team, 1 for home team
        
        # Process each play
        for _, play in game_plays.iterrows():
            team = int(play['vis_home'])  # 0 for visiting team, 1 for home team
            if team not in [0, 1]:
                continue
            
            pitcher_id = play['pitcher']
            pitcher_team = 1 - team  # Pitcher is on the opposite team
            
            # Check if this is a new pitcher
            if pitcher_id != current_pitcher[pitcher_team]:
                # If there was a previous pitcher, record their exit score
                if current_pitcher[pitcher_team] is not None:
                    prev_key = (game_id, current_pitcher[pitcher_team])
                    if prev_key in entry_exit_scores:
                        entry_score, _ = entry_exit_scores[prev_key]
                        # Record exit score (team_score, opponent_score)
                        exit_score = (game_score[pitcher_team], game_score[team])
                        entry_exit_scores[prev_key] = (entry_score, exit_score)
                
                # Record entry score for the new pitcher
                current_pitcher[pitcher_team] = pitcher_id
                key = (game_id, pitcher_id)
                # Record entry score (team_score, opponent_score)
                entry_score = (game_score[pitcher_team], game_score[team])
                # Initialize with same exit score, will be updated later if pitcher changes
                entry_exit_scores[key] = (entry_score, entry_score)
            
            # Track pitcher order (first appearance only)
            if pitcher_id not in pitcher_order[game_id][pitcher_team]:
                pitcher_order[game_id][pitcher_team].append(pitcher_id)
            
            # Calculate outs recorded by this pitcher on this play
            outs_pre = int(play['outs_pre']) if pd.notna(play['outs_pre']) else 0
            outs_post = int(play['outs_post']) if pd.notna(play['outs_post']) else 0
            outs_recorded = outs_post - outs_pre
            
            # Update innings pitched
            key = (game_id, pitcher_id)
            if key not in innings_pitched:
                innings_pitched[key] = 0
            innings_pitched[key] += outs_recorded
            
            # Update game score based on runs scored on this play
            runs = int(play['runs']) if pd.notna(play['runs']) else 0
            if runs > 0:
                game_score[team] += runs
            
            # Track earned runs allowed
            er = int(play['er']) if pd.notna(play['er']) else 0
            if er > 0:
                # Determine which pitcher(s) to charge with earned runs
                # The play data indicates which pitcher is responsible for each runner
                if key not in earned_runs:
                    earned_runs[key] = 0
                earned_runs[key] += er
                
                # Also check if there are specific pitcher-runner assignments
                # Only process if the necessary columns exist in the DataFrame
                if all(col in play.index for col in ['run_b', 'run1', 'run2', 'run3', 'prun1', 'prun2', 'prun3']):
                    for base, run_col, prun_col in [
                        ('b', 'run_b', None),  # Batter (charged to current pitcher)
                        ('1', 'run1', 'prun1'),  # Runner on 1st
                        ('2', 'run2', 'prun2'),  # Runner on 2nd
                        ('3', 'run3', 'prun3')   # Runner on 3rd
                    ]:
                        # Skip batter or if no run scored from this base
                        if base == 'b' or pd.isna(play[run_col]):
                            continue
                        
                        # If a specific pitcher is charged with this run
                        if prun_col is not None and pd.notna(play[prun_col]):
                            responsible_pitcher = play[prun_col]
                            resp_key = (game_id, responsible_pitcher)
                            if resp_key not in earned_runs:
                                earned_runs[resp_key] = 0
                            earned_runs[resp_key] += 1
                            # Subtract from current pitcher's count since we've assigned it specifically
                            earned_runs[key] -= 1
        
        # Update exit scores for the last pitcher in each team for this game
        for team in [0, 1]:
            if current_pitcher[team] is not None:
                key = (game_id, current_pitcher[team])
                if key in entry_exit_scores:
                    entry_score, _ = entry_exit_scores[key]
                    # Final exit score for this pitcher
                    exit_score = (game_score[team], game_score[1-team])
                    entry_exit_scores[key] = (entry_score, exit_score)
    
    return innings_pitched, earned_runs, pitcher_order, entry_exit_scores


def calculate_quality_starts(plays_df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate Quality Starts (QS) from play-by-play data.
    A Quality Start is when a starting pitcher goes 6+ innings and allows 3 or fewer earned runs.
    
    Args:
        plays_df (pd.DataFrame): DataFrame containing play-by-play data
        
    Returns:
        Dict mapping pitcher_id to number of quality starts
    """
    print("Calculating Quality Starts from play-by-play data")
    
    # Process pitcher appearances
    innings_pitched, earned_runs, pitcher_order, _ = process_pitcher_appearances(plays_df)
    
    # Calculate quality starts
    quality_starts = {}
    
    # Iterate through games
    for game_id, teams in pitcher_order.items():
        # Check starters for both teams
        for team in [0, 1]:  # 0 for visiting team, 1 for home team
            if not teams[team]:  # Skip if no pitchers recorded for this team
                continue
                
            # Get the starting pitcher (first in the list)
            starter_id = teams[team][0]
            
            # Check if the starter pitched at least 6 innings (18 outs)
            ip_key = (game_id, starter_id)
            if ip_key in innings_pitched and innings_pitched[ip_key] >= 18:
                # Check if the starter allowed 3 or fewer earned runs
                er_allowed = earned_runs.get(ip_key, 0)
                if er_allowed <= 3:
                    # This is a quality start
                    if starter_id not in quality_starts:
                        quality_starts[starter_id] = 0
                    quality_starts[starter_id] += 1
    
    return quality_starts


def calculate_holds(plays_df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate Holds (HLD) from play-by-play data.
    A Hold is when a relief pitcher enters in a save situation (lead of 3 runs or fewer),
    records at least one out, leaves without giving up the lead, and doesn't finish the game.
    
    Args:
        plays_df (pd.DataFrame): DataFrame containing play-by-play data
        
    Returns:
        Dict mapping pitcher_id to number of holds
    """
    print("Calculating Holds from play-by-play data")
    
    # Process pitcher appearances
    innings_pitched, _, pitcher_order, entry_exit_scores = process_pitcher_appearances(plays_df)
    
    # Calculate holds
    holds = {}
    
    # Track if a pitcher recorded an out
    recorded_out = set()  # Set of (game_id, pitcher_id) tuples
    
    # Track if a pitcher finished the game
    finished_game = set()  # Set of (game_id, pitcher_id) tuples
    
    # First, identify pitchers who recorded outs
    for key, outs in innings_pitched.items():
        if outs > 0:
            recorded_out.add(key)
    
    # Identify pitchers who finished the game (last pitcher for each team in each game)
    for game_id, teams in pitcher_order.items():
        for team in [0, 1]:  # 0 for visiting team, 1 for home team
            if teams[team]:  # Skip if no pitchers recorded for this team
                last_pitcher = teams[team][-1]
                finished_game.add((game_id, last_pitcher))
    
    # Now determine holds
    for game_id, teams in pitcher_order.items():
        for team in [0, 1]:  # 0 for visiting team, 1 for home team
            pitchers = teams[team]
            
            # Skip if less than 3 pitchers (need starter, middle reliever, closer)
            if len(pitchers) < 3:
                continue
                
            # Skip the starter and the last pitcher (closer)
            for i in range(1, len(pitchers) - 1):
                pitcher_id = pitchers[i]
                key = (game_id, pitcher_id)
                
                # Skip if we don't have entry/exit scores for this pitcher
                if key not in entry_exit_scores:
                    continue
                
                # Check if the pitcher recorded an out
                if key not in recorded_out:
                    continue
                
                # Check if the pitcher entered in a save situation (lead of 3 runs or fewer)
                entry_score, exit_score = entry_exit_scores[key]
                team_score, opponent_score = entry_score
                lead = team_score - opponent_score
                
                if lead > 0 and lead <= 3:
                    # Check if the pitcher maintained the lead
                    team_score_exit, opponent_score_exit = exit_score
                    exit_lead = team_score_exit - opponent_score_exit
                    
                    if exit_lead > 0:
                        # Check if the pitcher didn't finish the game
                        if key not in finished_game:
                            # This is a hold
                            if pitcher_id not in holds:
                                holds[pitcher_id] = 0
                            holds[pitcher_id] += 1
    
    return holds


def integrate_advanced_stats(pitching_stats: pd.DataFrame, plays_df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate Quality Starts and Holds into the pitching statistics DataFrame.
    
    Args:
        pitching_stats (pd.DataFrame): DataFrame with basic pitching statistics
        plays_df (pd.DataFrame): DataFrame with play-by-play data
        
    Returns:
        pd.DataFrame: DataFrame with additional QS and HLD statistics
    """
    print("Integrating Quality Starts and Holds into pitching statistics")
    
    # Make a copy to avoid modifying the original DataFrame
    stats = pitching_stats.copy()
    
    # Calculate Quality Starts and Holds
    quality_starts = calculate_quality_starts(plays_df)
    holds = calculate_holds(plays_df)
    
    # Add QS and HLD columns to the DataFrame
    stats['QS'] = stats['PLAYER_ID'].map(quality_starts).fillna(0).astype(int)
    stats['HLD'] = stats['PLAYER_ID'].map(holds).fillna(0).astype(int)
    
    # Calculate combined fantasy stats
    if 'SV' in stats.columns:
        stats['SV+HLD'] = stats['SV'] + stats['HLD']
    else:
        stats['SV+HLD'] = stats['HLD']
        
    if 'W' in stats.columns:
        stats['W+QS'] = stats['W'] + stats['QS']
    else:
        stats['W+QS'] = stats['QS']
    
    return stats


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load play-by-play data
    plays_file = os.path.join('data/raw', 'playsNewer100k.csv')
    if os.path.exists(plays_file):
        print(f"Loading plays file: {plays_file}")
        plays_df = pd.read_csv(plays_file)
        
        # Calculate Quality Starts
        qs = calculate_quality_starts(plays_df)
        print(f"Found {sum(qs.values())} Quality Starts for {len(qs)} pitchers")
        
        # Calculate Holds
        hld = calculate_holds(plays_df)
        print(f"Found {sum(hld.values())} Holds for {len(hld)} pitchers")
    else:
        print(f"Plays file not found: {plays_file}")
