#!/usr/bin/env python3
"""
Unit tests for the Retrosheet data processing module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import shutil

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing.process_retrosheet import (
    load_players_file,
    load_batting_file,
    load_pitching_file,
    load_fielding_file,
    load_plays_file,
    aggregate_batting_stats,
    calculate_advanced_batting_stats,
    aggregate_pitching_stats,
    calculate_advanced_pitching_stats,
    process_season_data,
    process_multiple_seasons,
    process_season_data_with_dataframes
)


class TestRetroSheetProcessing(unittest.TestCase):
    """Test cases for Retrosheet data processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Sample data for testing
        self.sample_players_data = """id,last,first,bat,throw,team,g,g_p,g_sp,g_rp,g_c,g_1b,g_2b,g_3b,g_ss,g_lf,g_cf,g_rf,g_of,g_dh,g_ph,g_pr,first_g,last_g,season
player1,Doe,John,R,R,NYY,150,0,0,0,0,150,0,0,0,0,0,0,0,0,0,0,20230401,20231001,2023
player2,Smith,Jane,L,R,BOS,145,0,0,0,145,0,0,0,0,0,0,0,0,0,0,0,20230401,20231001,2023
player3,Johnson,Bob,R,R,LAD,160,160,160,0,0,0,0,0,0,0,0,0,0,0,0,0,20230401,20231001,2023"""
        
        self.sample_batting_data = """gid,id,team,b_lp,b_seq,stattype,b_pa,b_ab,b_r,b_h,b_d,b_t,b_hr,b_rbi,b_sh,b_sf,b_hbp,b_w,b_iw,b_k,b_sb,b_cs,b_gdp,b_xi,b_roe,dh,ph,pr,date,number,site,vishome,opp,win,loss,tie,gametype,box,pbp
game1,player1,NYY,1,1,value,5,4,1,2,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,20230401,1,NYY01,h,BOS,1,0,0,regular,y,y
game2,player1,NYY,1,1,value,4,4,0,1,0,0,1,2,0,0,0,0,0,2,0,0,0,0,0,0,0,0,20230402,1,NYY01,h,BOS,0,1,0,regular,y,y
game1,player2,BOS,2,1,value,4,3,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,20230401,1,NYY01,v,NYY,0,1,0,regular,y,y
game2,player2,BOS,2,1,value,4,4,1,2,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,20230402,1,NYY01,v,NYY,1,0,0,regular,y,y
game3,player1,NYY,1,1,value,4,3,2,1,0,1,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,20220401,1,NYY01,h,BOS,1,0,0,regular,y,y
game4,player2,BOS,2,1,value,4,4,0,2,0,0,1,2,0,0,0,0,0,1,0,0,0,0,0,0,0,0,20220402,1,NYY01,v,NYY,0,1,0,regular,y,y"""
        
        self.sample_pitching_data = """gid,id,team,p_seq,stattype,p_ipouts,p_noout,p_bfp,p_h,p_d,p_t,p_hr,p_r,p_er,p_w,p_iw,p_k,p_hbp,p_wp,p_bk,p_sh,p_sf,p_sb,p_cs,p_pb,wp,lp,save,p_gs,p_gf,p_cg,date,number,site,vishome,opp,win,loss,tie,gametype,box,pbp
game1,player3,LAD,1,value,27,0,35,5,1,0,1,2,2,2,0,10,0,0,0,0,0,0,0,0,1,0,0,1,1,1,20230401,1,LAD01,h,SFN,1,0,0,regular,y,y
game2,player3,LAD,1,value,21,0,30,7,2,0,0,3,3,1,0,8,1,0,0,0,0,0,0,0,0,1,0,1,0,0,20230402,1,LAD01,h,SFN,0,1,0,regular,y,y
game3,player3,LAD,1,value,24,0,32,6,1,1,0,1,1,3,0,9,0,0,0,0,0,0,0,0,1,0,0,1,1,0,20220401,1,LAD01,h,SFN,1,0,0,regular,y,y"""
        
        # Create test CSV files
        self.players_file = os.path.join(self.test_dir, 'allplayers.csv')
        self.batting_file = os.path.join(self.test_dir, 'batting.csv')
        self.pitching_file = os.path.join(self.test_dir, 'pitching.csv')
        
        with open(self.players_file, 'w') as f:
            f.write(self.sample_players_data)
        
        with open(self.batting_file, 'w') as f:
            f.write(self.sample_batting_data)
        
        with open(self.pitching_file, 'w') as f:
            f.write(self.sample_pitching_data)
        
        # Create output directory
        self.output_dir = os.path.join(self.test_dir, 'processed')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def test_load_players_file(self):
        """Test loading player information from CSV."""
        players_df = load_players_file(self.players_file)
        
        # Check that the DataFrame has the expected shape
        self.assertEqual(players_df.shape[0], 3)  # 3 rows
        
        # Check that the player IDs are correct
        self.assertIn('player1', players_df['id'].values)
        self.assertIn('player2', players_df['id'].values)
        self.assertIn('player3', players_df['id'].values)
    
    def test_load_batting_file(self):
        """Test loading batting statistics from CSV."""
        batting_df = load_batting_file(self.batting_file)
        
        # Check that the DataFrame has the expected shape
        self.assertEqual(batting_df.shape[0], 6)  # 6 rows
        
        # Check that the columns are correct
        self.assertIn('b_pa', batting_df.columns)
        self.assertIn('b_ab', batting_df.columns)
        self.assertIn('b_hr', batting_df.columns)
        
        # Check that the date column was converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_dtype(batting_df['date']))
    
    def test_load_pitching_file(self):
        """Test loading pitching statistics from CSV."""
        pitching_df = load_pitching_file(self.pitching_file)
        
        # Check that the DataFrame has the expected shape
        self.assertEqual(pitching_df.shape[0], 3)  # 3 rows
        
        # Check that the columns are correct
        self.assertIn('p_ipouts', pitching_df.columns)
        self.assertIn('p_k', pitching_df.columns)
        self.assertIn('p_er', pitching_df.columns)
        
        # Check that the date column was converted to datetime
        self.assertTrue(pd.api.types.is_datetime64_dtype(pitching_df['date']))
    
    def test_load_file_not_found(self):
        """Test error handling for missing files."""
        with self.assertRaises(FileNotFoundError):
            load_players_file(os.path.join(self.test_dir, 'nonexistent.csv'))
    
    def test_aggregate_batting_stats(self):
        """Test aggregation of batting statistics."""
        batting_df = load_batting_file(self.batting_file)
        players_df = load_players_file(self.players_file)
        
        # Test aggregation for all seasons
        batting_stats = aggregate_batting_stats(batting_df, players_df)
        
        # Check that the DataFrame has the expected shape
        self.assertEqual(batting_stats.shape[0], 2)  # 2 players
        
        # Test aggregation for a specific season (2023)
        batting_stats_2023 = aggregate_batting_stats(batting_df, players_df, 2023)
        
        # Check that the DataFrame has the expected shape
        self.assertEqual(batting_stats_2023.shape[0], 2)  # 2 players
        
        # Check player1's stats for 2023
        player1_stats = batting_stats_2023[batting_stats_2023['PLAYER_ID'] == 'player1'].iloc[0]
        self.assertEqual(player1_stats['PA'], 9)
        self.assertEqual(player1_stats['AB'], 8)
        self.assertEqual(player1_stats['HR'], 1)
        self.assertEqual(player1_stats['RBI'], 3)
        
        # Check player2's stats for 2023
        player2_stats = batting_stats_2023[batting_stats_2023['PLAYER_ID'] == 'player2'].iloc[0]
        self.assertEqual(player2_stats['PA'], 8)
        self.assertEqual(player2_stats['AB'], 7)
        self.assertEqual(player2_stats['SB'], 3)
    
    def test_calculate_advanced_batting_stats(self):
        """Test calculation of advanced batting metrics."""
        batting_df = load_batting_file(self.batting_file)
        players_df = load_players_file(self.players_file)
        
        batting_stats = aggregate_batting_stats(batting_df, players_df, 2023)
        advanced_stats = calculate_advanced_batting_stats(batting_stats)
        
        # Check that the advanced metrics are calculated
        self.assertIn('TB', advanced_stats.columns)
        self.assertIn('AVG', advanced_stats.columns)
        self.assertIn('OBP', advanced_stats.columns)
        self.assertIn('SLG', advanced_stats.columns)
        self.assertIn('OPS', advanced_stats.columns)
        
        # Check player1's advanced stats
        player1_stats = advanced_stats[advanced_stats['PLAYER_ID'] == 'player1'].iloc[0]
        self.assertEqual(player1_stats['TB'], 7)  # 3 hits (1 single, 1 double, 1 HR) = 7 TB
        self.assertAlmostEqual(player1_stats['AVG'], 0.375)  # 3/8
        
        # Check player2's advanced stats
        player2_stats = advanced_stats[advanced_stats['PLAYER_ID'] == 'player2'].iloc[0]
        self.assertEqual(player2_stats['TB'], 4)  # 3 hits (2 singles, 1 double) = 4 TB
        self.assertAlmostEqual(player2_stats['AVG'], 3/7, places=5)  # 3/7 = 0.42857...
    
    def test_aggregate_pitching_stats(self):
        """Test aggregation of pitching statistics."""
        pitching_df = load_pitching_file(self.pitching_file)
        players_df = load_players_file(self.players_file)
        
        # Test aggregation for all seasons
        pitching_stats = aggregate_pitching_stats(pitching_df, players_df)
        
        # Check that the DataFrame has the expected shape
        self.assertEqual(pitching_stats.shape[0], 1)  # 1 pitcher
        
        # Test aggregation for a specific season (2023)
        pitching_stats_2023 = aggregate_pitching_stats(pitching_df, players_df, 2023)
        
        # Check player3's stats for 2023
        player3_stats = pitching_stats_2023[pitching_stats_2023['PLAYER_ID'] == 'player3'].iloc[0]
        self.assertEqual(player3_stats['IPouts'], 48)  # 27 + 21
        self.assertEqual(player3_stats['K'], 18)  # 10 + 8
        self.assertEqual(player3_stats['ER'], 5)  # 2 + 3
        self.assertEqual(player3_stats['GS'], 2)  # 1 + 1
    
    def test_calculate_advanced_pitching_stats(self):
        """Test calculation of advanced pitching metrics."""
        pitching_df = load_pitching_file(self.pitching_file)
        players_df = load_players_file(self.players_file)
        
        pitching_stats = aggregate_pitching_stats(pitching_df, players_df, 2023)
        advanced_stats = calculate_advanced_pitching_stats(pitching_stats)
        
        # Check that the advanced metrics are calculated
        self.assertIn('IP', advanced_stats.columns)
        self.assertIn('ERA', advanced_stats.columns)
        self.assertIn('WHIP', advanced_stats.columns)
        self.assertIn('QS', advanced_stats.columns)
        
        # Check player3's advanced stats
        player3_stats = advanced_stats[advanced_stats['PLAYER_ID'] == 'player3'].iloc[0]
        self.assertEqual(player3_stats['IP'], 16.0)  # 48/3
        self.assertAlmostEqual(player3_stats['ERA'], 2.8125)  # (5 * 9) / 16
        self.assertAlmostEqual(player3_stats['WHIP'], 0.9375)  # (3 + 12) / 16
    
    def test_process_season_data_with_dataframes(self):
        """Test processing data for a single season using pre-loaded DataFrames."""
        players_df = load_players_file(self.players_file)
        batting_df = load_batting_file(self.batting_file)
        pitching_df = load_pitching_file(self.pitching_file)
        
        # Filter data for 2023
        batting_df_2023 = batting_df[batting_df['date'].dt.year == 2023]
        pitching_df_2023 = pitching_df[pitching_df['date'].dt.year == 2023]
        
        batting_stats, pitching_stats = process_season_data_with_dataframes(
            2023, players_df, batting_df_2023, pitching_df_2023, None, None, self.output_dir
        )
        
        # Check that the output files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'batting_stats_2023.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pitching_stats_2023.csv')))
        
        # Check that the DataFrames have the expected shapes
        self.assertEqual(batting_stats.shape[0], 2)  # 2 batters
        self.assertEqual(pitching_stats.shape[0], 1)  # 1 pitcher
    
    def test_process_season_data(self):
        """Test processing data for a single season."""
        batting_stats, pitching_stats = process_season_data(2023, self.test_dir, self.output_dir)
        
        # Check that the output files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'batting_stats_2023.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pitching_stats_2023.csv')))
        
        # Check that the DataFrames have the expected shapes
        self.assertEqual(batting_stats.shape[0], 2)  # 2 batters
        self.assertEqual(pitching_stats.shape[0], 1)  # 1 pitcher
    
    def test_process_multiple_seasons(self):
        """Test processing data for multiple seasons."""
        batting_stats, pitching_stats = process_multiple_seasons(2022, 2023, self.test_dir, self.output_dir)
        
        # Check that the output files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'batting_stats_2022.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'batting_stats_2023.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pitching_stats_2022.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pitching_stats_2023.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'batting_stats_all.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'pitching_stats_all.csv')))
        
        # Check that the combined DataFrames have the expected shapes
        self.assertEqual(batting_stats.shape[0], 4)  # 2 batters * 2 seasons
        self.assertEqual(pitching_stats.shape[0], 2)  # 1 pitcher * 2 seasons
    
    def test_year_filtering(self):
        """Test that data is properly filtered by year."""
        # Load the data
        batting_df = load_batting_file(self.batting_file)
        
        # Check that we have data for both 2022 and 2023
        years = batting_df['date'].dt.year.unique()
        self.assertIn(2022, years)
        self.assertIn(2023, years)
        
        # Process multiple seasons with filtering
        batting_stats, pitching_stats = process_multiple_seasons(2023, 2023, self.test_dir, self.output_dir)
        
        # Check that only 2023 data was processed
        self.assertEqual(batting_stats['SEASON'].unique()[0], 2023)
        self.assertEqual(pitching_stats['SEASON'].unique()[0], 2023)
        
        # Process just 2022
        batting_stats, pitching_stats = process_multiple_seasons(2022, 2022, self.test_dir, self.output_dir)
        
        # Check that only 2022 data was processed
        self.assertEqual(batting_stats['SEASON'].unique()[0], 2022)
        self.assertEqual(pitching_stats['SEASON'].unique()[0], 2022)


if __name__ == '__main__':
    unittest.main()
