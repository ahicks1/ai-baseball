#!/usr/bin/env python3
"""
Tests for the PlayerRanker class.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ranking.player_ranker import PlayerRanker


class TestPlayerRanker(unittest.TestCase):
    """
    Test cases for the PlayerRanker class.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create sample batting projections
        self.batting_projections = pd.DataFrame({
            'PLAYER_ID': ['batter1', 'batter2', 'batter3'],
            'HR': [30, 20, 40],
            'OBP': [0.380, 0.350, 0.400],
            'R': [90, 80, 100],
            'RBI': [95, 85, 110],
            'SB': [10, 20, 5],
            'TB': [280, 250, 300]
        })
        
        # Create sample pitching projections
        self.pitching_projections = pd.DataFrame({
            'PLAYER_ID': ['pitcher1', 'pitcher2', 'pitcher3'],
            'ERA': [3.50, 2.80, 4.20],
            'WHIP': [1.20, 1.05, 1.35],
            'K': [200, 220, 180],
            'SV+HLD': [5, 30, 10],
            'W+QS': [15, 5, 20]
        })
        
        # Initialize the ranker with sample data
        self.ranker = PlayerRanker(
            batting_projections=self.batting_projections,
            pitching_projections=self.pitching_projections
        )
    
    def test_calculate_z_scores(self):
        """
        Test the calculate_z_scores method.
        """
        # Calculate z-scores for batting projections
        batting_z = self.ranker.calculate_z_scores(self.batting_projections, ['HR', 'OBP', 'R', 'RBI', 'SB', 'TB'])
        
        # Check that z-score columns were created
        self.assertIn('HR_Z', batting_z.columns)
        self.assertIn('OBP_Z', batting_z.columns)
        self.assertIn('R_Z', batting_z.columns)
        self.assertIn('RBI_Z', batting_z.columns)
        self.assertIn('SB_Z', batting_z.columns)
        self.assertIn('TB_Z', batting_z.columns)
        
        # Check that z-scores are calculated correctly
        # For HR, the values are [30, 20, 40] with mean 30 and std 10
        # So the z-scores should be [0, -1, 1]
        np.testing.assert_almost_equal(batting_z['HR_Z'].values, [0, -1, 1], decimal=2)
    
    def test_calculate_weighted_z_scores(self):
        """
        Test the calculate_weighted_z_scores method.
        """
        # Calculate z-scores
        batting_z = self.ranker.calculate_z_scores(self.batting_projections, ['HR', 'OBP'])
        
        # Define weights
        weights = {'HR': 2.0, 'OBP': 1.0}
        
        # Calculate weighted z-scores
        batting_wz = self.ranker.calculate_weighted_z_scores(batting_z, ['HR', 'OBP'], weights)
        
        # Check that weighted z-score columns were created
        self.assertIn('HR_WZ', batting_wz.columns)
        self.assertIn('OBP_WZ', batting_wz.columns)
        
        # Check that weighted z-scores are calculated correctly
        # For HR, the z-scores are [0, -1, 1] and the weight is 2.0
        # So the weighted z-scores should be [0, -2, 2]
        np.testing.assert_almost_equal(batting_wz['HR_WZ'].values, [0, -2, 2], decimal=2)
    
    def test_calculate_total_value(self):
        """
        Test the calculate_total_value method.
        """
        # Calculate z-scores
        batting_z = self.ranker.calculate_z_scores(self.batting_projections, ['HR', 'OBP'])
        
        # Define weights
        weights = {'HR': 1.0, 'OBP': 1.0}
        
        # Calculate weighted z-scores
        batting_wz = self.ranker.calculate_weighted_z_scores(batting_z, ['HR', 'OBP'], weights)
        
        # Calculate total value
        batting_value = self.ranker.calculate_total_value(batting_wz, ['HR', 'OBP'])
        
        # Check that total value column was created
        self.assertIn('TOTAL_VALUE', batting_value.columns)
        
        # Check that total value is calculated correctly
        # For batter1, the weighted z-scores are [0, 0.5] so the total value should be 0.5
        # For batter2, the weighted z-scores are [-1, -0.5] so the total value should be -1.5
        # For batter3, the weighted z-scores are [1, 1] so the total value should be 2
        np.testing.assert_almost_equal(batting_value['TOTAL_VALUE'].values, [0.5, -1.5, 2], decimal=2)
    
    def test_rank_batters(self):
        """
        Test the rank_batters method.
        """
        # Rank batters
        batting_rankings = self.ranker.rank_batters()
        
        # Check that rank column was created
        self.assertIn('RANK', batting_rankings.columns)
        
        # Check that player type column was created
        self.assertIn('PLAYER_TYPE', batting_rankings.columns)
        
        # Check that all player types are 'BATTER'
        self.assertTrue((batting_rankings['PLAYER_TYPE'] == 'BATTER').all())
        
        # Check that the rankings are in the correct order
        # batter3 should be ranked 1, batter1 should be ranked 2, batter2 should be ranked 3
        expected_order = ['batter3', 'batter1', 'batter2']
        actual_order = batting_rankings.sort_values('RANK')['PLAYER_ID'].values
        np.testing.assert_array_equal(actual_order, expected_order)
    
    def test_rank_pitchers(self):
        """
        Test the rank_pitchers method.
        """
        # Rank pitchers
        pitching_rankings = self.ranker.rank_pitchers()
        
        # Check that rank column was created
        self.assertIn('RANK', pitching_rankings.columns)
        
        # Check that player type column was created
        self.assertIn('PLAYER_TYPE', pitching_rankings.columns)
        
        # Check that all player types are 'PITCHER'
        self.assertTrue((pitching_rankings['PLAYER_TYPE'] == 'PITCHER').all())
    
    def test_create_overall_rankings(self):
        """
        Test the create_overall_rankings method.
        """
        # Create overall rankings
        overall_rankings = self.ranker.create_overall_rankings()
        
        # Check that rank column was created
        self.assertIn('RANK', overall_rankings.columns)
        
        # Check that player type column exists
        self.assertIn('PLAYER_TYPE', overall_rankings.columns)
        
        # Check that there are both batters and pitchers in the rankings
        self.assertTrue((overall_rankings['PLAYER_TYPE'] == 'BATTER').any())
        self.assertTrue((overall_rankings['PLAYER_TYPE'] == 'PITCHER').any())
        
        # Check that the total number of players is correct
        self.assertEqual(len(overall_rankings), 6)  # 3 batters + 3 pitchers


if __name__ == '__main__':
    unittest.main()
