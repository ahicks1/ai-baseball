import { useState, useEffect } from 'react';
import * as dataService from '../services/dataService';

/**
 * Custom hook for loading and managing player data
 * @param {string} dataType - Type of data to load ('batting', 'pitching', or 'overall')
 * @param {Object} options - Additional options
 * @returns {Object} - Player data and related functions
 */
const usePlayerData = (dataType = 'overall', options = {}) => {
  // State for player data
  const [players, setPlayers] = useState([]);
  
  // State for loading status
  const [loading, setLoading] = useState(true);
  
  // State for error
  const [error, setError] = useState(null);
  
  // State for available positions
  const [positions, setPositions] = useState([]);
  
  // State for available stats
  const [availableStats, setAvailableStats] = useState([]);
  
  // State for selected position filter
  const [positionFilter, setPositionFilter] = useState('ALL');
  
  // State for selected player type filter
  const [playerTypeFilter, setPlayerTypeFilter] = useState('ALL');
  
  // State for search query
  const [searchQuery, setSearchQuery] = useState('');
  
  // State for filtered players
  const [filteredPlayers, setFilteredPlayers] = useState([]);

  // Load player data
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Load player data
        const data = await dataService.loadPlayerData(dataType);
        setPlayers(data);
        
        // Load available positions
        const positionsData = await dataService.getAvailablePositions(dataType);
        setPositions(['ALL', ...positionsData]);
        
        // Load available stats
        const statsData = await dataService.getAvailableStats(dataType);
        setAvailableStats(statsData);
        
        setLoading(false);
      } catch (err) {
        console.error('Error loading player data:', err);
        setError('Failed to load player data. Please try again.');
        setLoading(false);
      }
    };
    
    loadData();
  }, [dataType]);

  // Filter players by position
  const filterByPosition = (position) => {
    setPositionFilter(position);
  };

  // Filter players by type (batter or pitcher)
  const filterByPlayerType = (type) => {
    setPlayerTypeFilter(type);
  };

  // Search players by name
  const searchPlayers = (query) => {
    setSearchQuery(query);
  };

  // Update filtered players whenever filters or players change
  useEffect(() => {
    let filtered = [...players];
    
    // Filter by position
    if (positionFilter && positionFilter !== 'ALL') {
      filtered = filtered.filter(player => {
        if (!player.POSITIONS) return false;
        const positions = player.POSITIONS.split('-').map(pos => pos.trim());
        return positions.includes(positionFilter);
      });
    }
    
    // Filter by player type
    if (playerTypeFilter && playerTypeFilter !== 'ALL') {
      filtered = filtered.filter(player => player.PLAYER_TYPE === playerTypeFilter);
    }
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(player => 
        player.PLAYER_NAME && player.PLAYER_NAME.toLowerCase().includes(query)
      );
    }
    console.error("Filtered down to players", filtered.length)
    setFilteredPlayers(filtered);
  }, [players, positionFilter, playerTypeFilter, searchQuery]);

  // Sort players by a specific stat
  const sortPlayersByStat = (stat, direction = 'desc') => {
    const sortedPlayers = [...players];
    
    sortedPlayers.sort((a, b) => {
      // Handle ERA and WHIP which are better when lower
      if (stat === 'ERA' || stat === 'WHIP') {
        return direction === 'asc' 
          ? (b[stat] || 0) - (a[stat] || 0)
          : (a[stat] || 0) - (b[stat] || 0);
      }
      
      // For all other stats, higher is better
      return direction === 'asc'
        ? (a[stat] || 0) - (b[stat] || 0)
        : (b[stat] || 0) - (a[stat] || 0);
    });
    
    setPlayers(sortedPlayers);
  };

  // Get top N players by a specific stat
  const getTopPlayersByStat = (stat, n = 10) => {
    // Make a copy to avoid mutating the original array
    const sortedPlayers = [...filteredPlayers];
    
    // Sort by the specified stat
    sortedPlayers.sort((a, b) => {
      // Handle ERA and WHIP which are better when lower
      if (stat === 'ERA' || stat === 'WHIP') {
        return (a[stat] || 0) - (b[stat] || 0);
      }
      // For all other stats, higher is better
      return (b[stat] || 0) - (a[stat] || 0);
    });
    
    // Return the top N players
    return sortedPlayers.slice(0, n);
  };

  // Refresh player data
  const refreshData = async () => {
    try {
      setLoading(true);
      const data = await dataService.loadPlayerData(dataType, true);
      setPlayers(data);
      setLoading(false);
    } catch (err) {
      console.error('Error refreshing player data:', err);
      setError('Failed to refresh player data. Please try again.');
      setLoading(false);
    }
  };

  return {
    players,
    loading,
    error,
    positions,
    availableStats,
    positionFilter,
    playerTypeFilter,
    searchQuery,
    filteredPlayers,
    filterByPosition,
    filterByPlayerType,
    searchPlayers,
    sortPlayersByStat,
    getTopPlayersByStat,
    refreshData
  };
};

export default usePlayerData;
