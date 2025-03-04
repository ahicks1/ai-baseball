import { fetchAndParseCSV } from '../utils/csvParser';

// Define the paths to the CSV files
const CSV_PATHS = {
  batting: '/data/rankings_with_scarcity/batting_rankings.csv',
  pitching: '/data/rankings_with_scarcity/pitching_rankings.csv',
  overall: '/data/rankings_with_scarcity/overall_rankings.csv'
};

// Cache for storing parsed data
const dataCache = {
  batting: null,
  pitching: null,
  overall: null
};

/**
 * Load player data from CSV files
 * @param {string} dataType - Type of data to load ('batting', 'pitching', or 'overall')
 * @param {boolean} forceRefresh - Whether to force a refresh of the cache
 * @returns {Promise<Array>} - A promise that resolves to an array of player data
 */
export const loadPlayerData = async (dataType = 'overall', forceRefresh = false) => {
  // Validate data type
  if (!CSV_PATHS[dataType]) {
    throw new Error(`Invalid data type: ${dataType}`);
  }
  
  // Return cached data if available and not forcing refresh
  if (dataCache[dataType] && !forceRefresh) {
    return dataCache[dataType];
  }
  
  try {
    // Fetch and parse the CSV file

    const data = await fetchAndParseCSV(CSV_PATHS[dataType]);
    
    // Cache the data
    dataCache[dataType] = data;
    
    return data;
  } catch (error) {
    console.error(`Error loading ${dataType} data:`, error);
    throw error;
  }
};

/**
 * Get a player by ID
 * @param {string} playerId - The player ID to look for
 * @param {string} dataType - Type of data to search in ('batting', 'pitching', or 'overall')
 * @returns {Promise<Object|null>} - A promise that resolves to the player object or null
 */
export const getPlayerById = async (playerId, dataType = 'overall') => {
  try {
    const data = await loadPlayerData(dataType);
    return data.find(player => player.PLAYER_ID === playerId) || null;
  } catch (error) {
    console.error(`Error getting player ${playerId}:`, error);
    throw error;
  }
};

/**
 * Get players by position
 * @param {string} position - The position to filter by
 * @param {string} dataType - Type of data to search in ('batting', 'pitching', or 'overall')
 * @returns {Promise<Array>} - A promise that resolves to an array of player objects
 */
export const getPlayersByPosition = async (position, dataType = 'overall') => {
  try {
    const data = await loadPlayerData(dataType);
    
    if (!position || position === 'ALL') {
      return data;
    }
    
    return data.filter(player => {
      if (!player.POSITIONS) return false;
      const positions = player.POSITIONS.split(',').map(pos => pos.trim());
      return positions.includes(position);
    });
  } catch (error) {
    console.error(`Error getting players for position ${position}:`, error);
    throw error;
  }
};

/**
 * Get the available positions from the data
 * @param {string} dataType - Type of data to search in ('batting', 'pitching', or 'overall')
 * @returns {Promise<Array>} - A promise that resolves to an array of position strings
 */
export const getAvailablePositions = async (dataType = 'overall') => {
  try {
    const data = await loadPlayerData(dataType);
    
    // Create a Set to store unique positions
    const positionsSet = new Set();
    
    // Add positions from each player
    data.forEach(player => {
      if (!player.POSITIONS) return;
      
      const positions = player.POSITIONS.split('-').map(pos => pos.trim());
      positions.forEach(pos => positionsSet.add(pos));
    });
    
    // Convert Set to Array and sort
    return Array.from(positionsSet).sort();
  } catch (error) {
    console.error('Error getting available positions:', error);
    throw error;
  }
};

/**
 * Get the available stats for a data type
 * @param {string} dataType - Type of data ('batting', 'pitching', or 'overall')
 * @returns {Promise<Array>} - A promise that resolves to an array of stat names
 */
export const getAvailableStats = async (dataType = 'overall') => {
  try {
    const data = await loadPlayerData(dataType);
    
    if (!data || !data.length) {
      return [];
    }
    
    // Get all keys from the first player object
    const allKeys = Object.keys(data[0]);
    
    // Filter out non-stat keys
    const nonStatKeys = [
      'PLAYER_ID', 'SEASON', 'PLAYER_NAME', 'POSITIONS', 
      'POS_ADJ', 'ADJ_VALUE', 'RANK', 'PLAYER_TYPE'
    ];
    
    // Also filter out keys that end with _arima, _exp_smooth, etc.
    const statKeys = allKeys.filter(key => {
      if (nonStatKeys.includes(key)) return false;
      if (key.endsWith('_arima') || 
          key.endsWith('_exp_smooth') || 
          key.endsWith('_ridge') || 
          key.endsWith('_random_forest') || 
          key.endsWith('_gradient_boosting') ||
          key.endsWith('_Z') ||
          key.endsWith('_WZ')) {
        return false;
      }
      return true;
    });
    
    return statKeys;
  } catch (error) {
    console.error(`Error getting available stats for ${dataType}:`, error);
    throw error;
  }
};

/**
 * Get batting-specific stats
 * @returns {Promise<Array>} - A promise that resolves to an array of batting stat names
 */
export const getBattingStats = async () => {
  return getAvailableStats('batting');
};

/**
 * Get pitching-specific stats
 * @returns {Promise<Array>} - A promise that resolves to an array of pitching stat names
 */
export const getPitchingStats = async () => {
  return getAvailableStats('pitching');
};
