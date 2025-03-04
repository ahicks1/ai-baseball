/**
 * Calculate the average of an array of numbers
 * @param {Array<number>} values - Array of numeric values
 * @returns {number} - The average value
 */
export const calculateAverage = (values) => {
  if (!values || !values.length) return 0;
  const sum = values.reduce((acc, val) => acc + (val || 0), 0);
  return sum / values.length;
};

/**
 * Calculate the standard deviation of an array of numbers
 * @param {Array<number>} values - Array of numeric values
 * @returns {number} - The standard deviation
 */
export const calculateStandardDeviation = (values) => {
  if (!values || values.length < 2) return 0;
  
  const avg = calculateAverage(values);
  const squareDiffs = values.map(value => {
    const diff = value - avg;
    return diff * diff;
  });
  
  const avgSquareDiff = calculateAverage(squareDiffs);
  return Math.sqrt(avgSquareDiff);
};

/**
 * Calculate Z-scores for a set of values
 * @param {Array<number>} values - Array of numeric values
 * @returns {Array<number>} - Array of Z-scores
 */
export const calculateZScores = (values) => {
  if (!values || !values.length) return [];
  
  const avg = calculateAverage(values);
  const stdDev = calculateStandardDeviation(values);
  
  if (stdDev === 0) return values.map(() => 0);
  
  return values.map(value => (value - avg) / stdDev);
};

/**
 * Get the top N players based on a specific stat
 * @param {Array<Object>} players - Array of player objects
 * @param {string} stat - The stat to sort by
 * @param {number} n - Number of players to return
 * @returns {Array<Object>} - Top N players
 */
export const getTopPlayersByStat = (players, stat, n = 10) => {
  if (!players || !players.length) return [];
  
  // Make a copy to avoid mutating the original array
  const sortedPlayers = [...players];
  
  // Sort by the specified stat in descending order
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

/**
 * Filter players by position
 * @param {Array<Object>} players - Array of player objects
 * @param {string} position - Position to filter by
 * @returns {Array<Object>} - Filtered players
 */
export const filterPlayersByPosition = (players, position) => {
  if (!players || !players.length) return [];
  if (!position || position === 'ALL') return players;
  
  return players.filter(player => {
    // Check if the player has position data
    if (!player.POSITIONS) return false;
    
    // Split the positions string and check if it includes the specified position
    const positions = player.POSITIONS.split(',').map(pos => pos.trim());
    return positions.includes(position);
  });
};

/**
 * Filter players by type (batter or pitcher)
 * @param {Array<Object>} players - Array of player objects
 * @param {string} type - Player type ('BATTER' or 'PITCHER')
 * @returns {Array<Object>} - Filtered players
 */
export const filterPlayersByType = (players, type) => {
  if (!players || !players.length) return [];
  if (!type || type === 'ALL') return players;
  
  return players.filter(player => player.PLAYER_TYPE === type);
};

/**
 * Get stats for graphing
 * @param {Array<Object>} players - Array of player objects
 * @param {string} stat - The stat to extract
 * @returns {Object} - Data for graphing
 */
export const getStatsForGraphing = (players, stat) => {
  if (!players || !players.length) return { labels: [], data: [] };
  
  const labels = players.map(player => player.PLAYER_NAME);
  const data = players.map(player => player[stat] || 0);
  
  return { labels, data };
};

/**
 * Calculate roster composition by position
 * @param {Array<Object>} players - Array of player objects
 * @returns {Object} - Counts by position
 */
export const calculateRosterComposition = (players) => {
  if (!players || !players.length) return {};
  
  const positionCounts = {};
  
  players.forEach(player => {
    if (!player.POSITIONS) return;
    
    const positions = player.POSITIONS.split(',').map(pos => pos.trim());
    positions.forEach(position => {
      positionCounts[position] = (positionCounts[position] || 0) + 1;
    });
  });
  
  return positionCounts;
};
