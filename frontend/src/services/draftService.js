// Draft status constants
export const DRAFT_STATUS = {
  AVAILABLE: 'available',
  DRAFTED_BY_YOU: 'drafted_by_you',
  DRAFTED_BY_OTHERS: 'drafted_by_others'
};

// Initial draft state
const initialDraftState = {
  // Players drafted by the user
  myRoster: [],
  
  // Players drafted by other teams
  othersDrafted: [],
  
  // All players with their draft status
  playerStatus: {},
  
  // Draft history
  draftHistory: [],
  
  // Current draft round
  currentRound: 1,
  
  // Draft configuration
  config: {
    totalRounds: 23,
    totalTeams: 12,
    myPosition: 1
  }
};

// Current draft state
let draftState = { ...initialDraftState };

/**
 * Initialize the draft state
 * @param {Object} config - Draft configuration
 * @param {Array} players - All available players
 * @returns {Object} - The initialized draft state
 */
export const initializeDraft = (config = {}, players = []) => {
  // Reset the draft state
  draftState = { ...initialDraftState };
  
  // Update configuration if provided
  if (config) {
    draftState.config = {
      ...draftState.config,
      ...config
    };
  }
  
  // Initialize player status for all players
  if (players && players.length) {
    players.forEach(player => {
      draftState.playerStatus[player.PLAYER_ID] = DRAFT_STATUS.AVAILABLE;
    });
  }
  
  return draftState;
};

/**
 * Get the current draft state
 * @returns {Object} - The current draft state
 */
export const getDraftState = () => {
  return { ...draftState };
};

/**
 * Draft a player for your team
 * @param {Object} player - The player to draft
 * @returns {Object} - The updated draft state
 */
export const draftPlayerForMe = (player) => {
  // Check if the player is already drafted
  if (draftState.playerStatus[player.PLAYER_ID] !== DRAFT_STATUS.AVAILABLE) {
    console.warn(`Player ${player.PLAYER_NAME} is already drafted.`);
    return draftState;
  }
  
  // Add the player to my roster
  draftState.myRoster.push(player);
  
  // Update player status
  draftState.playerStatus[player.PLAYER_ID] = DRAFT_STATUS.DRAFTED_BY_YOU;
  
  // Add to draft history
  draftState.draftHistory.push({
    round: draftState.currentRound,
    player: player,
    draftedBy: 'me',
    timestamp: new Date()
  });
  
  // Increment round if necessary
  if (draftState.draftHistory.length % draftState.config.totalTeams === 0) {
    draftState.currentRound++;
  }
  
  return { ...draftState };
};

/**
 * Draft a player for another team
 * @param {Object} player - The player to draft
 * @returns {Object} - The updated draft state
 */
export const draftPlayerForOthers = (player) => {
  // Check if the player is already drafted
  if (draftState.playerStatus[player.PLAYER_ID] !== DRAFT_STATUS.AVAILABLE) {
    console.warn(`Player ${player.PLAYER_NAME} is already drafted.`);
    return draftState;
  }
  
  // Add the player to others drafted
  draftState.othersDrafted.push(player);
  
  // Update player status
  draftState.playerStatus[player.PLAYER_ID] = DRAFT_STATUS.DRAFTED_BY_OTHERS;
  
  // Add to draft history
  draftState.draftHistory.push({
    round: draftState.currentRound,
    player: player,
    draftedBy: 'others',
    timestamp: new Date()
  });
  
  // Increment round if necessary
  if (draftState.draftHistory.length % draftState.config.totalTeams === 0) {
    draftState.currentRound++;
  }
  
  return { ...draftState };
};

/**
 * Undo the last draft pick
 * @returns {Object} - The updated draft state
 */
export const undoLastPick = () => {
  // Check if there's anything to undo
  if (draftState.draftHistory.length === 0) {
    console.warn('No draft picks to undo.');
    return draftState;
  }
  
  // Get the last draft pick
  const lastPick = draftState.draftHistory.pop();
  
  // Update player status
  draftState.playerStatus[lastPick.player.PLAYER_ID] = DRAFT_STATUS.AVAILABLE;
  
  // Remove from the appropriate roster
  if (lastPick.draftedBy === 'me') {
    draftState.myRoster = draftState.myRoster.filter(
      player => player.PLAYER_ID !== lastPick.player.PLAYER_ID
    );
  } else {
    draftState.othersDrafted = draftState.othersDrafted.filter(
      player => player.PLAYER_ID !== lastPick.player.PLAYER_ID
    );
  }
  
  // Decrement round if necessary
  if (draftState.draftHistory.length % draftState.config.totalTeams === draftState.config.totalTeams - 1) {
    draftState.currentRound--;
  }
  
  return { ...draftState };
};

/**
 * Reset the draft
 * @returns {Object} - The reset draft state
 */
export const resetDraft = () => {
  draftState = { ...initialDraftState };
  saveDraftState()
  return draftState;
};

/**
 * Get the draft status of a player
 * @param {string} playerId - The player ID
 * @returns {string} - The draft status
 */
export const getPlayerDraftStatus = (playerId) => {
  return draftState.playerStatus[playerId] || DRAFT_STATUS.AVAILABLE;
};

/**
 * Get all available players
 * @param {Array} allPlayers - All players
 * @returns {Array} - Available players
 */
export const getAvailablePlayers = (allPlayers) => {
  if (!allPlayers) return [];
  
  return allPlayers.filter(
    player => draftState.playerStatus[player.PLAYER_ID] === DRAFT_STATUS.AVAILABLE
  );
};

/**
 * Get my drafted players
 * @returns {Array} - Players drafted by me
 */
export const getMyDraftedPlayers = () => {
  return [...draftState.myRoster];
};

/**
 * Get players drafted by others
 * @returns {Array} - Players drafted by others
 */
export const getOthersDraftedPlayers = () => {
  return [...draftState.othersDrafted];
};

/**
 * Get the draft history
 * @returns {Array} - Draft history
 */
export const getDraftHistory = () => {
  return [...draftState.draftHistory];
};

/**
 * Get the current draft round
 * @returns {number} - Current round
 */
export const getCurrentRound = () => {
  return draftState.currentRound;
};

/**
 * Get roster composition by position
 * @returns {Object} - Counts by position
 */
export const getRosterComposition = () => {
  const positionCounts = {};
  
  draftState.myRoster.forEach(player => {
    if (!player.POSITIONS) return;
    
    const positions = player.POSITIONS.split(',').map(pos => pos.trim());
    positions.forEach(position => {
      positionCounts[position] = (positionCounts[position] || 0) + 1;
    });
  });
  
  return positionCounts;
};

/**
 * Save draft state to local storage
 * @returns {boolean} - Success status
 */
export const saveDraftState = () => {
  try {
    localStorage.setItem('fantasyBaseballDraftState', JSON.stringify(draftState));
    return true;
  } catch (error) {
    console.error('Error saving draft state:', error);
    return false;
  }
};

/**
 * Load draft state from local storage
 * @returns {Object|null} - Loaded draft state or null if not found
 */
export const loadDraftState = () => {
  try {
    const savedState = localStorage.getItem('fantasyBaseballDraftState');
    if (savedState) {
      draftState = JSON.parse(savedState);
      return { ...draftState };
    }
    return null;
  } catch (error) {
    console.error('Error loading draft state:', error);
    return null;
  }
};
