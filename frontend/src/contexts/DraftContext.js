import React, { createContext, useContext, useState, useEffect } from 'react';
import * as draftService from '../services/draftService';
import * as dataService from '../services/dataService';

// Create the context
const DraftContext = createContext();

// Custom hook to use the draft context
export const useDraft = () => {
  const context = useContext(DraftContext);
  if (!context) {
    throw new Error('useDraft must be used within a DraftProvider');
  }
  return context;
};

// Draft provider component
export const DraftProvider = ({ children }) => {
  // State for all players
  const [allPlayers, setAllPlayers] = useState([]);
  
  // State for available players
  const [availablePlayers, setAvailablePlayers] = useState([]);
  
  // State for my roster
  const [myRoster, setMyRoster] = useState([]);
  
  // State for players drafted by others
  const [othersDrafted, setOthersDrafted] = useState([]);
  
  // State for draft history
  const [draftHistory, setDraftHistory] = useState([]);
  
  // State for current round
  const [currentRound, setCurrentRound] = useState(1);
  
  // State for loading status
  const [loading, setLoading] = useState(true);
  
  // State for error
  const [error, setError] = useState(null);
  
  // State for draft configuration
  const [draftConfig, setDraftConfig] = useState({
    totalRounds: 23,
    totalTeams: 12,
    myPosition: 1
  });

  // Load player data and initialize draft
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Load player data
        const players = await dataService.loadPlayerData('overall');
        setAllPlayers(players);
        
        // Try to load saved draft state
        const savedState = draftService.loadDraftState();
        
        if (savedState) {
          // Update state from saved draft state
          updateStateFromDraftState(savedState);
        } else {
          // Initialize new draft
          const initialState = draftService.initializeDraft(draftConfig, players);
          updateStateFromDraftState(initialState);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error loading data:', err);
        setError('Failed to load player data. Please try again.');
        setLoading(false);
      }
    };
    
    loadData();
  }, []);

  // Update local state from draft service state
  const updateStateFromDraftState = (draftState) => {
    setMyRoster(draftState.myRoster);
    setOthersDrafted(draftState.othersDrafted);
    setDraftHistory(draftState.draftHistory);
    setCurrentRound(draftState.currentRound);
    setDraftConfig(draftState.config);
    
    // Update available players
    if (allPlayers.length > 0) {
      const available = draftService.getAvailablePlayers(allPlayers);
      setAvailablePlayers(available);
    }
  };

  // Draft a player for your team
  const draftPlayerForMe = (player) => {
    const updatedState = draftService.draftPlayerForMe(player);
    updateStateFromDraftState(updatedState);
    draftService.saveDraftState();
  };

  // Draft a player for another team
  const draftPlayerForOthers = (player) => {
    const updatedState = draftService.draftPlayerForOthers(player);
    updateStateFromDraftState(updatedState);
    draftService.saveDraftState();
  };

  // Undo the last draft pick
  const undoLastPick = () => {
    const updatedState = draftService.undoLastPick();
    updateStateFromDraftState(updatedState);
    draftService.saveDraftState();
  };

  // Reset the draft
  const resetDraft = () => {
    const updatedState = draftService.resetDraft();
    const initializedState = draftService.initializeDraft(draftConfig, allPlayers);
    updateStateFromDraftState(initializedState);
    draftService.saveDraftState();
  };

  // Update draft configuration
  const updateDraftConfig = (newConfig) => {
    setDraftConfig(prevConfig => {
      const updatedConfig = { ...prevConfig, ...newConfig };
      const updatedState = draftService.initializeDraft(updatedConfig, allPlayers);
      updateStateFromDraftState(updatedState);
      draftService.saveDraftState();
      return updatedConfig;
    });
  };

  // Get player draft status
  const getPlayerDraftStatus = (playerId) => {
    return draftService.getPlayerDraftStatus(playerId);
  };

  // Get roster composition
  const getRosterComposition = () => {
    return draftService.getRosterComposition();
  };

  // Context value
  const value = {
    allPlayers,
    availablePlayers,
    myRoster,
    othersDrafted,
    draftHistory,
    currentRound,
    draftConfig,
    loading,
    error,
    draftPlayerForMe,
    draftPlayerForOthers,
    undoLastPick,
    resetDraft,
    updateDraftConfig,
    getPlayerDraftStatus,
    getRosterComposition
  };

  return (
    <DraftContext.Provider value={value}>
      {children}
    </DraftContext.Provider>
  );
};

export default DraftContext;
