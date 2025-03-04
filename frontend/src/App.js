import React from 'react';
import { DraftProvider } from './contexts/DraftContext';
import Header from './components/Header';
import FilterControls from './components/FilterControls';
import PlayerTable from './components/PlayerTable';
import PlayerGraph from './components/PlayerGraph';
import DraftControls from './components/DraftControls';
import usePlayerData from './hooks/usePlayerData';
import './App.css';

// Main App component
const App = () => {
  const { 
    positions, 
    positionFilter, 
    filterByPosition, 
    playerTypeFilter, 
    filterByPlayerType,
    searchQuery,
    searchPlayers,
    filteredPlayers,
    availableStats,
    loading,
    error
  } = usePlayerData();
  
  return (
    <DraftProvider>
      <div className="app">
        <Header />
        
        <main className="main-content">
          <div className="controls-section">
          <FilterControls 
            positions={positions}
            positionFilter={positionFilter}
            setPositionFilter={filterByPosition}
            playerTypeFilter={playerTypeFilter}
            setPlayerTypeFilter={filterByPlayerType}
            searchQuery={searchQuery}
            setSearchQuery={searchPlayers}
          />
            <DraftControls />
          </div>
          
          <div className="data-section">
            {loading ? (
              <div className="loading">Loading player data...</div>
            ) : error ? (
              <div className="error">{error}</div>
            ) : (
              <>
                <div className="table-section">
                  <h2>Player Rankings</h2>
                  <PlayerTable 
                    players={filteredPlayers} 
                    availableStats={availableStats} 
                  />
                </div>
                
                <div className="graph-section">
                  <h2>Player Comparison</h2>
                  <PlayerGraph 
                    players={filteredPlayers.slice(0, 10)} 
                    availableStats={availableStats} 
                  />
                </div>
              </>
            )}
          </div>
        </main>
        
        <footer className="app-footer">
          <p>Fantasy Baseball Draft Tool &copy; 2025</p>
        </footer>
      </div>
    </DraftProvider>
  );
};

export default App;
