import React from 'react';
import { useDraft } from '../contexts/DraftContext';

const Header = () => {
  const { currentRound, draftConfig, myRoster, getRosterComposition } = useDraft();
  
  // Get roster composition
  const rosterComposition = getRosterComposition();
  
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="app-title">
          <h1>Fantasy Baseball Draft Tool</h1>
        </div>
        
        <div className="draft-info">
          <div className="draft-round">
            <span className="label">Round:</span>
            <span className="value">{currentRound} of {draftConfig.totalRounds}</span>
          </div>
          
          <div className="roster-count">
            <span className="label">Roster:</span>
            <span className="value">{myRoster.length} of {draftConfig.totalRounds}</span>
          </div>
        </div>
        
        <div className="roster-composition">
          <h3>Roster Composition</h3>
          <div className="position-counts">
            {Object.entries(rosterComposition).map(([position, count]) => (
              <div key={position} className="position-count">
                <span className="position">{position}:</span>
                <span className="count">{count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
