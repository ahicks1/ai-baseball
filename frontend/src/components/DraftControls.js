import React, { useState } from 'react';
import { useDraft } from '../contexts/DraftContext';

const DraftControls = () => {
  const { 
    draftPlayerForMe, 
    draftPlayerForOthers, 
    undoLastPick, 
    resetDraft,
    updateDraftConfig,
    draftConfig
  } = useDraft();
  
  // State for draft configuration form
  const [configForm, setConfigForm] = useState({
    totalRounds: draftConfig.totalRounds,
    totalTeams: draftConfig.totalTeams,
    myPosition: draftConfig.myPosition
  });
  
  // State for showing/hiding configuration modal
  const [showConfig, setShowConfig] = useState(false);
  
  // Handle configuration form changes
  const handleConfigChange = (e) => {
    const { name, value } = e.target;
    setConfigForm({
      ...configForm,
      [name]: parseInt(value, 10)
    });
  };
  
  // Save configuration changes
  const saveConfig = () => {
    updateDraftConfig(configForm);
    setShowConfig(false);
  };
  
  return (
    <div className="draft-controls">
      <div className="draft-actions">
        <button 
          className="action-button undo-button" 
          onClick={undoLastPick}
          title="Undo the last draft pick"
        >
          Undo Last Pick
        </button>
        
        <button 
          className="action-button reset-button" 
          onClick={resetDraft}
          title="Reset the entire draft"
        >
          Reset Draft
        </button>
        
        <button 
          className="action-button config-button" 
          onClick={() => setShowConfig(!showConfig)}
          title="Configure draft settings"
        >
          {showConfig ? 'Hide Settings' : 'Draft Settings'}
        </button>
      </div>
      
      {showConfig && (
        <div className="draft-config">
          <h3>Draft Settings</h3>
          
          <div className="config-form">
            <div className="form-group">
              <label htmlFor="totalRounds">Total Rounds:</label>
              <input 
                type="number" 
                id="totalRounds" 
                name="totalRounds" 
                min="1" 
                max="50" 
                value={configForm.totalRounds} 
                onChange={handleConfigChange} 
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="totalTeams">Total Teams:</label>
              <input 
                type="number" 
                id="totalTeams" 
                name="totalTeams" 
                min="2" 
                max="20" 
                value={configForm.totalTeams} 
                onChange={handleConfigChange} 
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="myPosition">My Draft Position:</label>
              <input 
                type="number" 
                id="myPosition" 
                name="myPosition" 
                min="1" 
                max={configForm.totalTeams} 
                value={configForm.myPosition} 
                onChange={handleConfigChange} 
              />
            </div>
            
            <button 
              className="action-button save-button" 
              onClick={saveConfig}
            >
              Save Settings
            </button>
          </div>
        </div>
      )}
      
      <div className="draft-instructions">
        <h3>Draft Actions</h3>
        <p>
          Click on a player in the table to see draft options. You can draft a player for your team or mark them as drafted by another team.
        </p>
        <p>
          Use the "Undo Last Pick" button to reverse the most recent draft selection.
        </p>
      </div>
    </div>
  );
};

export default DraftControls;
