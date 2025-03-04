import React from 'react';

const FilterControls = ({ 
  positions, 
  positionFilter, 
  setPositionFilter, 
  playerTypeFilter, 
  setPlayerTypeFilter,
  searchQuery,
  setSearchQuery
}) => {
  return (
    <div className="filter-controls">
      <div className="filter-section">
        <label htmlFor="position-filter">Position:</label>
        <select 
          id="position-filter" 
          value={positionFilter} 
          onChange={(e) => setPositionFilter(e.target.value)}
        >
          {positions.map(position => (
            <option key={position} value={position}>
              {position}
            </option>
          ))}
        </select>
      </div>
      
      <div className="filter-section">
        <label htmlFor="player-type-filter">Player Type:</label>
        <select 
          id="player-type-filter" 
          value={playerTypeFilter} 
          onChange={(e) => setPlayerTypeFilter(e.target.value)}
        >
          <option value="ALL">All</option>
          <option value="BATTER">Batters</option>
          <option value="PITCHER">Pitchers</option>
        </select>
      </div>
      
      <div className="filter-section">
        <label htmlFor="search-players">Search:</label>
        <input 
          id="search-players" 
          type="text" 
          placeholder="Search players..." 
          value={searchQuery} 
          onChange={(e) => setSearchQuery(e.target.value)} 
        />
      </div>
    </div>
  );
};

export default FilterControls;
