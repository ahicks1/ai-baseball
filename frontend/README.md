# Fantasy Baseball Draft Tool Frontend

A React-based web frontend for fantasy baseball draft analysis and player forecasting.

## Overview

This tool helps fantasy baseball players make informed draft decisions by:
- Displaying player rankings and forecasted stats
- Simulating draft scenarios
- Visualizing player comparisons
- Tracking your roster during the draft

## Features

- **Player Table**
  - Sortable columns for all stats
  - Filtering by position, player type, etc.
  - Mark players as drafted by you or others
  - Configurable columns to show specific stats

- **Draft Simulation**
  - Track your roster composition
  - Maintain a list of available players
  - Simulate draft scenarios
  - Undo draft picks
  - Reset draft

- **Visualization**
  - Compare stats across players
  - Configurable graph types (bar, line)
  - Select specific stats to visualize
  - Adjust number of players to compare

## Implementation Details

### Data Loading

The application loads player data from CSV files:
- `batting_rankings.csv`: Contains batting stats, Z-scores, and rankings
- `pitching_rankings.csv`: Contains pitching stats, Z-scores, and rankings
- `overall_rankings.csv`: Combined rankings for all players

### Key Components

1. **PlayerTable**: Displays player data in a sortable, filterable table
   - Uses @tanstack/react-table for advanced table functionality
   - Allows marking players as drafted
   - Supports pagination and column selection

2. **PlayerGraph**: Visualizes player stats for comparison
   - Uses Chart.js for data visualization
   - Supports different chart types
   - Configurable to show different stats

3. **DraftControls**: Provides controls for draft simulation
   - Undo last pick
   - Reset draft
   - Configure draft settings

4. **FilterControls**: Allows filtering the player list
   - Filter by position
   - Filter by player type
   - Search by player name

### State Management

- Uses React Context API for global state management
- Maintains draft state (drafted players, available players)
- Persists draft state to localStorage

## Getting Started

1. Install dependencies:
   ```
   npm install
   ```

2. Copy data files (automatically done before start/build):
   ```
   npm run copy-data
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Build for production:
   ```
   npm run build
   ```

## Usage

1. **Viewing Player Data**:
   - Sort columns by clicking on column headers
   - Filter players using the filter controls
   - Search for specific players

2. **Draft Simulation**:
   - Click on a player to select them
   - Choose to draft for your team or mark as drafted by others
   - View your roster composition in the header
   - Use the undo button to reverse the last draft pick

3. **Player Comparison**:
   - Select a stat to visualize in the graph section
   - Choose between bar and line charts
   - Adjust the number of players to compare

## Technologies Used

- React 19
- @tanstack/react-table for table functionality
- Chart.js and react-chartjs-2 for data visualization
- PapaParse for CSV parsing
