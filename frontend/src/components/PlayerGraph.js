import React, { useState, useEffect } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend } from 'chart.js';
import { getStatsForGraphing } from '../utils/statsCalculator';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const PlayerGraph = ({ players, availableStats }) => {
  // State for selected stat to graph
  const [selectedStat, setSelectedStat] = useState('');
  
  // State for graph type
  const [graphType, setGraphType] = useState('bar');
  
  // State for number of players to show
  const [playerCount, setPlayerCount] = useState(10);
  
  // State for chart data
  const [chartData, setChartData] = useState(null);
  
  // Set initial selected stat when available stats change
  useEffect(() => {
    if (availableStats && availableStats.length > 0 && !selectedStat) {
      setSelectedStat(availableStats[0]);
    }
  }, [availableStats, selectedStat]);
  
  // Update chart data when players, selected stat, or player count changes
  useEffect(() => {
    if (!players || !selectedStat) return;
    
    // Limit to the specified number of players
    const limitedPlayers = players.slice(0, playerCount);
    
    // Get data for graphing
    const { labels, data } = getStatsForGraphing(limitedPlayers, selectedStat);
    
    // Create chart data
    const chartData = {
      labels,
      datasets: [
        {
          label: selectedStat,
          data,
          backgroundColor: 'rgba(54, 162, 235, 0.5)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1,
        },
      ],
    };
    
    setChartData(chartData);
  }, [players, selectedStat, playerCount]);
  
  // Handle stat selection change
  const handleStatChange = (e) => {
    setSelectedStat(e.target.value);
  };
  
  // Handle graph type change
  const handleGraphTypeChange = (e) => {
    setGraphType(e.target.value);
  };
  
  // Handle player count change
  const handlePlayerCountChange = (e) => {
    setPlayerCount(parseInt(e.target.value, 10));
  };
  
  // Chart options
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `${selectedStat} Comparison`,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };
  
  return (
    <div className="player-graph">
      <div className="graph-controls">
        <div className="control-group">
          <label htmlFor="stat-select">Stat to Graph:</label>
          <select 
            id="stat-select" 
            value={selectedStat} 
            onChange={handleStatChange}
          >
            {availableStats.map(stat => (
              <option key={stat} value={stat}>
                {stat}
              </option>
            ))}
          </select>
        </div>
        
        <div className="control-group">
          <label htmlFor="graph-type">Graph Type:</label>
          <select 
            id="graph-type" 
            value={graphType} 
            onChange={handleGraphTypeChange}
          >
            <option value="bar">Bar Chart</option>
            <option value="line">Line Chart</option>
          </select>
        </div>
        
        <div className="control-group">
          <label htmlFor="player-count">Number of Players:</label>
          <input 
            id="player-count" 
            type="number" 
            min="1" 
            max="50" 
            value={playerCount} 
            onChange={handlePlayerCountChange} 
          />
        </div>
      </div>
      
      <div className="graph-container">
        {chartData ? (
          graphType === 'bar' ? (
            <Bar data={chartData} options={chartOptions} />
          ) : (
            <Line data={chartData} options={chartOptions} />
          )
        ) : (
          <div className="no-data">No data available for graphing</div>
        )}
      </div>
    </div>
  );
};

export default PlayerGraph;
