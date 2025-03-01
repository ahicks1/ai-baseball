# Fantasy Baseball Draft Tool

A Python-based tool for fantasy baseball draft analysis and player forecasting.

## Overview

This tool helps fantasy baseball players make informed draft decisions by:
- Processing historical baseball data from Retrosheet
- Analyzing player performance across relevant fantasy categories
- Forecasting player performance for the upcoming season
- Ranking players based on projected value in H2H categories leagues
- Providing a command-line interface for draft assistance

## League Settings

- **Format**: Head-to-Head Categories
- **Hitting Categories (6)**: HR, OBP, R, RBI, SB, TB
- **Pitching Categories (5)**: ERA, WHIP, K, Saves+Holds, Wins+Quality Starts

## Project Structure

```
fantasy-baseball-draft-tool/
├── data/                      # Directory for data files
│   ├── raw/                   # Raw Retrosheet data
│   ├── processed/             # Processed player statistics
│   └── projections/           # Generated player projections
├── src/                       # Source code
│   ├── data_processing/       # Data processing modules
│   ├── analysis/              # Statistical analysis modules
│   ├── forecasting/           # Forecasting model modules
│   ├── ranking/               # Player ranking modules
│   └── draft_tool/            # Draft interface modules
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Test modules
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download Retrosheet data to the `data/raw/` directory
4. Run data processing: `python src/data_processing/process_retrosheet.py`
5. Generate projections: `python src/forecasting/generate_projections.py`
6. Start draft tool: `python src/draft_tool/cli.py`

## Data Sources

This tool uses data from [Retrosheet](https://www.retrosheet.org/), which provides play-by-play data for MLB games.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Retrosheet for providing the historical baseball data
