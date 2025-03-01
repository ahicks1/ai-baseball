# Data Directory

This directory contains the data files used by the Fantasy Baseball Draft Tool.

## Directory Structure

- `raw/`: Raw Retrosheet data files
- `processed/`: Processed player statistics
- `projections/`: Generated player projections
- `rankings/`: Player rankings for fantasy drafts
- `analysis/`: Statistical analysis results
- `draft/`: Draft simulation results

## Retrosheet Data

To use this tool with real MLB data, you need to download play-by-play data from [Retrosheet](https://www.retrosheet.org/).

### How to Download Retrosheet Data

1. Visit the [Retrosheet Event Files](https://www.retrosheet.org/game.htm) page
2. Download the event files for the seasons you want to analyze (e.g., 2014-2023)
3. Extract the downloaded files to the `data/raw/` directory

### File Format

Retrosheet data comes in several formats:

- `.EVN` and `.EVA` files: Play-by-play event files for National and American League games
- `.ROS` files: Team roster files
- `.TEAM` files: Team information

The data processing module will parse these files and convert them into player statistics.

## Data Processing Pipeline

The tool processes data through the following stages:

1. Raw Retrosheet data → Processed player statistics
2. Processed statistics → Statistical analysis
3. Processed statistics → Player projections
4. Player projections → Player rankings
5. Player rankings → Draft simulation

Each stage saves its output to the corresponding subdirectory.
