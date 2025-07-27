# Area Occupancy Prediction Model

A probabilistic machine learning system for predicting area occupancy based on sensor data. This project uses Naive Bayes and Hidden Markov Models to analyze historical sensor readings and predict whether areas are currently occupied.

## Features

- **Time-based Priors**: Learn occupancy patterns by day of week and time of day
- **Sensor Likelihoods**: Calculate probability of sensor readings given occupancy state
- **Naive Bayes Prediction**: Real-time occupancy prediction from current sensor readings
- **HMM Smoothing**: Temporal smoothing of occupancy predictions over time
- **Multi-sensor Support**: Handles motion sensors, light sensors, and other binary sensors
- **CLI Interface**: Command-line tools for training models and making predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd area_probability
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python main.py init
```

## Quick Start

### 1. Import Historical Data

Import sensor data from a SQL file:
```bash
python main.py import --file your_data.sql
```

### 2. Train the Model

Compute time-based priors and sensor likelihoods for all areas:
```bash
python main.py learn-all --slot 60
```

Or train for a specific area:
```bash
python main.py priors living_room --slot 60
python main.py likelihoods living_room
```

### 3. Make Predictions

Predict occupancy based on current sensor readings:
```bash
python main.py predict living_room --entities motion_sensor light_sensor --values 1 0
```

### 4. Analyze Timeline Data

Run HMM analysis over a CSV timeline:
```bash
python main.py hmm living_room --timeline sensor_data.csv --slot 15
```

## Database Schema

The system uses the following main tables:

- `area_occupancy`: Area configurations and thresholds
- `entities`: Sensor entity definitions
- `area_entity_config`: Sensor configurations per area with learned likelihoods
- `area_time_priors`: Time-based occupancy priors (day of week + time slot)
- `state_intervals`: Historical sensor state data

## CLI Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize database schema |
| `import` | Import data from SQL file |
| `priors` | Compute time-based occupancy priors |
| `likelihoods` | Compute sensor likelihoods |
| `predict` | Make real-time occupancy prediction |
| `hmm` | Run HMM analysis over timeline |
| `learn-all` | Train models for all areas |
| `reset` | Reset all learned parameters |
| `help` | Show detailed help |

## Example Workflow

```bash
# 1. Set up database and import data
python main.py init
python main.py import --file export.sql

# 2. Train models for all areas
python main.py learn-all --slot 30

# 3. Make a prediction
python main.py predict bedroom --entities motion_sensor door_sensor --values 1 0
# Output: Naïve Bayes P(occupied) = 0.847

# 4. Compute area prior
python main.py area-prior bedroom
# Output: Updated area prior for bedroom: 0.234
```

## Time Slot Configuration

The `--slot` parameter controls the time resolution for learning patterns:
- `60` minutes: Hourly patterns (24 slots per day)
- `30` minutes: Half-hourly patterns (48 slots per day)  
- `15` minutes: Quarter-hourly patterns (96 slots per day)

## CSV Format for Timeline Analysis

For HMM timeline analysis, use CSV format:
```csv
timestamp,entity_id,value
2024-01-01T10:00:00,motion_sensor,1
2024-01-01T10:00:00,light_sensor,0
2024-01-01T10:15:00,motion_sensor,0
```

## License

[Add your license here] 