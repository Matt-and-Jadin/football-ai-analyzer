# Football AI Analyzer

AI-powered football film analysis for coaches. This project analyzes game film to identify formations, player personnel, and provide play recommendations.

## Current Focus

We are currently in the **research/development phase** - building the machine learning and computer vision pipeline. The physical app interface will come later once the core AI functionality is proven.

## Project Structure

```
football-ai-analyzer/
├── data/              # Raw game film and labeled data
├── models/            # Trained ML models
├── notebooks/         # Jupyter notebooks for experimentation
├── src/               # Core AI pipeline code
│   ├── detection/     # Computer vision (players, formations)
│   ├── analysis/      # Film analysis logic
│   └── ml/            # Machine learning models
└── tests/             # Unit tests
```

## Setup

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA (optional but recommended for speed)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mattmcc78/football-ai-analyzer.git
cd football-ai-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Player Detection

#### Option 1: Command Line
1. Place your MP4 video file in `data/raw/`
2. Update `config.yaml` to set `video.input_path` to your video file
3. Run:
```bash
python src/detection/player_detector.py
```

#### Option 2: Jupyter Notebook
1. Start Jupyter:
```bash
jupyter notebook notebooks/
```
2. Open `test_player_detection.ipynb`
3. Run the cells to test detection

### Configuration

Edit `config.yaml` to adjust:
- Model size (yolov8n for speed, yolov8x for accuracy)
- Confidence threshold
- Video resolution
- Output settings

## Collaboration

This is a collaborative project between two engineers. Both contributors can push to this repository.

## License

Proprietary - All rights reserved
