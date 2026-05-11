# LiteGL-Net: Lightweight Global-Local Network for Low-Light Image Enhancement

This repository contains the source code, training scripts, and an API demo for the **LiteGL-Net** model. The project structure has been refactored for scientific research and deployment.

## Directory Structure

```text
├── README.md
├── requirements.txt
├── app.py                          # Streamlit Demo
├── ckpts/                          # Pre-trained models
│   ├── lolv1.ckpt
│   └── lolv2-real.ckpt
├── scripts/                        # Shell scripts for execution
│   └── run_train.sh                # Script to run training
└── src/
    ├── api/                        # FastAPI demo application
    │   └── main.py
    ├── data/                       # Dataset loading and processing
    │   └── dataset.py
    ├── models/                     # Model architecture and components
    │   ├── components.py
    │   └── model.py
    ├── train/                      # Training logic and custom losses
    │   ├── loss.py
    │   └── trainer.py              # PyTorch Lightning module
    └── main.py                     # Main entrypoint (CLI for train/eval)
```

## Setup & Installation

**Prerequisites:** Python 3.12 (Recommended). Higher versions like Python 3.13 may cause compatibility issues with certain libraries.

1. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Download your dataset (e.g., LOL dataset) and update the paths in the scripts or CLI arguments.

## Usage

### Training

To train the model, you can either run the bash script:
```bash
bash scripts/run_train.sh
```

Or use the python CLI directly:
```bash
python -m src.main train --data_dir /path/to/dataset --batch_size 2 --epochs 200
```

### Evaluation

To evaluate a checkpoint:
```bash
python -m src.main eval --ckpt ./ckpts/lolv2-real.ckpt --data_dir /path/to/dataset
```

### FastAPI Backend

To run the local inference API server:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
Then visit `http://localhost:8000/docs` to test the `/enhance` endpoint.

### Streamlit Frontend Demo

After starting the FastAPI server, run the Streamlit UI in a separate terminal:
```bash
streamlit run app.py
```
The browser will open automatically at `http://localhost:8501`.

The UI allows you to:
- Upload a low-light image (JPG / PNG)
- Send it to the FastAPI backend for enhancement
- View the side-by-side before/after comparison
- Download the enhanced image
