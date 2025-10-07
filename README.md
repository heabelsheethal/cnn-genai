## CNN GenAI

A modular PyTorch + FastAPI project for building, training, evaluating, and serving Convolutional Neural Networks (CNNs).
This project demonstrates the use of multiple neural network architectures (FCNN, CNN, EnhancedCNN, and Assignment2 CNN) on the CIFAR-10 dataset.


## Project Structure
```
cnn_genai/
├── app/
│   ├── __init__.py
│   └── api.py                # FastAPI app with prediction endpoints
├── helper_lib/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities (CIFAR-10)
│   ├── evaluator.py          # Model evaluation functions
│   ├── trainer.py            # Training loop implementation
│   ├── model.py              # Model selector / factory
│   ├── fcnn.py               # Fully Connected Neural Network (FCNN)
│   ├── simple_cnn.py         # Basic CNN architecture
│   ├── enhanced_cnn.py       # CNN with BatchNorm and Dropout
│   └── assignment2.py        # 64×64 CNN architecture for assignment
├── main.py                   # Command-line training script
├── pyproject.toml            # Dependencies and project metadata
├── .python-version           # Python version (>=3.12)
├── .gitignore                # Ignored files and directories
└── README.md                 # Project documentation
```


## Datasets

This project uses the **CIFAR-10** dataset. 
The dataset will be automatically downloaded and stored in the `data/` folder when you run the `main.py` script for the first time.


## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd cnn_genai
```


2.	Create and Activate a Virtual Environment:

```bash
python -m venv .venv            # Create venv

source .venv/bin/activate       # Activate (Mac/Linux)

.venv\Scripts\activate          # Activate (Windows)
```

3.	Install dependencies:

```bash
uv sync
```

## Train a Model

Train any of the supported models using:
```bash
python main.py --model CNN
python main.py --model FCNN
python main.py --model EnhancedCNN
python main.py --model CNN_Assignment2

```

Available models:
	•	FCNN — Fully Connected Neural Network
	•	CNN — Simple 2-layer CNN
	•	EnhancedCNN — Deeper CNN with batch normalization & dropout
	•	CNN_Assignment2 — 64×64 CNN for extended architecture testing

Each trained model is saved in:
```bash
models/<model_name>.pth
```


## Evaluate a Model

Evaluation runs automatically after training in main.py.
Alternatively, you can load the saved model and run:
```bash
from helper_lib.evaluator import evaluate_model
```


##  Run FastAPI Server

Start the FastAPI server:
```bash
uvicorn app.api:app --reload
```
Once running, visit:
	•	Root endpoint: http://127.0.0.1:8000/
	•	Interactive docs: http://127.0.0.1:8000/docs







