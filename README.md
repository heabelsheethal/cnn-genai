## CNN GenAI

A modular PyTorch + FastAPI project for building, training, evaluating, and serving Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs).

This repo supports multiple model architectures:
	•	FCNN, CNN, EnhancedCNN, CNN_Assignment2 → trained on CIFAR-10
	•	GAN (WGAN-CelebA) → generates 64×64 color faces
	•	GAN_Assignment3 (Standard GAN-MNIST) → generates 28×28 grayscale digits


## Project Structure
```
cnn_genai/
├── app/
│   ├── __init__.py
│   └── api.py                    # FastAPI app with prediction + generation endpoints
├── helper_lib/
│   ├── __init__.py
│   ├── data_loader.py            # Handles CIFAR-10, MNIST, CelebA
│   ├── evaluator.py              # Evaluation utilities
│   ├── trainer.py                # Training loops (CNN, GAN, Diffusion, Energy)
│   ├── model.py                  # Model selector / factory
│   ├── fcnn.py                   # Fully Connected Network
│   ├── simple_cnn.py             # Basic CNN
│   ├── enhanced_cnn.py           # CNN + BatchNorm + Dropout
│   ├── assignment2.py            # 64×64 CNN architecture (Assignment 2)
│   ├── gan.py                    # WGAN (CelebA)
│   ├── assignment3_gan.py        # Standard GAN (MNIST)
│   ├── diffusion_model.py        # UNet Diffusion model with offset cosine schedule
│   ├── energy_model.py           # Energy-based model (EBM)
│   └── generator.py              # Generation utilities (Diffusion & Energy)
├── main.py                       # Command-line training driver
├── pyproject.toml                # Dependencies / metadata
├── uv.lock                       # Locked environment
├── .gitignore
└── README.md
```


## Datasets

•	CIFAR-10 (32×32 RGB) → classification, diffusion, energy
•	CelebA (64×64 faces) → WGAN
•	MNIST (28×28 digits) → Assignment 3 GAN

All datasets download automatically into data/ when you run main.py.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/heabelsheethal/cnn-genai.git
cd cnn_genai

```


2.	Create and Activate a Virtual Environment:

```bash
python -m venv .venv
source .venv/bin/activate        # (Mac/Linux)
.venv\Scripts\activate          # Activate (Windows)
```

3.	Install dependencies:

```bash
uv sync                          # install dependencies
```

## Train a Model

Train any of the supported models using:
```bash
python main.py --model CNN
python main.py --model FCNN
python main.py --model EnhancedCNN
python main.py --model CNN_Assignment2
python main.py --model GAN
python main.py --model GAN_Assignment3
python main.py --model Diffusion
python main.py --model Energy

```

Available models:

	•	FCNN — Fully Connected Neural Network

	•	CNN — Simple 2-layer CNN

	•	EnhancedCNN — Deeper CNN with batch normalization & dropout

	•	CNN_Assignment2 — 64×64 CNN for extended architecture testing

	•	GAN -  Wasserstein GAN (CelebA, 64×64 RGB faces)

	•	GAN_Assignment3 - Standard GAN (MNIST, 28×28 digits)

	•	Diffusion - UNet Diffusion model with offset-cosine schedule + EMA (CIFAR-10)
	•	Energy - Convolutional Energy-Based Model (EBM) with Langevin sampling (CIFAR-10)




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



## Author
Sheethal Heabel




