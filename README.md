# CNN-GenAI

A PyTorch project to train and evaluate different neural network models (FCNN, SimpleCNN, and EnhancedCNN) on the CIFAR-10 dataset.


## Project Structure
```
cnn_genai/
├── app/                  # API layer using FastAPI
│   ├── __init__.py
│   └── api.py
├── helper_lib/           # Core library: models, data loaders, trainers, evaluation
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── model.py
│   ├── fcnn.py
│   ├── assignment2.py
│   ├── simple_cnn.py
│   ├── enhanced_cnn.py
│   └── trainer.py
├── main.py               # CLI entry point for training
├── pyproject.toml        # Dependencies and metadata
├── README.md
└── .python-version
```



## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd cnn_genai
```


2.	Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
```

3.	Install dependencies:
```bash
pip install torch torchvision tqdm
```
## Datasets

This project uses the **CIFAR-10** dataset. 
The dataset will be automatically downloaded and stored in the `data/` folder when you run the `main.py` script for the first time.
Folder structure after downloading:


## Usage

Run the main script to train and evaluate a model. 
You can select the model via the --model argument:
```bash
python main.py --model FCNN
python main.py --model CNN
python main.py --model EnhancedCNN
```

Available models:
	•	FCNN       - Fully Connected Neural Network
	•	CNN        - Simple Convolutional Neural Network
	•	EnhancedCNN - CNN with additional layers, BatchNorm, and Dropout


## Results

After training, the script prints:
	•	Epoch-wise training loss and accuracy
	•	Test set accuracy and average loss

