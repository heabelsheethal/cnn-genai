# CNN-GenAI

A PyTorch project to train and evaluate different neural network models (FCNN, SimpleCNN, and EnhancedCNN) on the CIFAR-10 dataset.


## Project Structure
```
cnn_genai/
├── helper_lib/
│   ├── data_loader.py     # Data loading module
│   ├── evaluator.py       # Evaluation module
│   ├── model.py           # Model definitions
│   └── trainer.py         # Training module
├── main.py                # Entry point
├── pyproject.toml         # Project metadata & dependencies
├── README.md              # Project documentation
└── .python-version        # Python version
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

