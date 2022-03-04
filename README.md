# C-V2X Machine Learning

## Setup
Create virtual env:
```
python3 -m venv .
```

Activate virtual env:
MacOS / Unix:
```
source ./bin/activate
```

Windows: 
```
.\Scripts\activate.bat
```

Install packages:
```
pip install -r requirements.txt
```

### Creating Models
Run *createModel.py* by `python createModel.py` to create data to train and test the models from models.py.
```
NOTE: createModel.py uses the models located in models.py. To create a custom ML model, create a class in models.py.
```

### Evaluating Models
Run *evaluation.py* by `python evaluation.py` to evaluate and run all models located in the `models/` directory. Evaluation produces loss and accuracy metrics.

## Main Files
### CreateModel.py
Launches simulator, trains and tests the data from simulator, and saves the newly tested model under the directory `models/`

### Evaluation.py
Uses data located in `data/` and models in `models/` to run an evaluation on all models. Evaluation may include accuracy and loss relation graphs, matplotlib images to show resource pools being used, etc.

## Subsidiary Files and Libraries
### Models.py
A library using TensorFlow models. Each class has a different ML model for ease of training and testing our models. Each model is saved under `models/` after being completed (trained and tested).

### Preprocess.py
Preprocessing library that matches the required matrices to match input shapes for TensorFlow models.

### Simulator.py
A simulation class that invokes and creates artificial signals in shapes of resource pools. After creating the signals, it serializes them for better preprocessing.

## Example
Evaluation on SimpleMNISTModel
```
SimpleMNISTModel
625/625 - 1s - loss: 4.2801 - sparse_categorical_accuracy: 0.4997 - 783ms/epoch - 1ms/step
accuracy:       49.970000982284546%
loss:   428.01265716552734%
```