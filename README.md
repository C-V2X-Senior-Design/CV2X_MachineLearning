# C-V2X Machine Learning

## Setup
`pip install -r requirements.txt`

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