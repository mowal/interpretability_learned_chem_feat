# Interpreting Neural Network Models for Toxicity Prediction by Extracting Learned Chemical Features

Accompanying code and data for the manuscript: <!-- TODO: Add link -->

## Respository structure

| Folder            | Description |
| -----------       | ----------- |
| code              | Python code for model building, interpretation and visualisation|
| data              | Training, test and validation data for the public Ames mutagenicity models|
| pythorch_models   | Built pytorch model on public Ames mutagenicity data |
| results           | Interpretation results in terms of atom attribution for the public Ames mutagencity model |

## Description of Python scripts
`./code/gridsearch_1layer_model_Ames.py`: does a gridearch of hyperparameters for the model on experimental Ames data and saves the best model.
`./code/do_substructure_extraction.py`: for each hidden neuron extracts and stores chemical substructures that acivate the neuron.
`./code/do_ig_input.py`: generates model explanations using IG_input and calculates attribution AUC values if ground truth is provided.
`./code/do_ig_hidden.py`: generates model explanations using IG_hidden and calculates attribution AUC values if ground truth is provided.

## Interpretation Example

The notebook `./code/visualise_model_explanations.ipynb` can be used to visualise the interpretation of predictions
of the validation set for the public Ames mutagencity model. 

## Depenencies

| Library           | Version       |
| -----------       | -----------   |
| Python            | 3.9.5         |
| RDKit             | 2021.03.3     |
| Pytorch           | 1.9.0         |
| Pandas            | 1.2.4         |
| Scikit-learn      | 0.24.2        |
| Numpy             | 1.20.3        |
| Concepts          | 0.9.2         |
| Captum            | 0.4.0         |
