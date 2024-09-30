# Project Overview

This project focuses on optimizing stellarator configurations using machine learning models, including a fully connected neural network. The project leverages the [QSC](https://landreman.github.io/pyQSC/) and [DESC](https://desc-docs.readthedocs.io/) codes to compute specific physical parameters of interest, such as magnetic field gradients and plasma beta, and integrates these calculations into the model's loss function.

The goal is to develop a model that updates an initial stellarator configuration to one that minimizes the differences between key quantities computed by QSC and DESC. Currently, the project is a work in progress and is not yet fully functional. Future updates will address convergence issues and improve the integration of physical constraints.

# Project Structure

The project is divided into several key Python scripts, each responsible for a different aspect of the workflow:

### 1. `data_preprocessing.py`
This file handles the data loading and preprocessing. The script:
- Loads the dataset `pass_desc.csv`, which contains 100000 rows of stellarator configurations, each with 20 parameters.
- Splits the data into training, validation, and testing sets using an 80/20/20 ratio.
- Converts the data into PyTorch tensors and organizes them into DataLoader objects for batch processing.

### 2. `model_definition.py`
This file contains the architecture of the neural network model. The model is a fully connected neural network with two hidden layers:
- Input dimension: 10 (corresponding to the 10 features in the dataset).
- Hidden layers: Two layers with 256 units each, ReLU activations, and Batch Normalization.
- Output dimension: 10 (same as input, since the task is to update the configuration).

It also includes an optional weight initialization function (`init_weights`) that uses Kaiming initialization for the linear layers.

### 3. `loss_function_qsc_desc.py`
This script defines a custom loss function that integrates QSC and DESC computations:
- For each stellarator configuration (i.e., for each batch), it computes quantities such as `iota`, `|B|`, and `L_grad(B)` using QSC and DESC.
- The loss is calculated as the difference between these computed quantities and their respective target values.
- If the DESC solver fails to converge, the loss is set to a high value to penalize the model for those configurations.

### 4. `train.py`
This script is responsible for the training process. It:
- Loads the preprocessed data.
- Defines the model, loss function, and optimizer.
- Runs the training loop with early stopping. The best model is saved if the validation loss improves.
- Outputs the train and validation loss at each epoch and plots these values for visualization.

# Current Status

- **Incomplete Functionality**: The project is still under development and does not yet run to completion. A significant issue is that the DESC solver often fails to converge for certain configurations, resulting in large residual errors during optimization. 


# Requirements

To run this project, you need the following dependencies:

- Python 3.8+
- PyTorch 1.9+
- QSC: Install from the [QSC GitHub repository](https://landreman.github.io/pyQSC/)
- DESC: Install from the [DESC GitHub repository](https://desc-docs.readthedocs.io/)
- JAX: Ensure that you have JAX with the CPU backend installed, as this project is currently using JAX for DESC calculations.
  
You can install the required Python packages using the following commands:

```bash
pip install torch jax jaxlib numpy pandas matplotlib scikit-learn
