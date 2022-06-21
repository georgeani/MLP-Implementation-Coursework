# MLP-Implementation-Coursework

This is the code for a simple MLP implementation. The implementation was part of the coursework for the Artificial Intelligence module in the second year. It is a simple MLP regression model. This model uses backpropagation and has somne improvements perfromed on it. These include, annealing, momentum, bold driver an their combinations.

It should be noted that due to the nature of the assignment the dataset used cannot be published as I do not hold the Intellectual Property rights. The dataset was pre-processed by removing outliers, non-numerical values and applying Min-Max scaling and standardisation was performed. 

Tanh and Sigmoid activation functions have been implemented. Mini-batch was not implemented in this code.

## Code structure and use of the modules

### csv_reader.py

This file is used to read and load the CSV files that will be used for training, testing and validation. The file had two different variations, one with reduced size which was only the files that had the highest correlation and one with the the entire number of features. The file has been created to only be able to read the CSV files derived from the original dataset.

### backpropagation_algorithm.py

Main backpropagation implementation. The algorithm takes care of the forward pass and the backward pass. It also includes the activation functions and their derivatives. The output is also destandardised to allow for easier comparison with the dataset. The weights in this case are randomly generated.

### backpropagation_annealing.py

Same implementation with backpropagation_algorithm.py, the main difference is that annealing is applied.

### backpropagation_annealing_momentum.py

Same implementation with backpropagation_algorithm.py, but annealing and momentum have been applied.

### backpropagation_momentum.py

Same implementation with backpropagation_algorithm.py, but momentum has been applied.

### backpropagation_bold_driver.py

Same implementation with backpropagation_algorithm.py, but bold driver has been applied.

### backpropagation_momentum_bold_driver.py

Same implementation with backpropagation_algorithm.py, but momentum and bold driver has been applied.

## Running the code

To execute the code, the user should run the script for each implementation of the backpropagation algorithm.

Additionally, the code should be modified in the main function to allow for diferent conigurations to be tested.
