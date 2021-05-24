# Learning Physical Constraints with Neural Projections

## Dependencies
All the dependencies are standard python packages:
- Pytorch, Numpy, Matplotlib, PIL (used for visualization) 

## Code Structures:
- \_\*.py: Some common classes/functions used for training and testing.
    + \_constraint\_net.py: A simple network used to represent the constraints to be learned
    + \_iterative\_proj.py: The iterative projection to solve the constrains
    + \_training.py: Utils used for training
    + \_dataloader.py: Load training data
    + \_run_simulation.py: Run simulation use the learned constraint net and the projection operator
- training\_\*.py: Train the model.
    + Data used for training can be found in this [shared google drive folder](https://drive.google.com/drive/folders/15u6nJte4k7xnIbL_emDVLyFkiQxQ7UZn?usp=sharing).
    + The trained models are in \model folder.
- simulation\_\*.py: Use the trained model to generate simulations.
    + Simulation results will be written to \results folder.
    + In this branch, simulation scripts are modified to run multiple simulation samples

## Webpage:
[https://y-sq.github.io/proj/neural_proj/](https://y-sq.github.io/proj/neural_proj/)

