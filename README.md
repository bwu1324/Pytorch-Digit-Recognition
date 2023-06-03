# Pytorch Digit Recognition

This is a practice project to get familiar with using PyTorch

## Notes

* `dataset.py` loads the MNIST dataset and transforms it, run this alone to visualize the dataset
    * Note: The dataset is quite slow for training since tranformations are not cached (this can be improved)
* `model.py` defines the structure of the model
* `train.py` trains the dataset and saves the state dictionary every 10 epochs
* `convert_state_dict.py` uses the model defined in `model.py` and the selected state dictionary file and converts it to a TensorScript file
* `server.py` loads a selected TensorScript file and runs a webserver for trying out the model with your own writing