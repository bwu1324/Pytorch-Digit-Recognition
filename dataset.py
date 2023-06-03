import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets
from torchvision.transforms import Resize, CenterCrop, Lambda

# Loads and transforms dataset to correct format for training


def one_hot_func(y):
    '''
    Convert target to one-hot encoded tensor

    y - number [0, 9]
    returns - (,10) one-hot encoded tensor
    '''
    return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


def crop_in(x):
    '''
    Remove black borders around image

    x - 28x28 imagelike
    returns - (1, 28, 28) tensor
    '''
    # Convert to numpy array
    x = np.array(x, dtype='float32').reshape((28, 28))
    
    # Remove border
    mask = x > 0
    x = x[np.ix_(mask.any(1), mask.any(0))]

    # Make back into square
    x = torch.from_numpy(x)
    x = CenterCrop(max(x.shape))(x)

    # Resize to have correct number of dimensions
    x = x.reshape((1, max(x.shape), max(x.shape)))

    # Make back into 28x28 square
    x = Resize(28, antialias=True)(x)
    return x


# Load training data
training_data = ConcatDataset([
    datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=Lambda(crop_in),
        target_transform=Lambda(one_hot_func)
    )
])

# Load test data
test_data = ConcatDataset([
    datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=Lambda(crop_in)
    )
])


if __name__ == '__main__':
    print(test_data)
    # Visualize dataset
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    def show(data_loader):
        images, foo = next(iter(data_loader))

        npimg = make_grid(images, normalize=True, pad_value=.5).numpy()
        fig, ax = plt.subplots(figsize=((13, 5)))
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.setp(ax, xticks=[], yticks=[])
        return fig, ax

    data_loader = torch.utils.data.DataLoader(training_data,
                                              batch_size=8,
                                              shuffle=True)
    fig, ax = show(data_loader)
    plt.show()
