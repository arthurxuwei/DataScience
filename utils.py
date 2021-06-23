import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

#获取和读取数据
def load_data(batch_size,num_workers):
    mnist_train = torchvision.datasets.FashionMNIST(root='/home/arthur.xw/dl/Datasets/FashionMNIST',
                                                    train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='/home/arthur.xw/dl/Datasets/FashionMNIST',
                                                train=False, download=True,
                                                transform=transforms.ToTensor())

    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_iter,test_iter


def imshow_batch(sample_batch):
    images = sample_batch[0]
    labels = sample_batch[1]
    images = make_grid(images, nrow=4, pad_value=255)
    # 1,2, 0 
    images_transformed = np.transpose(images.numpy(), (1, 2, 0))
    plt.imshow(images_transformed)
    plt.axis('off')
    labels = labels.numpy()
    plt.title(labels)