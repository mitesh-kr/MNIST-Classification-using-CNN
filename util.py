import torch
import numpy as np 

# Extract image data from file, normalize and convert it in tensor with shape as (28,28).
def load_images(file_path, data_count):
    with open(file_path, 'rb') as data:
        data.read(16)
        data = data.read(28 * 28 * data_count)
    images = torch.tensor(np.frombuffer(data, dtype=np.uint8).astype(np.float32)/255)
    images = images.reshape(data_count, 28, 28)
    torch.manual_seed(19)
    idx = torch.randperm(data_count)
    images = images[idx]
    return images

# Extract label data from file,suffle it and convert it in tensor with shape as (60000).
def load_labels(file_path, data_count):
    with open(file_path, 'rb') as data:
        data.read(8)
        data = data.read(data_count)
    labels = torch.tensor(np.frombuffer(data, dtype=np.uint8))
    torch.manual_seed(19)
    idx = torch.randperm(data_count)
    labels = labels[idx]
    return labels

#Split train dataset into train and validation dataset
def train_validation_split(images, labels, train_ratio):
    n = int(train_ratio * images.shape[0])
    X_train = images[:n]
    X_validation = images[n:]
    y_train = labels[:n]
    y_validation = labels[n:]
    return X_train, X_validation, y_train, y_validation


# Number of parameters in model
def count_parameters(self):
    trainable_params = sum(i.numel() for i in self.parameters() if i.requires_grad)
    return trainable_params