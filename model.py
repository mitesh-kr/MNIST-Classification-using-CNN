from torch import nn


# Define CNN Model class
class CNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (7,7), stride = (1,1), padding = (3,3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = (5,5), stride = (1,1), padding = (2,2)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 8, out_channels = 4, kernel_size = (3,3), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = (2,2), stride = (1,1))
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 3 * 3 * 4, out_features = output_shape),
        )
# Forward function
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output_layer(x)
        return x

