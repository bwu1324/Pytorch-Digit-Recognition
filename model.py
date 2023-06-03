from torch import nn

# Define Network
class DigitRecognition(nn.Module):
    def __init__(self, DROPOUT_RATE):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout2d(DROPOUT_RATE),
        )

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x
