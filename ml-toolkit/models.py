import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18
class MinimalCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc = nn.Linear(32 * 7500, num_classes)
    
    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 32 * 7500)  # Flatten the tensor
        x = self.fc(x)
        return x
    

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=2, num_classes=8):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.squeeze(x)
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Get the output from the last time step
        last_time_step = lstm_out[:, -1, :]

        # Pass the output through the fully connected layer
        out = self.fc(last_time_step)
        return out

    



def sleep_resnet18(input_channels=1, num_classes=8):
    model = resnet18(pretrained=True)
    # Adjust the first convolutional layer to accept single-channel input
    model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Modify the fully connected layer to match the number of output classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# Load the pretrained ResNet model



if __name__ == '__main__':
    # Model parameters
    input_size = 6
    hidden_size = 128
    num_layers = 2
    num_classes = 8

    # Initialize the LSTM model
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

    # Test the model with random input data
    batch_size = 64
    sequence_length = 30000
    input_data = torch.randn(batch_size, sequence_length, input_size)
    output_data = lstm_model(input_data)
    print(output_data.shape)  # Expected output shape: (batch_size, num_classes)
