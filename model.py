import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Conventional and convolutional neural network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 8)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

def train_model(train_dataloader, test_dataloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    num_epochs=50

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):

            inputs, labels = data
            inputs=inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Reshape the input tensors to match the expected input size
            inputs = inputs.view(-1, 4, 128, 128) 

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {running_loss / 100}")
                running_loss = 0.0
        print(f"Epoch: {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

    net.eval()  # Set the model in evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs=inputs.to(device)
            labels = labels.to(device)
            # Reshape the input tensors to match the expected input size
            inputs = inputs.view(-1, 2, 201, 1200) 

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch: {epoch+1}, Validation Accuracy: {accuracy}")

    