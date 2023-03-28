import torch
import torch.nn as nn
import torch.optim as optim


# Define a shared encoder network
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x


# Define separate decoder networks for each task
class Task1Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Task1Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class Task2Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Task2Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


# Define a multi-task model that shares some features
class MultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes):
        super(MultiTaskModel, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.task1_decoder = Task1Decoder(hidden_size, output_sizes[0])
        self.task2_decoder = Task2Decoder(hidden_size, output_sizes[1])

    def forward(self, x):
        shared_features = self.encoder(x)
        task1_output = self.task1_decoder(shared_features)
        task2_output = self.task2_decoder(shared_features)
        return [task1_output, task2_output]


# Create a multi-task dataset
input_data = torch.randn(100, 10, dtype=torch.float)
task1_labels = torch.randint(0, 2, (100, 2), dtype=torch.float)
task2_labels = torch.randint(0, 2, (100, 1), dtype=torch.float)

# Define a multi-task loss function
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()

# Define a multi-task optimizer
model = MultiTaskModel(10, 5, [2, 1])
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model on both tasks
for epoch in range(100):
    for batch in range(32):
        optimizer.zero_grad()
        task1_output, task2_output = model(input_data)
        loss1 = criterion1(task1_output, task1_labels)
        loss2 = criterion2(task2_output, task2_labels)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

    print("Epoch {}: loss1={}, loss2={}, total_loss={}".format(epoch+1, loss1.item(), loss2.item(), loss.item()))
