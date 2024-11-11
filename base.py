import numpy
import torch
import torchvision
import lightly

# @(stackoverflow/46572475) sklearn does not automatically import its subpackages.
import sklearn.neighbors
import sklearn.metrics


import lightly.data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                     MAKE TRANSFORM                                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    torchvision.transforms.ToTensor()
])

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                    LOAD TRAIN DATA                                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

dataset_train = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
)
dataset_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=64, shuffle=True
)

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                    MAKE CN-NETWORK                                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

# Define the CNN model using nn.Sequential
model = torch.nn.Sequential(
    # First convolutional layer
    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    
    # Second convolutional layer
    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    
    # Flatten the output from the convolutional layers
    torch.nn.Flatten(),
    
    # Fully connected layer
    torch.nn.Linear(32 * 8 * 8, 64),
    torch.nn.ReLU(),
    
    # Fully connected layer
    torch.nn.Linear(64, 64),
    torch.nn.Sigmoid(),
    
    # Output layer
    torch.nn.Linear(64, 10)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                  TRAIN CN-NETWORK                                                       | #
# --------------------------------------------------------------------------------------------------------------------------- #

print(f"Train [Device: {device}]")

epochs = 4
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for index, batch in enumerate(dataset_loader):
        (inputs, labels, _) = batch
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (index + 1) % 128 == 0:
            print(f"Batch [{index+1}/{len(dataset_loader)}], Size: {len(inputs)}, Loss: {loss.item():.4f}")
    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataset_loader):.4f}")

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                    LOAD TEST DATA                                                       | #
# --------------------------------------------------------------------------------------------------------------------------- #

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

dataset_train = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
)
dataset_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=64, shuffle=False
)

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                   EVALUATE NETWORK                                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

print(f'Evaluate')

model.eval()
correct = 0
total = 0
running_loss = 0.0
for index, batch in enumerate(dataset_loader):
    (inputs, labels, _) = batch
    
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    running_loss += loss.item()
    predicted = outputs.argmax(dim=1)
    correct += (predicted == labels).sum().item()
    total += labels.size(0)
    if (index + 1) % 128 == 0:
        print(f"Batch [{index+1}/{len(dataset_loader)}], Size: {len(inputs)}, Loss: {loss.item():.4f}")
print(f'Test, Loss: {running_loss / len(dataset_loader):.4f}, Accuracy: {(100 * correct / total):.2f}%')

