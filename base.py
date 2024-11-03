import numpy
import torch
import torchvision
import lightly

# @(stackoverflow/46572475) sklearn does not automatically import its subpackages.
import sklearn.neighbors
import sklearn.metrics

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for img, lbl in dataloader:
            output = model(img)
            features.append(output.numpy())
            labels.extend(lbl.numpy())
    features = numpy.concatenate(features)
    return features, numpy.array(labels)

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                    LOAD TRAIN DATA                                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

dataset_train_images = []
dataset_train_labels = []

if torch.cuda.is_available():
    print(f'Cuda is enabled')
else:
    print(f'Cuda is not available')

for i in range(1, 6):
    dataset_batch = unpickle(f"cifar-10/data_batch_{i}")
    dataset_train_images.extend(dataset_batch[b'data'])
    dataset_train_labels.extend(dataset_batch[b'labels'])
dataset_train_images = numpy.concatenate(dataset_train_images)
dataset_train_images = dataset_train_images.reshape(-1, 3, 32, 32)

# After all the data are loaded for the training, we need to convert them to Tensors.
dataset_train_images_tensor = torch.tensor(dataset_train_images, dtype=torch.float32) / 255.0
dataset_train_labels_tensor = torch.tensor(dataset_train_labels, dtype=torch.long)

dataset_train = torch.utils.data.TensorDataset(dataset_train_images_tensor, dataset_train_labels_tensor)
dataset_loader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

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
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                  TRAIN CN-NETWORK                                                       | #
# --------------------------------------------------------------------------------------------------------------------------- #

print(f'Train')

epochs = 16
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for index, (images, labels) in enumerate(dataset_loader):
        optimizer.zero_grad()
        
        # We feed the network with the image data but we flatten them fist!
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (index + 1) % 128 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{index+1}/{len(dataset_loader)}], Loss: {loss.item():.4f}')
    scheduler.step()
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(dataset_loader):.4f}')

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                        BUILD K-NN (k=1, k=3) AND NEAREST CENTROID                                       | #
# --------------------------------------------------------------------------------------------------------------------------- #

# After training, modify the model to remove the final classification layer
model_ex = torch.nn.Sequential(
    *list(model.children())[:-1]
)

# Since we want to compare use the same train features for all three of them.
features, labels = extract(model_ex, dataset_loader)

knn1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
knn1.fit(features, labels);
knn3 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(features, labels);
centroid = sklearn.neighbors.NearestCentroid()
centroid.fit(features, labels);

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                    LOAD TEST DATA                                                       | #
# --------------------------------------------------------------------------------------------------------------------------- #

# Not a huge fun of the way the following code is written, 
# the fact that it is extremely similar to the above one, 
# means that I won't have to explain most of it again!

dataset_test_images = []
dataset_test_labels = []

for i in range(1, 2):
    dataset_batch = unpickle(f"cifar-10/test_batch_{i}")
    dataset_test_images.extend(dataset_batch[b'data'])
    dataset_test_labels.extend(dataset_batch[b'labels'])
dataset_test_images = numpy.concatenate(dataset_test_images)
dataset_test_images = dataset_test_images.reshape(-1, 3, 32, 32)

dataset_test_images_tensor = torch.tensor(dataset_test_images, dtype=torch.float32) / 255.0
dataset_test_labels_tensor = torch.tensor(dataset_test_labels, dtype=torch.long)

dataset_test = torch.utils.data.TensorDataset(dataset_test_images_tensor, dataset_test_labels_tensor)
dataset_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, shuffle=False)

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                                   EVALUATE NETWORK                                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

print(f'Evaluate')

epochs = 1
for epoch in range(epochs):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    for index, (images, labels) in enumerate(dataset_loader):
        optimizer.zero_grad()
        
        # We feed the network with the image data but we flatten them fist!
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # We’re ignoring the maximum scores because we only care about the predicted class indices.
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # Calculate average loss and accuracy
    accuracy = 100 * correct / total

    print(f'Test [{epoch+1}/{epochs}], Loss: {running_loss / len(dataset_loader):.4f}, Accuracy: {accuracy:.2f}%')

# --------------------------------------------------------------------------------------------------------------------------- #
# |                                       EAVLUTE K-NN (k=1, k=3) AND NEAREST CENTROID                                      | #
# --------------------------------------------------------------------------------------------------------------------------- #

# Since we want to compare use the same test features for all three of them.
features, labels = extract(model_ex, dataset_loader)

knn1_preds = knn1.predict(features)
knn3_preds = knn3.predict(features)
centroid_preds = centroid.predict(features)

knn1_accuracy = 100 * sklearn.metrics.accuracy_score(labels, knn1_preds)
knn3_accuracy = 100 * sklearn.metrics.accuracy_score(labels, knn3_preds)
centroid_accuracy = 100 * sklearn.metrics.accuracy_score(labels, centroid_preds)

print(f'k-NN (k=1) Accuracy: {knn1_accuracy:.4f}%')
print(f'k-NN (k=3) Accuracy: {knn3_accuracy:.4f}%')
print(f'Nearest Centroid Accuracy: {centroid_accuracy:.4f}%')
