# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

## Neural Network Model

<img width="962" height="468" alt="image" src="https://github.com/user-attachments/assets/56863678-71a2-42de-a4bd-e1cc1e709c83" />


## DESIGN STEPS

STEP 1:
Load Fashion-MNIST dataset from torchvision, apply transformations, and create DataLoaders for batch processing

STEP 2:
Build CNN architecture with 3 convolutional layers (32,64,128 filters) and 3 fully connected layers (128,64,10 nodes)

STEP 3:
Train model using CrossEntropyLoss and Adam optimizer while tracking training and validation loss metrics

STEP 4:
Evaluate model performance using confusion matrix, classification report, and test on new handwritten images

STEP 5:
Visualize results with loss plots and display predictions with actual vs predicted labels
## PROGRAM

### Name:SANTHABABU G
### Register Number:212224040292
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.ToTensor()

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nName: SANTHABABU  G")
print("Register Number: 212224040292")
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

accuracy = accuracy_score(all_labels, all_preds)
print("accuracy                           {:.2f}     {}".format(accuracy, len(all_labels)))

index = 0
image, label = test_dataset[index]

with torch.no_grad():
    output = model(image.unsqueeze(0).to(device))
    _, pred = torch.max(output, 1)

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[pred.item()]}")
plt.axis('off')
plt.show()




```

## OUTPUT
### Training Loss per Epoch

<img width="516" height="158" alt="image" src="https://github.com/user-attachments/assets/0b2eb65b-394a-4406-b14c-3799d006c61d" />


### Confusion Matrix

<img width="819" height="707" alt="image" src="https://github.com/user-attachments/assets/a2df29ad-ea39-44c1-b317-ed9d71cec311" />


### Classification Report

<img width="672" height="519" alt="image" src="https://github.com/user-attachments/assets/acba76da-589c-4ec6-91ca-458339a2c0e9" />



### New Sample Data Prediction

<img width="546" height="552" alt="image" src="https://github.com/user-attachments/assets/7c75793d-9c1b-4f66-9d43-bea2b0e0bbab" />


## RESULT
The Convolutional Neural Network (CNN) was successfully implemented for image classification. The model was trained on the dataset, and its performance was evaluated using accuracy metrics, confusion matrix, and classification report. Predictions on new sample images were verified, confirming the model's effectiveness in classifying images.
