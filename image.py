import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

dataset_dir = "C:/Users/LEELA NARESH/OneDrive/Desktop/imagenet1k"
train_dir = "train"
val_dir = "validation"
test_dir = "test"

def split_data(dataset_dir, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.2, min_samples=1):
    class_folders = os.listdir(dataset_dir)
    for class_folder in class_folders:
        class_path = os.path.join(dataset_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        if len(files) < min_samples:
            print(f"Skipping {class_folder} due to insufficient samples.")
            continue
        
        train_files, test_val_files = train_test_split(files, test_size=test_size+val_size, random_state=42)
        val_files, test_files = train_test_split(test_val_files, test_size=test_size/(test_size+val_size), random_state=42)
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)
        for file in train_files:
            src = os.path.join(class_path, file)
            dest = os.path.join(train_dir, class_folder, file)
            shutil.copy(src, dest)
        for file in val_files:
            src = os.path.join(class_path, file)
            dest = os.path.join(val_dir, class_folder, file)
            shutil.copy(src, dest)
        for file in test_files:
            src = os.path.join(class_path, file)
            dest = os.path.join(test_dir, class_folder, file)
            shutil.copy(src, dest)

split_data(dataset_dir, train_dir, val_dir, test_dir, min_samples=5)

img_size = 224
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

resnet_model = models.resnet50(weights='imagenet')  

for param in resnet_model.parameters():
    param.requires_grad = False

num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet_model.fc.parameters(), lr=0.001, momentum=0.9)


resnet_model.train()
for epoch in range(5): 
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = resnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss}")

torch.save(resnet_model.state_dict(), "resnet_model.pth")
print("Model trained and saved successfully!")

resnet_model = models.resnet50()  
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(train_dataset.classes))
resnet_model.load_state_dict(torch.load("resnet_model.pth"))
resnet_model.eval()

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = resnet_model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")

image_path = "path_to_your_new_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
image = cv2.resize(image, (img_size, img_size))  
image = image.astype(np.float32) / 255.0  
image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] 
image = np.transpose(image, (2, 0, 1))  
image = torch.from_numpy(image).unsqueeze(0)  

with torch.no_grad():
    output = resnet_model(image)
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()

class_label = train_dataset.classes[class_index]
print("Predicted class:", class_label)
