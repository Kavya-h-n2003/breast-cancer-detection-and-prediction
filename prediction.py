from tkinter import filedialog
from tkinter import Tk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import h5py
# Define CNN Model (Make sure to load the saved PyTorch model here)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 11 * 17, 64)  # Adjust dimensions based on input image size
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = CNNModel(num_classes=2)
model.load_state_dict(torch.load('350230finalmodel91.h5'))  # Load PyTorch model
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Infinite loop to process images
while True:
    root = Tk()
    root.withdraw()  # Hide tkinter root window
    path = filedialog.askopenfilename(filetypes=(("png", "*.png"), ("All files", "*.*")))
    if not path:
        print("No file selected. Exiting...")
        break

    # Read and preprocess the image
    img = cv2.imread(path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    test_image = cv2.resize(img, (140, 92))  # Resize to match input size
    test_image = transform(test_image)      # Apply transformations
    test_image = test_image.unsqueeze(0)    # Add batch dimension

    # Predict using the model
    with torch.no_grad():
        output = model(test_image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Display results
    if predicted_class == 0:
        s = f"BENIGN with Accuracy: {probabilities[0][0] * 100:.2f}%\n"
    else:
        s = f"MALIGNANT with Accuracy: {probabilities[0][1] * 100:.2f}%\n"

    print(s)
