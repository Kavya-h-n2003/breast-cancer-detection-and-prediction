from tkinter import *
from tkinter import filedialog
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Global variables
model = None
path = "img.png"
window = tk.Tk()
window.title("Graphical Interface")
wd = str(window.winfo_screenwidth() - 260) + "x" + str(window.winfo_screenheight() - 200)
window.geometry(wd)

folder_path = './data/'
images = []
labels = []
class_label = 0

# Load images and resize
def load_images_from_folder(folder, class_label):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (140, 92))
            img = img.flatten()  # Flatten the image for ML models
            images.append(img)
            labels.append(class_label)
    return class_label + 1

class_label = 0
class_label = load_images_from_folder(folder_path + 'benign', class_label)
class_label = load_images_from_folder(folder_path + 'malignant', class_label)

Data = np.asarray(images)
Labels = np.asarray(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(Data, Labels, test_size=0.2, random_state=2)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)  # Reduce features to 100 dimensions
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train SVM
def train_model(X_train, y_train):
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return acc

def classify_image(model, path, scaler, pca):
    img = cv2.imread(path)
    img = cv2.resize(img, (140, 92))
    img = img.flatten()  # Flatten image
    img = scaler.transform([img])  # Scale
    img_pca = pca.transform(img)  # PCA
    pred = model.predict(img_pca)
    proba = model.predict_proba(img_pca)
    return pred[0], proba[0]

# Train and Evaluate
model = train_model(X_train_pca, y_train)
evaluate_model(model, X_test_pca, y_test)

def random_image_test():
    path = filedialog.askopenfilename(filetypes=(("JPG", ".jpg"), ("All files", "*.*")))
    pred, proba = classify_image(model, path, scaler, pca)
    label = "BENIGN" if pred == 0 else "MALIGNANT"
    confidence = max(proba) * 100
    print(f"Prediction: {label} with Confidence: {confidence:.2f}%")
    img = cv2.imread(path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} ({confidence:.2f}%)")
    plt.show()

# Tkinter Buttons
labelfont = ('Arial', 40, 'bold')
label1 = tk.Label(text="   Breast Cancer Detection using SVM   ", anchor='n', font=labelfont, fg="midnight blue", bg="mint cream")
label1.grid(column=0, row=0)

button1 = tk.Button(text="Test Random Image", command=random_image_test, bg="powder blue")
button1.grid(column=0, row=1)

window.mainloop()
