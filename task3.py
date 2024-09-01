import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Load and preprocess the images
def load_images(folder, use_hog=False):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize for consistency
            if use_hog:
                img = rgb2gray(img)  # Convert to grayscale
                fd = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                img = fd  # Use the HOG descriptor as the feature
            else:
                img = img.flatten()  # Flatten the image to a 1D array if not using HOG
            images.append(img)
            if 'cat' in filename:
                labels.append(0)
            elif 'dog' in filename:
                labels.append(1)
    return np.array(images), np.array(labels)


# Load training data
train_folder = 'train'
X, y = load_images(train_folder, use_hog=True)

# Dimensionality reduction using PCA
pca = PCA(n_components=100)
X = pca.fit_transform(X)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM with hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm = GridSearchCV(SVC(), param_grid, cv=5, verbose=1, n_jobs=-1)
svm.fit(X_train, y_train)

# Validate the model
y_pred = svm.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")

# Load and classify test images
test_folder = 'test1'
test_images, test_filenames = [], []

for filename in os.listdir(test_folder):
    img = cv2.imread(os.path.join(test_folder, filename))
    if img is not None:
        img = cv2.resize(img, (64, 64))
        img = rgb2gray(img)  # Convert to grayscale
        fd = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        img = fd  # Use the HOG descriptor as the feature
        img = img.flatten()  # Flatten the image to a 1D array
        test_images.append(img)
        test_filenames.append(os.path.splitext(filename)[0])

test_images = np.array(test_images)
test_images = pca.transform(test_images)  # Apply PCA transformation
test_preds = svm.predict(test_images)

# Save the results to a CSV file
results = pd.DataFrame({
    'id': test_filenames,
    'label': test_preds
})
results.to_csv('submission.csv', index=False)

print("Classification complete. Results saved to submission.csv.")

# Display a portion of the test results
for i in range(5):
    img = cv2.imread(os.path.join(test_folder, test_filenames[i] + '.jpg'))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {'Dog' if test_preds[i] == 1 else 'Cat'}")
    plt.show()
