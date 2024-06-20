import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

train_dir = '/Users/ashis/Desktop/spyder/training_set/training_set'  
test_dir = '/Users/ashis/Desktop/spyder/test_set/test_set'   

IMG_SIZE = (64, 64)

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

def load_images(directory, img_size):
    images = []
    labels = []
    for category in ['cats', 'dogs']:
        class_num = 0 if category == 'cats' else 1
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                if not os.path.isfile(img_path) or img.startswith('.'):
                    continue
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, img_size)
                images.append(resized_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    return np.array(images), np.array(labels)

def extract_hog_features(images):
    features = []
    for image in images:
        hog_feature = hog(image,
                          orientations=HOG_ORIENTATIONS,
                          pixels_per_cell=HOG_PIXELS_PER_CELL,
                          cells_per_block=HOG_CELLS_PER_BLOCK,
                          block_norm='L2-Hys')
        features.append(hog_feature)
    return np.array(features)

X, y = load_images(train_dir, IMG_SIZE)

X_hog = extract_hog_features(X)

X_train, X_val, y_train, y_val = train_test_split(X_hog, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
