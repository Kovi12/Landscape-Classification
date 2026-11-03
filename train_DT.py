import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import joblib
import time


DATA_DIR = 'data/seg_train/'
MODEL_PATH = 'models/landscape_classifier_DT.joblib'

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                img = Image.open(img_path).resize((150, 150))
                img = np.array(img).flatten()
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

print("Loading data...")
X, y = load_data(DATA_DIR)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

joblib.dump(label_encoder, 'label_encoder.joblib')
print("Label encoder saved.")


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")


clf = DecisionTreeClassifier(max_depth=10, random_state=42)
start_time = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

joblib.dump(clf, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

with open("training_time.txt", "w") as f:
    f.write(f"Decision Tree Training Time: {training_time:.2f} seconds\n")

print("Training script completed.")
