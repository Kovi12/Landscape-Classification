import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

DATA_DIR = 'data/seg_test/'
MODEL_PATH = 'models/landscape_classifier_DT.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'
RESULTS_FILE = 'evaluation_results/evaluation_results_DT.txt'
OUTPUT_DIR = 'evaluation_results/DT'


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

# Load data
print("Loading data...")
X_test, y_test_labels = load_data(DATA_DIR)

label_encoder = joblib.load(LABEL_ENCODER_PATH)
y_test = label_encoder.transform(y_test_labels)

clf = joblib.load(MODEL_PATH)
print("Model and label encoder loaded successfully.")

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
target_names = label_encoder.inverse_transform(np.unique(y_test))
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_DT.png'))
plt.show()
plt.close()
classification_rep = classification_report(y_test, y_pred, target_names=target_names)
print("Classification Report:")
print(classification_rep)

with open(RESULTS_FILE, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(classification_rep)

print(f"Evaluation results saved to {RESULTS_FILE}")
