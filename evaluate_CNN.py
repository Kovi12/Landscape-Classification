import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import random as rd

# Define directories
VAL_DIR = 'data/seg_test/'
PRED_DIR = 'data/seg_pred'

# Output directory for evaluation results
OUTPUT_DIR = 'evaluation_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data generator
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(1.0 / 255)

# Validation data generator
val_gen = data_gen.flow_from_directory(
    VAL_DIR,
    target_size=(150, 150),
    class_mode="categorical",
    shuffle=False
)

# Load the saved model
model = load_model('models/landscape_classifier.h5')
print("Model loaded successfully.")

# Evaluate the model on validation data
test_loss, test_acc = model.evaluate(val_gen, steps=len(val_gen))
with open(os.path.join(OUTPUT_DIR, 'test_metrics.txt'), 'w') as f:
    f.write(f'Test Loss: {test_loss:.4f}\n')
    f.write(f'Test Accuracy: {test_acc:.4f}\n')

# Read and preprocess prediction images
pred_images = []
for filename in os.listdir(PRED_DIR):
    img_path = os.path.join(PRED_DIR, filename)
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img = np.array(img)
    pred_images.append(img)

pred_images = np.array(pred_images)

# Generate predictions
predictions = model.predict(pred_images)

# Generate validation predictions for confusion matrix
Y_pred = model.predict(val_gen, steps=len(val_gen))
y_pred = np.argmax(Y_pred, axis=1)

# Accuracy Score
accuracy = accuracy_score(val_gen.classes, y_pred)
print("Accuracy Score:", accuracy)
with open(os.path.join(OUTPUT_DIR, 'test_metrics.txt'), 'a') as f:
    f.write(f'Accuracy Score: {accuracy:.4f}\n')

# Confusion Matrix
cm = confusion_matrix(val_gen.classes, y_pred)
target_names = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
plt.close()

# Classification Report
class_report = classification_report(val_gen.classes, y_pred, target_names=target_names)
print('Classification Report')
print(class_report)
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(class_report)

# Display and save random predictions from `PRED_DIR`
class_labels = val_gen.class_indices
reverse_labels = {v: k for k, v in class_labels.items()}

for i in range(10):  # Save 10 random predictions
    rand = rd.randint(0, len(pred_images) - 1)
    predicted_class = reverse_labels[predictions[rand].argmax()]
    plt.figure(figsize=(6, 6))
    plt.imshow(pred_images[rand])
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, f'prediction_{i + 1}.png'))
    plt.close()

print(f"Evaluation results saved in: {OUTPUT_DIR}")
