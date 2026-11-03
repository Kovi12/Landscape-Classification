import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import time

# Define directories
TRAIN_DIR = 'data/seg_train/'
VAL_DIR = 'data/seg_test/'

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Data generator
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# Train data generator
train_gen = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    class_mode="categorical"
)

# Validation data generator
val_gen = data_gen.flow_from_directory(
    VAL_DIR,
    target_size=(150, 150),
    class_mode="categorical"
)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.5),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()
# Train the model
model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen,
    verbose=1,
    callbacks=[early_stopping]
)
end_time = time.time()
total_time = end_time - start_time
print('Total time:', total_time)
# Save the model
# model.save('models/landscape_classifier.h5')
# print("Model saved as 'landscape_classifier.h5'")
