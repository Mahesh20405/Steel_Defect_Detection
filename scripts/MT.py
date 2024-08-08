import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from PP import train_generator, validation_generator
import os

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Assuming 4 classes for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',          # Monitor validation accuracy
    patience=5,                      # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True,       # Restore model weights from the epoch with the best value of the monitored quantity
    mode='max',                      # Mode 'max' to maximize validation accuracy
    verbose=1                        # Print messages when stopping
)

# Train the model with early stopping
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=16,
    callbacks=[early_stopping]  # Include the early stopping callback
)

# Save the trained model
model.save("E:/PJ/AI VT11/model1.keras")

# Model summary
mod_sum = []
mod_sum.append("Model Summary:\n")
mod_sum.append(str(model.summary()))

# Training history
mod_sum.append("\nTraining History:\n")
for epoch in range(len(history.history['accuracy'])):
    mod_sum.append(f"Epoch {epoch+1}: "
                         f"Loss: {history.history['loss'][epoch]:.4f}, "
                         f"Accuracy: {history.history['accuracy'][epoch]:.4f}, "
                         f"Validation Loss: {history.history['val_loss'][epoch]:.4f}, "
                         f"Validation Accuracy: {history.history['val_accuracy'][epoch]:.4f}\n")

# Save the summary to a text file
sum_path = "E:/PJ/AI VT11/mod_sum.txt"
with open(summary_path, 'w') as file:
    file.writelines(mod_sum)

print(f"Mod_sum saved to {summary_path}")
