import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
plt.ion()

# Load and normalize data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize and reshape data
train_images = train_images / 255.00
test_images = test_images / 255.00
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

# Split into training and validation sets
val_images = train_images[-12000:]
val_labels = train_labels[-12000:]
train_images = train_images[:-12000]
train_labels = train_labels[:-12000]

# Define model
model = Sequential([
    Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(56, (3, 3), activation='relu'),
    Flatten(),
    Dense(56, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Number of trainable parameters:", model.count_params())

# Train model
history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    batch_size=32,
    epochs=10
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_accuracy)

# Plot training and validation accuracy
epochs = range(1, len(history.history['accuracy']) + 1) 
plt.plot(epochs, history.history['accuracy'], label='Training accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(epochs)
plt.savefig('imgs/training_validation_accuracy.png') # Generate PNG picture of line plot

# Show misclassified examples
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
misclassified_indices = np.where(predicted_classes != test_labels)[0]

# Display an example for each class
classes = np.unique(test_labels)
examples = []

for class_label in classes:
    found = False
    for i in misclassified_indices:
        if test_labels[i] == class_label and i not in examples:
            examples.append(i)
            found = True
            break
    if not found:
        print(f"No misclassified examples for class {class_label}")

label_descriptions = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(10, 10))
for i, idx in enumerate(examples):
    if idx is not None:
        plt.subplot(5, 2, i + 1)
        plt.imshow(test_images[idx].squeeze(), cmap='gray')
        actual_label = label_descriptions[test_labels[idx]]
        predicted_label = label_descriptions[predicted_classes[idx]]
        plt.title(f"Actual: {actual_label}, Predicted: {predicted_label}")
        plt.axis('off')
plt.tight_layout()
plt.savefig('imgs/misclassified_examples.png') # Display misclassified examples (each image is labeled with both actual and predicted class labels)