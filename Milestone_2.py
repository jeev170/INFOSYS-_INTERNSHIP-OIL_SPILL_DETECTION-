import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Milestone 2: CNN Model Training and Evaluation for Oil Spill Detection

def load_preprocessed_data():
    """
    Load train, validation, and test datasets saved in .npy format from Milestone 1.
    Assumes data are numpy arrays of images normalized to [0,1].
    """
    train_imgs = np.load('train_images.npy')
    val_imgs = np.load('val_images.npy')
    test_imgs = np.load('test_images.npy')

    # For demonstration, generate dummy labels (replace with your actual labels)
    # Here, assuming binary classification 'oil spill' vs 'no spill'
    train_labels = np.random.randint(2, size=len(train_imgs))
    val_labels = np.random.randint(2, size=len(val_imgs))
    test_labels = np.random.randint(2, size=len(test_imgs))

    # Add channel dimension for CNN: (samples, height, width) -> (samples, height, width, 1)
    train_imgs = train_imgs[..., np.newaxis]
    val_imgs = val_imgs[..., np.newaxis]
    test_imgs = test_imgs[..., np.newaxis]

    return train_imgs, val_imgs, test_imgs, train_labels, val_labels, test_labels

def build_cnn_model(input_shape):
    """
    Build a simple CNN model for image classification.
    Extra feature: Added Dropout layers and Batch normalization could be added here for better generalization.
    """
    model = Sequential()

    # Conv block 1
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))

    # Conv block 2
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # Conv block 3
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Extra feature: Dropout to reduce overfitting
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history):
    """Plot training and validation accuracy and loss over epochs."""
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

def main():
    # Load data saved from milestone 1
    train_imgs, val_imgs, test_imgs, train_labels, val_labels, test_labels = load_preprocessed_data()
    print(f"Training samples: {len(train_imgs)}")
    print(f"Validation samples: {len(val_imgs)}")
    print(f"Testing samples: {len(test_imgs)}")

    # Build model
    input_shape = train_imgs[0].shape
    model = build_cnn_model(input_shape)
    model.summary()

    # Callbacks for early stopping and best model saving (extra feature)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    # Train model
    history = model.fit(train_imgs, train_labels,
                        validation_data=(val_imgs, val_labels),
                        epochs=50,
                        batch_size=32,
                        callbacks=[early_stop, checkpoint])

    # Plot training curves
    plot_training_history(history)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_imgs, test_labels)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Predictions and classification report (extra feature)
    y_pred_prob = model.predict(test_imgs)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(test_labels, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, y_pred))

if __name__ == "__main__":
    main()
