import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.applications.efficientnet import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def load_gender_data(data_dir, img_size=(128, 128)):
    """Load and preprocess gender dataset"""
    images, labels = [], []
    class_names = ['male', 'female']
    num_classes = len(class_names)  # Get number of classes
    
    for label, gender in enumerate(class_names):
        folder_path = os.path.join(data_dir, gender)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
            
        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
                
    if not images:
        raise ValueError("No valid images found in the dataset directory")
        
    return np.array(images), to_categorical(np.array(labels), num_classes), class_names

def build_model(input_shape=(128, 128, 3)):
    """Build EfficientNet-based gender classifier"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base model layers
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(2, activation='softmax')  # 2 output units for binary classification
    ])
    
    return model

def train_gender_classifier(train_dir, val_dir, epochs=15, batch_size=32):
    """Train gender classification model"""
    # Load data
    train_images, train_labels, class_names = load_gender_data(train_dir)
    val_images, val_labels, _ = load_gender_data(val_dir)
    
    # Preprocess data
    train_images = preprocess_input(train_images)
    val_images = preprocess_input(val_images)
    
    # Build model
    model = build_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'gender_classification_efficientnet_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('gender_classification_efficientnet_final.keras')
    print("Training completed. Model saved as 'gender_classification_efficientnet_final.keras'.")
    
    # Plot training history
    plot_training_history(history)

    return model, history

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == '__main__':
    train_dir = r'E:\comsys hackathon\Comys_Hackathon5\Task_A\train'
    val_dir = r'E:\comsys hackathon\Comys_Hackathon5\Task_A\val'
    train_gender_classifier(train_dir, val_dir)


    