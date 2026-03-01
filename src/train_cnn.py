import os
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import argparse

# --- ConfigurationDefaults ---
DEFAULT_DATASET_PATH = r"C:\Users\chand\Downloads\archive\kaggle_3m"
DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_model_v2.h5')

def load_data(dataset_path):
    print(f"DEBUG: Searching in {dataset_path}")
    if os.path.exists(dataset_path):
        print(f"DEBUG: Directory exists. Contents: {os.listdir(dataset_path)[:5]}")
    else:
        print("DEBUG: Directory DOES NOT EXIST!")
    
    # Find all mask files
    mask_files = glob.glob(os.path.join(dataset_path, '**/*_mask.tif'), recursive=True)
    print(f"DEBUG: Glob found {len(mask_files)} files.")
    
    image_paths = []
    labels = []
    
    for mask_path in mask_files:
        # Infer image path from mask path
        img_path = mask_path.replace('_mask.tif', '.tif')
        
        if os.path.exists(img_path):
            # Check if mask has any positive pixels (Tumor present)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                has_tumor = 1 if np.max(mask) > 0 else 0
                
                image_paths.append(img_path)
                labels.append(has_tumor)
    
    print(f"Found {len(image_paths)} images.")
    return image_paths, labels

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, dim=(128, 128), shuffle=True, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        self.augmentor = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_paths = [self.image_paths[k] for k in indexes]
        list_labels = [self.labels[k] for k in indexes]
        
        X, y = self.__data_generation(list_paths, list_labels)
        return X, y

    def __data_generation(self, list_paths, list_labels):
        X = np.empty((self.batch_size, *self.dim, 3))
        y = np.empty((self.batch_size), dtype=int)

        for i, path in enumerate(list_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image {path}")
                img = np.zeros((*self.dim, 3), dtype=np.float32)
            else:
                img = cv2.resize(img, self.dim)
                img = img / 255.0  # Normalize
            X[i,] = img
            y[i] = list_labels[i]
            
        if self.augment:
             # Apply augmentation to the batch
             # Note: ImageDataGenerator.flow expects (N, H, W, C)
             it = self.augmentor.flow(X, y, batch_size=self.batch_size, shuffle=False)
             X, y = next(it)

        return X, y

def build_model(input_shape):
    # Use EfficientNetB0 (Transfer Learning) for better accuracy
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Freeze base model initially
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for Glioma Segmentation")
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_PATH, help="Path to kaggle_3m dataset")
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Configuration: Dataset={args.dataset}, Epochs={args.epochs}, Batch={args.batch_size}")
    
    image_paths, labels = load_data(args.dataset)
    
    if len(image_paths) == 0:
        print("No images found! Check path.")
        return

    # Split data
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    
    # Generators
    training_generator = DataGenerator(X_train_paths, y_train, batch_size=args.batch_size, dim=DEFAULT_IMG_SIZE, augment=True)
    validation_generator = DataGenerator(X_val_paths, y_val, batch_size=args.batch_size, dim=DEFAULT_IMG_SIZE, augment=False)
    
    model = build_model((*DEFAULT_IMG_SIZE, 3))
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("Starting training...")
    try:
        history = model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=args.epochs,
            callbacks=[early_stop]
        )
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        model.save(MODEL_SAVE_PATH)
        
        # Save training history for accuracy curve visualization
        history_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'cnn_history.pkl')
        print(f"Saving training history to {history_path}...")
        import pickle
        with open(history_path, 'wb') as f:
            pickle.dump(history.history, f)
            
        print("Done!")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
