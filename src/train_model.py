"""
Model Training Pipeline — Audio Classification CNN
====================================================
Trains a 2D CNN on MFCC features extracted from the border
intrusion detection dataset (footstep / gunshot / noise).

Handles class imbalance via augmentation and class weighting.
Exports to both Keras H5 and TFLite formats.
"""

import os
import sys
import glob
import json
import logging
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

SAMPLE_RATE = 22050
DURATION = 1.0               # seconds
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
TARGET_LENGTH = 44            # MFCC time frames for 1s at 22050Hz

CLASS_MAP = {
    "footsteps": 0,
    "balastic": 1,    # gunshot
    "noise": 2,
}
CLASS_LABELS = ["footstep", "gunshot", "noise"]

BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Loading & Feature Extraction
# ──────────────────────────────────────────────

def load_audio(file_path: str) -> np.ndarray:
    """Load and normalize audio to fixed length."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            audio = audio[:target_len]
        return audio
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Extract normalized MFCC features."""
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    # Fix time dimension
    if mfcc.shape[1] < TARGET_LENGTH:
        mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_LENGTH - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :TARGET_LENGTH]
    return mfcc


def augment_audio(audio: np.ndarray) -> list:
    """
    Generate augmented versions of an audio clip.
    
    Augmentations:
    - Time shift
    - Pitch shift
    - Speed change
    - Add noise
    - Volume change
    """
    augmented = []
    
    # Time shift (±20%)
    shift = int(len(audio) * np.random.uniform(-0.2, 0.2))
    shifted = np.roll(audio, shift)
    augmented.append(shifted)
    
    # Pitch shift (±2 semitones)
    try:
        pitched = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=np.random.uniform(-2, 2))
        augmented.append(pitched)
    except Exception:
        pass
    
    # Speed change (0.8x–1.2x)
    try:
        speed_factor = np.random.uniform(0.85, 1.15)
        stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
        target_len = int(SAMPLE_RATE * DURATION)
        if len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)), mode='constant')
        else:
            stretched = stretched[:target_len]
        augmented.append(stretched)
    except Exception:
        pass
    
    # Add white noise
    noise = np.random.randn(len(audio)) * 0.005
    noisy = audio + noise
    augmented.append(noisy)
    
    # Volume change (±30%)
    gain = np.random.uniform(0.7, 1.3)
    louder = audio * gain
    augmented.append(louder)
    
    return augmented


def load_dataset(augment_minority: bool = True):
    """
    Load entire dataset, extract features, and balance classes.
    
    Returns:
        X: numpy array of shape (N, N_MFCC, TARGET_LENGTH, 1)
        y: numpy array of shape (N,) with integer labels
    """
    X_all, y_all = [], []
    class_counts = Counter()
    
    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Directory not found: {class_dir}")
            continue
        
        files = glob.glob(os.path.join(class_dir, "*.wav"))
        logger.info(f"Loading {class_name}: {len(files)} files")
        
        for fpath in files:
            audio = load_audio(fpath)
            if audio is None:
                continue
            
            mfcc = extract_mfcc(audio)
            X_all.append(mfcc)
            y_all.append(label)
            class_counts[class_name] += 1
        
        logger.info(f"  Loaded {class_counts[class_name]} samples for {class_name}")
    
    logger.info(f"Raw dataset: {dict(class_counts)}")
    
    # Augment minority classes to balance
    if augment_minority:
        max_count = max(class_counts.values())
        
        for class_name, label in CLASS_MAP.items():
            current = class_counts[class_name]
            needed = max_count - current
            
            if needed <= 0:
                continue
            
            logger.info(f"Augmenting {class_name}: {current} → ~{max_count} (need {needed} more)")
            
            class_dir = os.path.join(DATASET_DIR, class_name)
            files = glob.glob(os.path.join(class_dir, "*.wav"))
            
            generated = 0
            while generated < needed:
                fpath = files[generated % len(files)]
                audio = load_audio(fpath)
                if audio is None:
                    generated += 1
                    continue
                
                augmented_list = augment_audio(audio)
                for aug_audio in augmented_list:
                    if generated >= needed:
                        break
                    mfcc = extract_mfcc(aug_audio)
                    X_all.append(mfcc)
                    y_all.append(label)
                    generated += 1
            
            logger.info(f"  Generated {generated} augmented samples for {class_name}")
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    # Add channel dimension: (N, N_MFCC, TARGET_LENGTH) → (N, N_MFCC, TARGET_LENGTH, 1)
    X = X.reshape(-1, N_MFCC, TARGET_LENGTH, 1).astype(np.float32)
    
    logger.info(f"Final dataset: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {Counter(y)}")
    
    return X, y


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────

def build_model(input_shape: tuple, num_classes: int = 3):
    """
    Build 2D CNN for audio classification.
    
    Architecture:
    Conv2D(32) → BN → ReLU → MaxPool → Dropout
    Conv2D(64) → BN → ReLU → MaxPool → Dropout
    Conv2D(128) → BN → ReLU → MaxPool → Dropout
    GlobalAveragePooling2D
    Dense(128) → BN → ReLU → Dropout
    Dense(3, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        
        # Classifier
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax'),
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ──────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────

def train():
    """Full training pipeline."""
    import tensorflow as tf
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("BORDER INTRUSION DETECTION — MODEL TRAINING")
    logger.info("=" * 60)
    
    # 1. Load dataset
    logger.info("\n[1/6] Loading dataset...")
    X, y = load_dataset(augment_minority=True)
    
    # 2. Train/val/test split
    logger.info("\n[2/6] Splitting dataset...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_SEED, stratify=y_train_val
    )
    
    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    # 3. Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")
    
    # 4. Build model
    logger.info("\n[3/6] Building model...")
    input_shape = (N_MFCC, TARGET_LENGTH, 1)
    model = build_model(input_shape, num_classes=len(CLASS_LABELS))
    model.summary(print_fn=logger.info)
    
    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]
    
    # 6. Train
    logger.info("\n[4/6] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )
    
    # 7. Evaluate
    logger.info("\n[5/6] Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, digits=4)
    logger.info(f"\nClassification Report:\n{report}")
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # 8. Save results
    _save_training_plots(history, cm, RESULTS_DIR)
    
    # 9. Export models
    logger.info("\n[6/6] Exporting models...")
    
    # Save Keras model
    keras_path = os.path.join(MODEL_DIR, "border_intrusion_model.h5")
    model.save(keras_path)
    logger.info(f"Keras model saved: {keras_path}")
    
    # Export to TFLite
    tflite_path = os.path.join(MODEL_DIR, "border_intrusion_model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    logger.info(f"TFLite model saved: {tflite_path} ({len(tflite_model) / 1024:.1f} KB)")
    
    # Save training metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": int(X.shape[0]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_trained": len(history.history['loss']),
        "best_val_accuracy": float(max(history.history['val_accuracy'])),
        "class_labels": CLASS_LABELS,
        "input_shape": list(input_shape),
        "model_params": int(model.count_params()),
    }
    with open(os.path.join(MODEL_DIR, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TRAINING COMPLETE — Test Accuracy: {test_acc:.2%}")
    logger.info("=" * 60)
    
    return model, history, metadata


def _save_training_plots(history, cm, output_dir: str):
    """Save training visualization plots."""
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0e14')
    
    for ax in axes:
        ax.set_facecolor('#131920')
        ax.tick_params(colors='#c5c8c6')
        ax.xaxis.label.set_color('#c5c8c6')
        ax.yaxis.label.set_color('#c5c8c6')
        ax.title.set_color('#39FF14')
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], color='#39FF14', linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#FFB000', linewidth=2, label='Validation')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(facecolor='#131920', edgecolor='#39FF14', labelcolor='#c5c8c6')
    axes[0].grid(True, alpha=0.2, color='#39FF14')
    
    # Loss
    axes[1].plot(history.history['loss'], color='#39FF14', linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'], color='#FFB000', linewidth=2, label='Validation')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(facecolor='#131920', edgecolor='#39FF14', labelcolor='#c5c8c6')
    axes[1].grid(True, alpha=0.2, color='#39FF14')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight',
                facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0a0e14')
    ax.set_facecolor('#131920')
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='#0a0e14')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#39FF14')
    ax.set_xlabel('Predicted', color='#c5c8c6')
    ax.set_ylabel('Actual', color='#c5c8c6')
    ax.tick_params(colors='#c5c8c6')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight',
                facecolor='#0a0e14', edgecolor='none')
    plt.close()
    
    logger.info(f"Training plots saved to {output_dir}")


if __name__ == "__main__":
    train()
