"""
Fast-Track Plant Disease Detection Training - Windows Compatible
Uses Kaggle API or manual download to avoid Windows path issues
Trains only top layers for speed (~60-90 min on CPU)
"""

import tensorflow as tf
import json
import os
import numpy as np
from pathlib import Path
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
print("=" * 60)
print(" Plant Disease Detection - Fast Training Pipeline")
print("=" * 60)
dataset_dir = Path("PlantVillage")

if not dataset_dir.exists():
    print("\n  PlantVillage dataset not found locally")
    print("\n Downloading PlantVillage dataset from Kaggle...")
    print("\nTo download the dataset, please:")
    print("1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("2. Download the dataset")
    print("3. Extract to: backend/PlantVillage/")
    print("\nAlternatively, use Kaggle API:")
    print("   pip install kaggle")
    print("   kaggle datasets download -d abdallahalidev/plantvillage-dataset")
    print("   unzip plantvillage-dataset.zip -d PlantVillage")    
    try:
        import subprocess
        print("\n Attempting automatic download via Kaggle API...")        
        subprocess.run(["pip", "install", "kaggle", "-q"], check=True)
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "abdallahalidev/plantvillage-dataset"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(" Download complete, extracting...")
            import zipfile
            with zipfile.ZipFile("plantvillage-dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("PlantVillage")
            print(" Dataset ready!")
        else:
            print(f"\n Kaggle download failed: {result.stderr}")
            print("\nPlease download manually and re-run this script.")
            exit(1)
    except Exception as e:
        print(f"\n Auto-download failed: {e}")
        print("\nPlease download manually and re-run this script.")
        exit(1)
class_folders = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
class_names = [f.name for f in class_folders]
NUM_CLASSES = len(class_names)
print(f"\n Dataset found: {NUM_CLASSES} classes")
print(f"   Sample classes: {class_names[:3]}...")
def load_dataset(split='train'):
    """Load images from directory structure"""
    images = []
    labels = []
    print(f"\n Loading {split} data...")
    for label_idx, folder in enumerate(class_folders):
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
        n_train = int(len(image_files) * 0.8)
        if split == 'train':
            files = image_files[:n_train]
        else:
            files = image_files[n_train:]        
        for img_path in files:
            try:
                img = tf.keras.preprocessing.image.load_img(
                    str(img_path), 
                    target_size=IMG_SIZE
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                print(f"   Skipped {img_path.name}: {e}")
                continue
    print(f"   Loaded {len(images)} images")
    return np.array(images), np.array(labels)
X_train, y_train = load_dataset('train')
X_val, y_val = load_dataset('val')
print(f"\n Data loaded:")
print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
def train_model(base_model_fn, name, description):
    print(f"\n{'='*60}")
    print(f" Training {name}")
    print(f"   {description}")
    print(f"{'='*60}")    
    base = base_model_fn(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base.trainable = False      
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ], name=name)    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"\n📊 Model architecture:")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"   Total params: {model.count_params():,}")
    print(f"   Trainable params: {trainable_params:,}")    
    print(f"\n Training for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )    
    model_path = os.path.join(MODELS_DIR, f"{name.lower()}.h5")
    model.save(model_path)
    print(f" Saved to {model_path}")    
    val_acc = history.history['val_accuracy'][-1] * 100
    print(f" Final validation accuracy: {val_acc:.2f}%")
    return val_acc
print("\n" + "="*60)
print("Starting ensemble training (3 models)")
print("="*60)
accuracies = {}
#EfficientNetV2 
accuracies['EfficientNetV2'] = train_model(
    tf.keras.applications.EfficientNetV2B0,
    "EfficientNetV2",
    "High accuracy model - primary ensemble component"
)
#ResNet50V2 
accuracies['ResNet50V2'] = train_model(
    tf.keras.applications.ResNet50V2,
    "ResNet50V2",
    "Robust architecture - diverse ensemble component"
)
#MobileNetV3
accuracies['MobileNetV3'] = train_model(
    tf.keras.applications.MobileNetV3Large,
    "MobileNetV3",
    "Lightweight model - fast inference"
)
class_names_path = os.path.join(MODELS_DIR, "class_names.json")
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"\n Saved class names to {class_names_path}")
print("\n" + "="*60)
print(" TRAINING COMPLETE!")
print("="*60)
print("\n Final Results:")
for model, acc in accuracies.items():
    print(f"   {model:20s}: {acc:.2f}%")
ensemble_acc = sum(accuracies.values()) / len(accuracies)
print(f"\n   {'Ensemble (average)':20s}: {ensemble_acc:.2f}%")
print("\n Models saved in:", os.path.abspath(MODELS_DIR))
print("\n Next steps:")
print("   1. Run backend with: python main.py")
print("   2. Test predictions through your existing frontend")
print("   3. Models will now give REAL ML predictions!")
print("\n" + "="*60)
