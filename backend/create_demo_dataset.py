"""
FAST DEMO VERSION - Creates minimal synthetic dataset for quick training
This gets you up and running in MINUTES instead of hours
Perfect for college demo - still uses real ML, just smaller dataset
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# PlantVillage classes (using subset for speed)
CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]

SAMPLES_PER_CLASS = 100  # Reduced for speed

def create_synthetic_plant_image(class_name, size=(256, 256)):
    """Create synthetic plant leaf image with disease patterns"""
    
    # Base green leaf
    img = Image.new('RGB', size, color=(50, 120, 50))
    draw = ImageDraw.Draw(img)
    if "___" in class_name:
        plant, disease = class_name.split("___")
        disease = disease.replace("_", " ")
    else:
        plant, disease = class_name, "healthy"
        if "healthy" in disease.lower():
    for y in range(size[1]):
            green_val = int(100 + (y / size[1]) * 80)
            draw.line([(0, y), (size[0], y)], fill=(30, green_val, 40))
    for i in range(5):
            x = random.randint(0, size[0])
            draw.line([(x, 0), (x + random.randint(-20, 20), size[1])], 
                     fill=(40, 140, 50), width=2)
    else:
        for y in range(size[1]):
            green_val = int(80 + (y / size[1]) * 50)
            draw.line([(0, y), (size[0], y)], fill=(green_val-20, green_val, 30))
        num_spots = random.randint(10, 30)
        for _ in range(num_spots):
            x = random.randint(0, size[0]-30)
            y = random.randint(0, size[1]-30)
            spot_size = random.randint(10, 30)
            if "blight" in disease.lower():
                color = (80, 60, 40)  
            elif "spot" in disease.lower():
                color = (60, 50, 30)  
            elif "virus" in disease.lower() or "curl" in disease.lower():
                color = (120, 100, 40)  
            else:
                color = (70, 55, 35)  
            draw.ellipse([x, y, x+ spot_size, y+spot_size], fill=color)
    pixels = np.array(img)
    noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    return img
def main():
    print("="*60)
    print(" FAST DEMO - Creating Synthetic Dataset")
    print("="*60)
    print("\n This creates a DEMO dataset for quick training")
    print("   - Uses synthetic images (not real PlantVillage)")
    print("   - Still trains REAL ML models")
    print("   - Perfect for college demo/testing")
    print("   - Training will take ~15-25 minutes!")
    
    if os.path.exists("PlantVillage"):
        print("\n PlantVillage folder already exists")
        response = input("   Delete and recreate with synthetic data? (y/n): ")
        if response.lower() != 'y':
            print("   Keeping existing dataset")
            return
        import shutil
        shutil.rmtree("PlantVillage")
    
    print(f"\n Creating {len(CLASSES)} classes...")
    print(f"   {SAMPLES_PER_CLASS} samples per class")
    print(f"   Total: {len(CLASSES) * SAMPLES_PER_CLASS} images")
    
    os.makedirs("PlantVillage", exist_ok=True)
    
    for class_name in CLASSES:
        class_dir = os.path.join("PlantVillage", class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"\n   Creating {class_name}...")
        for i in range(SAMPLES_PER_CLASS):
            img = create_synthetic_plant_image(class_name)
            img.save(os.path.join(class_dir, f"img_{i:04d}.jpg"))
            
            if (i + 1) % 20 == 0:
                print(f"      {i+1}/{SAMPLES_PER_CLASS}", end='\r')
        
        print(f"       {SAMPLES_PER_CLASS}/{SAMPLES_PER_CLASS}")
    
    print("\n" + "="*60)
    print(" Synthetic Dataset Created!")
    print("="*60)
    print(f"\n Dataset Statistics:")
    print(f"   Classes: {len(CLASSES)}")
    print(f"   Images per class: {SAMPLES_PER_CLASS}")
    print(f"   Total images: {len(CLASSES) * SAMPLES_PER_CLASS}")
    print(f"   Location: {os.path.abspath('PlantVillage')}")
    
    print("\n Next: Train models (15-25 min)")
    print("   python train_models.py")
    print("\n NOTE: This is synthetic data for DEMO purposes")
    print("   For production, download real PlantVillage dataset")

if __name__ == "__main__":
    main()
