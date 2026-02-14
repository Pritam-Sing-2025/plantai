"""
Quick PlantVillage Dataset Downloader
Uses direct download link without authentication
"""
import os
import zipfile
import urllib.request
import sys
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
def download_with_progress(url, filename):
    """Download file with progress indicator"""    
    class DownloadProgressBar:
        def __init__(self):
            self.pbar = None
        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                print(f" Downloading {filename}...")
                print(f"   Total size: {total_size / (1024*1024):.1f} MB")
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            bar_length = 40
            filled = int(bar_length * downloaded // total_size)
            bar = '█' * filled + '-' * (bar_length - filled)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f'\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='', flush=True)
    try:
        urllib.request.urlretrieve(url, filename, DownloadProgressBar())
        print("\n Download complete!")
        return True
    except Exception as e:
        print(f"\n Download failed: {e}")
        return False
def main():
    print("="*60)
    print(" PlantVillage Dataset Auto-Downloader")
    print("="*60)
    if os.path.exists("PlantVillage") and os.path.isdir("PlantVillage"):
        folders = [f for f in os.listdir("PlantVillage") if os.path.isdir(os.path.join("PlantVillage", f))]
        if len(folders) > 30:
            print("\n Dataset already exists!")
            print(f"   Found {len(folders)} class folders in PlantVillage/")
            print("\n Ready to train! Run: python train_models.py")
            return
    print("\n Downloading PlantVillage dataset...")
    print("   This will take 5-15 minutes depending on your connection")
    urls = [
        ("https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded", "plantvillage.zip"),
        ("https://github.com/spMohanty/PlantVillage-Dataset/archive/master.zip", "plantvillage-github.zip"),
    ]
    success = False
    for url, filename in urls:
        print(f"\n Trying source: {url[:60]}...")
        
        if download_with_progress(url, filename):
            success = True
            
            print(f"\n Extracting {filename}...")
            try:
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    if any('color' in f.lower() for f in file_list):
                        extract_path = "temp_extract"
                        zip_ref.extractall(extract_path)
                        import shutil
                        for root, dirs, files in os.walk(extract_path):
                            if 'color' in root.lower():
                                if os.path.exists("PlantVillage"):
                                    shutil.rmtree("PlantVillage")
                                shutil.move(root, "PlantVillage")
                                break
                        shutil.rmtree(extract_path)
                    else:
                        zip_ref.extractall("PlantVillage")
                print(" Extraction complete!")
                if os.path.exists(filename):
                     os.remove(filename)
                print(f"🧹 Cleaned up {filename}")
                if os.path.exists("PlantVillage"):
                    folders = [f for f in os.listdir("PlantVillage") if os.path.isdir(os.path.join("PlantVillage", f))]
                    print(f"\n Dataset ready! Found {len(folders)} class folders")
                    print("\n Next: Run training with:")
                    print("   python train_models.py")
                else:
                    print("\n  Extraction issue, trying next source...")
                    continue
                break
            except Exception as e:
                print(f" Extraction failed: {e}")
                continue
    if not success:
        print("\n" + "="*60)
        print(" Automatic download didn't work")
        print("="*60)
        print("\n Please download manually:")
        print("   1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("   2. Click 'Download' (needs free Kaggle account)")
        print("   3. Extract the zip")
        print("   4. Copy all class folders to: backend/PlantVillage/")
        print("\nExpected structure:")
        print("   PlantVillage/")
        print("   ├── Tomato___Early_blight/")
        print("   ├── Tomato___Late_blight/")
        print("   ├── Pepper__bell___Bacterial_spot/")
        print("   └── ... (38 folders total)")
        print("\n   Then run: python train_models.py")
if __name__ == "__main__":
    main()
